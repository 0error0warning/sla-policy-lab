"""Frozen direct Qwen; body/oracle/answer-key code is not packaged on GPU."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from experiments.frozen_capability.gpu import digest, write
from experiments.frozen_capability.wire import strict_json
from experiments.frozen_interface.gpu import Runner as PreviousRunner


class Runner(PreviousRunner):
    def __init__(self, root):
        import torch

        self.root, self.out = root, root / "output"
        self.out.mkdir(exist_ok=False)
        self.config = json.loads((root / "execution.json").read_bytes())
        self.rows = json.loads((root / "public-prompts.json").read_bytes())
        assert len(self.rows) == 200 and len({r["id"] for r in self.rows}) == 200
        assert (
            digest(root / "public-prompts.json") == self.config["public_prompts_sha256"]
        )
        assert not any(root.glob("*key*.json"))
        assert self.config["models"] == ["4B"] and self.config["thinking"] is False
        assert self.config["sampler"] == {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
        }
        self.torch = torch
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert "L4" in torch.cuda.get_device_name(0)
        assert torch.__version__ == self.config["torch_version"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.started = time.monotonic()
        self.deadline = float(os.environ["SLA_WORKER_DEADLINE"])
        self.paths, self.results, self.smokes, self.active = {}, [], [], {}
        self.hardware = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": {
                n: version(n)
                for n in (
                    "transformers",
                    "accelerate",
                    "safetensors",
                    "huggingface-hub",
                    "numpy",
                    "tokenizers",
                    "flash-linear-attention",
                    "fla-core",
                )
            },
            "model_dtype": "bfloat16",
            "batch_size": 1,
            "tf32": False,
            "deterministic_algorithms": True,
        }
        write(self.out / "hardware.json", self.hardware)
        write(self.out / "execution.json", self.config)

    def wait_gate(self):
        hashes = {
            r["id"]: digest(self.out / "4B/records" / f"{r['number']:03d}.json")
            for r in self.config["jobs"][:8]
        }
        write(self.out / "positive-ready.json", hashes)
        self.status("await_positive_gate")
        expires = min(self.deadline - 300, time.time() + 300)
        while time.time() < expires:
            path = self.root / "positive-gate.json"
            if path.exists():
                gate = json.loads(path.read_bytes())
                assert gate["records"] == hashes
                assert (
                    gate["public_prompts_sha256"]
                    == self.config["public_prompts_sha256"]
                )
                assert type(gate["passed"]) is bool
                write(self.out / "positive-gate.json", gate)
                return gate["passed"]
            if (self.root / "STOP").exists():
                break
            time.sleep(1)
        raise RuntimeError("external positive gate unavailable within budget")

    def run(self):
        self.download()
        model, tok, meta = self.load("4B")
        versions = [p._version for p in model.parameters()]
        outcome = "infrastructure_failure"
        try:
            if self.deadline - time.time() < 600:
                raise RuntimeError("insufficient budget after model loading")
            self.active = {"model": "4B", "case": "startup-canary"}
            canary = self.generate(
                model,
                tok,
                meta,
                [
                    {
                        "role": "user",
                        "content": 'Return exactly {"ready":true} as your final answer without Markdown.',
                    }
                ],
                False,
                926001,
                128,
                60,
            )
            write(self.out / "4B/canary.json", canary)
            if (
                canary["finish_reason"] != "eos"
                or canary["final_text"] is None
                or strict_json(canary["final_text"]) != {"ready": True}
            ):
                raise RuntimeError("startup canary failed normal final answer")
            lookup = {r["id"]: r for r in self.rows}
            jobs = self.config["jobs"]
            for offset in range(0, len(jobs), 4):
                if offset == 8 and not self.wait_gate():
                    outcome = "positive_failed"
                    break
                if self.deadline - time.time() < 4 * 60 + 15:
                    outcome = "budget_stop"
                    break
                for job in jobs[offset : offset + 4]:
                    self.active = {
                        "model": "4B",
                        "case": job["id"],
                        "number": job["number"],
                    }
                    r = self.generate(
                        model,
                        tok,
                        meta,
                        lookup[job["id"]]["messages"],
                        False,
                        job["seed"],
                        128,
                        60,
                    )
                    row = {**job, **r, "model": "4B", "thinking": False}
                    write(self.out / "4B/records" / f"{job['number']:03d}.json", row)
                    self.results.append(
                        {
                            k: row[k]
                            for k in ("number", "id", "arm", "seed", "finish_reason")
                        }
                    )
                    self.status("case_saved", **self.active)
                    assert versions == [p._version for p in model.parameters()]
                    if r["finish_reason"] == "operator_stop":
                        raise RuntimeError("operator stop")
            else:
                outcome = "complete"
            write(
                self.out / "completed.json",
                {
                    "outcome": outcome,
                    "records": self.results,
                    "calls": len(self.results),
                    "probe_calls": max(0, len(self.results) - 8),
                    "all_parameters_frozen": True,
                },
            )
            self.status(outcome)
        finally:
            write(
                self.out / "4B/unchanged.json",
                {
                    "parameter_versions_unchanged": versions
                    == [p._version for p in model.parameters()],
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "gradients_absent": all(p.grad is None for p in model.parameters()),
                    "elapsed_seconds": time.monotonic() - self.started,
                    "outcome": outcome,
                },
            )
            del model, tok
            self.clear()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["transparentbody"], required=True)
    args = p.parse_args()
    runner = Runner(args.root)
    try:
        runner.run()
    except Exception as error:
        runner.status("stopped", error_type=type(error).__name__, reason=str(error))
        raise
