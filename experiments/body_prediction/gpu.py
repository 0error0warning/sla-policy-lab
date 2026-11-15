"""Run paired frozen models with public evidence only; no body oracle."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from experiments.frozen_capability.gpu import digest, write
from experiments.frozen_capability.wire import strict_json
from experiments.self_future.gpu import Runner as SelfFutureRunner


class Runner(SelfFutureRunner):
    def __init__(self, root):
        import torch

        self.root, self.out = root, root / "output"
        self.out.mkdir(exist_ok=False)
        self.config = json.loads((root / "execution.json").read_bytes())
        self.rows = json.loads((root / "public-prompts.json").read_bytes())
        assert (
            digest(root / "public-prompts.json") == self.config["public_prompts_sha256"]
        )
        assert len(self.rows) == 102 and len({r["id"] for r in self.rows}) == 102
        assert not any(root.glob("*key*.json"))
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert "A100" in torch.cuda.get_device_name(0)
        assert torch.cuda.get_device_properties(0).total_memory > 35 * 1024**3
        assert torch.__version__ == self.config["torch_version"]
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.started, self.deadline = (
            time.monotonic(),
            float(os.environ["SLA_WORKER_DEADLINE"]),
        )
        self.paths, self.results, self.smokes, self.active = {}, [], [], {}
        write(
            self.out / "hardware.json",
            {
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
            },
        )
        write(self.out / "execution.json", self.config)

    def load(self, short):
        if short == "4B":
            return super().load(short)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.status("load", model=short)
        model, info = AutoModelForCausalLM.from_pretrained(
            self.paths[short],
            local_files_only=True,
            trust_remote_code=False,
            dtype=self.torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        assert not any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ), str(info)
        model.requires_grad_(False).eval()
        tok = AutoTokenizer.from_pretrained(
            self.paths[short], local_files_only=True, trust_remote_code=False
        )
        chat_end = tok.encode("<|im_end|>", add_special_tokens=False)
        assert len(chat_end) == 1 and tok.decode(chat_end) == "<|im_end|>"
        eos = model.generation_config.eos_token_id or tok.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        meta = {
            "loading_empty": True,
            "trainable_parameters": 0,
            "parameters": sum(p.numel() for p in model.parameters()),
            "eos_ids": sorted(set(eos + chat_end)),
            "configured_eos_ids": eos,
            "chat_end_id": chat_end[0],
            "open_id": None,
            "close_id": None,
            "generation_default": model.generation_config.to_dict(),
            "model_type": model.config.model_type,
        }
        write(self.out / short / "loading.json", meta)
        return model, tok, meta

    def run(self):
        self.download()
        self.inference_deadline = min(
            self.deadline, time.time() + self.config["inference_seconds"]
        )
        lookup = {r["id"]: r for r in self.rows}
        outcome = "infrastructure_failure"
        for short in self.config["models"]:
            model, tok, meta = self.load(short)
            versions = [p._version for p in model.parameters()]
            model_outcome = "infrastructure_failure"
            try:
                for job in self.config["jobs"]:
                    cap, seconds = self.config["budgets"][job["task"]]
                    if self.inference_deadline - time.time() < seconds + 15:
                        model_outcome = "budget_stop"
                        break
                    self.active = {
                        "model": short,
                        "case": job["id"],
                        "number": job["number"],
                        "task": job["task"],
                    }
                    self.status("generating", **self.active)
                    result = self.generate(
                        model,
                        tok,
                        meta,
                        lookup[job["id"]]["messages"],
                        max_tokens=cap,
                        seconds=seconds,
                    )
                    row = {**job, **result, "model": short, "thinking": False}
                    write(
                        self.out / short / "records" / f"{job['number']:03d}.json", row
                    )
                    self.results.append(
                        {
                            k: row[k]
                            for k in ("model", "number", "id", "task", "finish_reason")
                        }
                    )
                    write(self.out / "record-index.json", self.results)
                    self.status("case_saved", **self.active)
                    assert versions == [p._version for p in model.parameters()]
                    if job["task"] == "startup":
                        truth = (
                            {"ready": True}
                            if job["id"] == "startup-1"
                            else {"check": {"values": [7, 0, None], "flag": False}}
                        )
                        if (
                            result["completion_status"] != "completed"
                            or strict_json(result["final_text"]) != truth
                        ):
                            model_outcome = "startup_failed"
                            break
                else:
                    model_outcome = "complete"
            finally:
                write(
                    self.out / short / "unchanged.json",
                    {
                        "parameter_versions_unchanged": versions
                        == [p._version for p in model.parameters()],
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "gradients_absent": all(
                            p.grad is None for p in model.parameters()
                        ),
                        "outcome": model_outcome,
                    },
                )
                del model, tok
                self.clear()
            if model_outcome != "complete":
                outcome = model_outcome
                break
        else:
            outcome = "complete"
        write(
            self.out / "completed.json",
            {
                "outcome": outcome,
                "records": self.results,
                "all_parameters_frozen": True,
            },
        )
        self.status(outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runner = Runner(args.root)
    try:
        runner.run()
    except Exception as error:
        runner.status("stopped", error_type=type(error).__name__, reason=str(error))
        raise
