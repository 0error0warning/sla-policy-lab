"""Run frozen self-future prompts; no body simulator or answer key on GPU."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from experiments.frozen_capability.gpu import digest, write
from experiments.frozen_capability.scoring import assistant_message
from experiments.frozen_capability.wire import strict_json
from experiments.frozen_interface.gpu import Runner as PreviousRunner


class Runner(PreviousRunner):
    def __init__(self, root):
        import torch

        self.root, self.out = root, root / "output"
        self.out.mkdir(exist_ok=False)
        self.config = json.loads((root / "execution.json").read_bytes())
        self.rows = json.loads((root / "public-prompts.json").read_bytes())
        assert (
            digest(root / "public-prompts.json") == self.config["public_prompts_sha256"]
        )
        assert len(self.rows) == len({r["id"] for r in self.rows})
        assert not any(root.glob("*key*.json"))
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

    def generate(self, model, tok, meta, messages):
        from transformers import (
            GenerationConfig,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        torch = self.torch
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert not prompt.endswith("<think>\n")
        inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        n_input = int(inputs.input_ids.shape[1])
        cap, seconds = self.config["max_new_tokens"], self.config["case_seconds"]
        assert n_input + cap <= 10000
        config = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=cap,
            eos_token_id=meta["eos_ids"],
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
        )
        torch.cuda.synchronize()
        started, owner = time.monotonic(), self

        class Deadline(StoppingCriteria):
            expired = False
            cancelled = False

            def __call__(self, input_ids, scores, **kwargs):
                self.expired = (
                    time.monotonic() - started >= seconds
                    or time.time() >= owner.inference_deadline
                )
                self.cancelled = (owner.root / "STOP").exists()
                return torch.full(
                    (input_ids.shape[0],),
                    self.expired or self.cancelled,
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        stopper = Deadline()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            seq = model.generate(
                **inputs,
                generation_config=config,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )
        torch.cuda.synchronize()
        ids = seq[0, n_input:].tolist()
        reason = (
            "operator_stop"
            if stopper.cancelled
            else "eos"
            if ids and ids[-1] in meta["eos_ids"]
            else "time_limit"
            if stopper.expired
            else "length_limit"
            if len(ids) >= cap
            else "other_stop"
        )
        final, channel, thoughts, boundary = assistant_message(
            ids,
            False,
            meta["close_id"],
            meta["open_id"],
            meta["eos_ids"],
            meta["chat_end_id"],
        )
        return {
            "prompt_text": prompt,
            "input_ids": inputs.input_ids[0].tolist(),
            "output_ids": ids,
            "output_text": tok.decode(ids, skip_special_tokens=False),
            "final_ids": final,
            "final_text": None
            if final is None
            else tok.decode(final, skip_special_tokens=False),
            "channel": channel,
            "thinking_tokens": thoughts,
            "message_end_position": boundary,
            "generated_tokens": len(ids),
            "input_tokens": n_input,
            "elapsed_seconds": time.monotonic() - started,
            "finish_reason": reason,
            "completion_status": "completed"
            if reason == "eos" and boundary is not None
            else reason
            if reason in ("time_limit", "length_limit")
            else "error",
            "time_limit_triggered": stopper.expired,
            "max_new_tokens": cap,
            "max_seconds": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generation_config": config.to_dict(),
            "presence_penalty": 0.0,
        }

    def wait_gate(self):
        hashes = {
            r["id"]: digest(self.out / "4B/records" / f"{r['number']:03d}.json")
            for r in self.results
            if r.get("split") == "development"
        }
        write(self.out / "development-ready.json", hashes)
        self.status("await_development_gate")
        expires = min(self.inference_deadline - 60, time.time() + 300)
        while time.time() < expires:
            path = self.root / "development-gate.json"
            if path.exists():
                gate = json.loads(path.read_bytes())
                assert gate["records"] == hashes
                assert (
                    gate["public_prompts_sha256"]
                    == self.config["public_prompts_sha256"]
                )
                assert type(gate["passed"]) is bool
                write(self.out / "development-gate.json", gate)
                return gate["passed"]
            if (self.root / "STOP").exists():
                break
            time.sleep(1)
        raise RuntimeError("external development gate unavailable within budget")

    def run(self):
        self.download()
        model, tok, meta = self.load("4B")
        versions = [p._version for p in model.parameters()]
        self.inference_deadline = min(self.deadline, time.time() + 1800)
        outcome = "infrastructure_failure"
        try:
            lookup = {r["id"]: r for r in self.rows}
            gate_done = False
            for job in self.config["jobs"]:
                if job["split"] == "confirmation" and not gate_done:
                    gate_done = True
                    if not self.wait_gate():
                        outcome = "development_failed"
                        break
                if (
                    self.inference_deadline - time.time()
                    < self.config["case_seconds"] + 15
                ):
                    outcome = "budget_stop"
                    break
                self.active = {
                    "model": "4B",
                    "case": job["id"],
                    "number": job["number"],
                }
                self.status("generating", **self.active)
                result = self.generate(model, tok, meta, lookup[job["id"]]["messages"])
                row = {**job, **result, "model": "4B", "thinking": False}
                write(self.out / "4B/records" / f"{job['number']:03d}.json", row)
                self.results.append(
                    {k: row[k] for k in ("number", "id", "split", "finish_reason")}
                )
                self.status("case_saved", **self.active)
                assert versions == [p._version for p in model.parameters()]
                if job["split"] == "startup":
                    expected = (
                        {"ready": True}
                        if job["id"] == "startup-1"
                        else {"check": {"values": [7, 0, None], "flag": False}}
                    )
                    if (
                        result["completion_status"] != "completed"
                        or strict_json(result["final_text"]) != expected
                    ):
                        outcome = "startup_failed"
                        break
            else:
                outcome = "complete"
            write(
                self.out / "completed.json",
                {
                    "outcome": outcome,
                    "records": self.results,
                    "calls": len(self.results),
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
                    "outcome": outcome,
                },
            )
            del model, tok
            self.clear()


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
