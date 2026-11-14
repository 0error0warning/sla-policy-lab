"""Frozen inference with completion, throughput and cache preflight gates."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from experiments.frozen_capability.gpu import Runner as BaseRunner
from experiments.frozen_capability.gpu import digest, write
from experiments.frozen_capability.scoring import assistant_message
from experiments.frozen_capability.wire import strict_json

from .contracts import full_case_fits, projected_seconds


class Runner(BaseRunner):
    def __init__(self, root):
        import torch

        self.root, self.out = root, root / "output"
        self.out.mkdir(exist_ok=False)
        self.config = json.loads((root / "execution.json").read_bytes())
        self.rows = json.loads((root / "public-prompts.json").read_bytes())
        assert len(self.rows) == 20 and len({r["id"] for r in self.rows}) == 20
        assert (
            digest(root / "public-prompts.json") == self.config["public_prompts_sha256"]
        )
        assert not (root / "evaluator-only-answer-key.json").exists()
        self.torch = torch
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert self.config["gpu_name_contains"] in torch.cuda.get_device_name(0)
        assert torch.__version__ == self.config["torch_version"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.started = time.monotonic()
        self.deadline = float(os.environ["SLA_WORKER_DEADLINE"])
        self.paths, self.results, self.smokes = {}, [], []
        self.active = {}
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

    def load(self, short):
        model, tok, meta = super().load(short)
        ids = tok.encode("<|im_end|>", add_special_tokens=False)
        assert len(ids) == 1 and tok.decode(ids) == "<|im_end|>"
        meta["chat_end_id"] = ids[0]
        meta["configured_eos_ids"] = meta["eos_ids"]
        meta["eos_ids"] = sorted(set(meta["eos_ids"] + ids))
        write(self.out / short / "loading.json", meta)
        return model, tok, meta

    def generate(
        self,
        model,
        tok,
        meta,
        messages,
        thinking,
        seed,
        max_tokens,
        seconds,
        min_tokens=0,
    ):
        from transformers import (
            GenerationConfig,
            StoppingCriteria,
            StoppingCriteriaList,
            set_seed,
        )

        torch = self.torch
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        assert prompt.endswith("<think>\n") == thinking
        inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        n_input = int(inputs.input_ids.shape[1])
        assert n_input + max_tokens <= 10000
        config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            eos_token_id=meta["eos_ids"],
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
        )
        set_seed(seed)
        torch.cuda.synchronize()
        started = time.monotonic()
        owner = self

        class Deadline(StoppingCriteria):
            expired = False
            cancelled = False
            first_64_seconds = None
            last_status_count = -1

            def __call__(self, input_ids, scores, **kwargs):
                count = int(input_ids.shape[1]) - n_input
                elapsed = time.monotonic() - started
                if count >= 64 and self.first_64_seconds is None:
                    self.first_64_seconds = elapsed
                self.expired = elapsed >= seconds
                self.cancelled = (owner.root / "STOP").exists()
                if count // 256 > self.last_status_count:
                    self.last_status_count = count // 256
                    owner.status(
                        "generating",
                        **owner.active,
                        thinking=thinking,
                        generated_tokens=count,
                        generation_seconds=elapsed,
                    )
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
        elapsed = time.monotonic() - started
        ids = seq[0, n_input:].tolist()
        reason = (
            "operator_stop"
            if stopper.cancelled
            else "eos"
            if ids and ids[-1] in meta["eos_ids"]
            else (
                "time_limit"
                if stopper.expired
                else "length_limit"
                if len(ids) >= max_tokens
                else "other_stop"
            )
        )
        final, channel, thoughts, boundary = assistant_message(
            ids,
            thinking,
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
            "elapsed_seconds": elapsed,
            "first_64_seconds": stopper.first_64_seconds,
            "finish_reason": reason,
            "time_limit_triggered": stopper.expired,
            "max_new_tokens": max_tokens,
            "max_seconds": seconds,
            "seed": seed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generation_config": config.to_dict(),
            "presence_penalty": 0.0,
        }

    def memory_probe(self, model, tok, short):
        torch = self.torch
        token = tok.encode(" sample", add_special_tokens=False)[0]
        ids = torch.full((1, 10000), token, dtype=torch.long, device="cuda")
        self.status("cache_preflight", model=short)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        assert output.logits.shape[1] == 1 and output.past_key_values is not None
        record = {
            "input_tokens": 10000,
            "last_position_logits_only": True,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del output, ids
        self.clear()
        write(self.out / short / "cache-preflight.json", record)

    def preflight(self):
        canary = [
            {
                "role": "user",
                "content": 'Return exactly the JSON object {"ready":true} as your final answer, without Markdown.',
            }
        ]
        probe = [
            {
                "role": "user",
                "content": "Continue the neutral sequence of integers separated by spaces: 1 2 3 4 5",
            }
        ]
        for short in self.config["models"]:
            if time.time() >= self.preflight_deadline:
                raise RuntimeError("preflight phase budget exceeded")
            model, tok, meta = self.load(short)
            before = [p._version for p in model.parameters()]
            self.active = {"model": short, "case": "startup-canary"}
            for mode in (False, True):
                left = self.preflight_deadline - time.time()
                if not full_case_fits(left, 240):
                    raise RuntimeError(
                        "not enough preflight time for complete canary budget"
                    )
                record = self.generate(
                    model, tok, meta, canary, mode, 918001, 2048, 240
                )
                write(self.out / short / f"canary-{mode}.json", record)
                if record["final_text"] is None or strict_json(
                    record["final_text"]
                ) != {"ready": True}:
                    raise RuntimeError(
                        f"{short} canary did not produce valid final JSON"
                    )
                if record["finish_reason"] != "eos":
                    raise RuntimeError("canary failed normal message termination")
            self.active = {"model": short, "case": "throughput-probe"}
            if self.preflight_deadline - time.time() < 120:
                raise RuntimeError("insufficient speed probe time")
            speed = self.generate(
                model, tok, meta, probe, False, 918002, 512, 120, min_tokens=512
            )
            write(self.out / short / "speed-preflight.json", speed)
            if speed["generated_tokens"] != 512 or speed["time_limit_triggered"]:
                raise RuntimeError("speed probe did not complete")
            needed = projected_seconds(
                64,
                speed["first_64_seconds"],
                448,
                speed["elapsed_seconds"] - speed["first_64_seconds"],
            )
            summary = {
                "model": short,
                "projected_8192_seconds_with_margin": needed,
                "per_case_seconds": 900,
                "throughput_passed": needed <= 900,
            }
            write(self.out / short / "preflight-summary.json", summary)
            if needed > 900:
                raise RuntimeError(
                    "8192 tokens not reachable under registered wall time"
                )
            self.memory_probe(model, tok, short)
            assert before == [p._version for p in model.parameters()]
            self.smokes.append(summary)
            del model, tok
            self.clear()
        if time.time() > self.preflight_deadline:
            raise RuntimeError("preflight phase budget exceeded")
        write(
            self.out / "preflight-passed.json",
            {"models": self.smokes, "all_passed": True},
        )

    def run(self):
        self.preflight_deadline = min(
            time.time() + self.config["preflight_seconds"],
            self.deadline - 2 * self.config.get("min_model_seconds", 4200) - 60,
        )
        if self.preflight_deadline <= time.time():
            raise RuntimeError("not enough allocated time for paired run")
        self.download()
        self.preflight()
        allowance = min(
            self.config["per_model_seconds"], (self.deadline - time.time() - 60) / 2
        )
        if allowance < self.config.get("min_model_seconds", 4200):
            raise RuntimeError("insufficient paired model time after preflight")
        write(self.out / "time-allocation.json", {"per_model_seconds": allowance})
        lookup = {r["id"]: r for r in self.rows}
        for short in self.config["models"]:
            started = time.monotonic()
            model, tok, meta = self.load(short)
            before = [p._version for p in model.parameters()]
            try:
                for job in self.config["jobs"]:
                    left = min(
                        allowance - (time.monotonic() - started),
                        self.deadline - time.time(),
                    )
                    if not full_case_fits(left):
                        raise RuntimeError(
                            "complete case no longer fits: stop, do not shorten"
                        )
                    row = lookup[job["id"]]
                    self.active = {
                        "model": short,
                        "case": row["id"],
                        "number": job["number"],
                    }
                    value = self.generate(
                        model,
                        tok,
                        meta,
                        row["messages"],
                        job["thinking"],
                        job["seed"],
                        8192,
                        900,
                    )
                    result = {
                        **job,
                        **value,
                        "model": short,
                        "arm": row["arm"],
                        "form": row["form"],
                        "state": row["state"],
                        "order": row["order"],
                    }
                    write(
                        self.out / short / "records" / f"{job['number']:03d}.json",
                        result,
                    )
                    self.results.append(
                        {
                            k: result[k]
                            for k in (
                                "number",
                                "model",
                                "id",
                                "thinking",
                                "finish_reason",
                                "generated_tokens",
                            )
                        }
                    )
                    self.status("case_saved", **self.active)
                    if value["finish_reason"] == "operator_stop":
                        raise RuntimeError(
                            "operator requested stop; raw generation preserved"
                        )
                    assert before == [p._version for p in model.parameters()]
                    assert all(
                        not p.requires_grad and p.grad is None
                        for p in model.parameters()
                    )
                    if job["thinking"] and value["time_limit_triggered"]:
                        raise RuntimeError(
                            "formal thinking wall time exceeded; stop further cases"
                        )
            finally:
                write(
                    self.out / short / "unchanged.json",
                    {
                        "parameter_versions_unchanged": before
                        == [p._version for p in model.parameters()],
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "gradients_absent": all(
                            p.grad is None for p in model.parameters()
                        ),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                del model, tok
                self.clear()
        write(
            self.out / "completed.json",
            {
                "calls": len(self.results),
                "records": self.results,
                "all_parameters_frozen": True,
                "seconds": time.monotonic() - self.started,
            },
        )
        self.status("complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozeninterface"], required=True)
    args = parser.parse_args()
    runner = Runner(args.root)
    try:
        runner.run()
    except Exception as error:
        runner.status("stopped", error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == "__main__":
    main()
