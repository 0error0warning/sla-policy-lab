"""Frozen capability inference. Reads public prompts only; no teacher or scoring."""

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from .wire import final_tokens


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    tmp.replace(path)


class Runner:
    def __init__(self, root):
        import torch

        self.root = root
        self.out = root / "output"
        self.out.mkdir(exist_ok=False)
        self.config = json.loads((root / "execution.json").read_bytes())
        self.rows = json.loads((root / "public-prompts.json").read_bytes())
        assert len(self.rows) == 64 and len({r["id"] for r in self.rows}) == 64
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
        self.paths = {}
        self.results = []
        self.smokes = []
        self.hardware = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": {
                name: version(name)
                for name in (
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

    def status(self, stage, **kw):
        write(
            self.out / "status.json",
            {
                "stage": stage,
                "finished": len(self.results),
                "elapsed_seconds": time.monotonic() - self.started,
                **kw,
            },
        )

    def download(self):
        from huggingface_hub import snapshot_download

        for short in self.config["models"]:
            self.status("download", model=short)
            expected = json.loads((self.root / f"expected-{short}.json").read_bytes())
            dest = Path(
                snapshot_download(
                    expected["id"],
                    revision=expected["revision"],
                    token=False,
                    allow_patterns=list(expected["files"]) + list(expected["metadata"]),
                )
            )
            for name, row in {**expected["files"], **expected["metadata"]}.items():
                path = dest / name
                assert (
                    path.stat().st_size == row["bytes"]
                    and digest(path) == row["sha256"]
                ), name
            self.paths[short] = dest
            write(self.out / short / "weights-manifest.json", expected)

    def load(self, short):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.status("load", model=short)
        torch = self.torch
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.paths[short],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
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
        eos = model.generation_config.eos_token_id or tok.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        for text in ("<think>", "</think>"):
            assert len(tok.encode(text, add_special_tokens=False)) == 1
        record = {
            "loading_empty": True,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "eos_ids": eos,
            "open_id": tok.encode("<think>", add_special_tokens=False)[0],
            "close_id": tok.encode("</think>", add_special_tokens=False)[0],
            "generation_default": model.generation_config.to_dict(),
            "model_type": model.config.model_type,
        }
        write(self.out / short / "loading.json", record)
        return model, tok, record

    def generate(self, model, tok, meta, messages, thinking, seed, max_tokens, seconds):
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
        if thinking:
            assert prompt.endswith("<think>\n"), repr(prompt[-100:])
        else:
            assert not prompt.endswith("<think>\n"), repr(prompt[-100:])
        inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        assert inputs.input_ids.shape[1] < 4096
        config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=max_tokens,
            eos_token_id=meta["eos_ids"],
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            use_cache=True,
            num_beams=1,
            num_return_sequences=1,
        )
        # Presence penalty 0 is exactly the identity; no extra logits processor.
        set_seed(seed)
        torch.cuda.synchronize()
        started = time.monotonic()

        class Deadline(StoppingCriteria):
            expired = False

            def __call__(self, input_ids, scores, **kwargs):
                self.expired = time.monotonic() - started >= seconds
                return torch.full(
                    (input_ids.shape[0],),
                    self.expired,
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
        ids = seq[0, inputs.input_ids.shape[1] :].tolist()
        ends = bool(ids and ids[-1] in meta["eos_ids"])
        reason = (
            "eos"
            if ends
            else "time_limit"
            if stopper.expired
            else "length_limit"
            if len(ids) >= max_tokens
            else "other_stop"
        )
        final, channel, thought_count = final_tokens(
            ids, thinking, meta["close_id"], meta["open_id"], meta["eos_ids"]
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
            "thinking_tokens": thought_count,
            "generated_tokens": len(ids),
            "input_tokens": int(inputs.input_ids.shape[1]),
            "elapsed_seconds": elapsed,
            "finish_reason": reason,
            "time_limit_triggered": stopper.expired,
            "max_new_tokens": max_tokens,
            "max_seconds": seconds,
            "seed": seed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generation_config": config.to_dict(),
            "presence_penalty": 0.0,
        }

    def clear(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def smoke(self):
        message = [
            {
                "role": "user",
                "content": "Return a JSON object with one key ready and the boolean value true.",
            }
        ]
        for short in self.config["models"]:
            model, tok, meta = self.load(short)
            before = [p._version for p in model.parameters()]
            for mode in (False, True):
                self.status("smoke", model=short, thinking=mode)
                result = self.generate(model, tok, meta, message, mode, 814000, 32, 120)
                write(self.out / short / f"smoke-{mode}.json", result)
                assert result["generated_tokens"] > 0
                self.smokes.append(
                    {
                        "model": short,
                        "thinking": mode,
                        "seconds": result["elapsed_seconds"],
                        "tokens": result["generated_tokens"],
                    }
                )
            assert before == [p._version for p in model.parameters()]
            del model, tok
            self.clear()
        write(self.out / "smoke-summary.json", self.smokes)

    def run(self):
        self.download()
        self.smoke()
        remaining = max(0.0, self.deadline - time.time() - 90)
        allowance = min(self.config["per_model_seconds"], remaining / 2)
        write(
            self.out / "time-allocation.json",
            {
                "per_model_seconds": allowance,
                "available_after_preflight_seconds": remaining,
            },
        )
        lookup = {r["id"]: r for r in self.rows}
        for short in self.config["models"]:
            model_started = time.monotonic()
            model, tok, meta = self.load(short)
            before = [p._version for p in model.parameters()]
            for job in self.config["jobs"]:
                row = lookup[job["id"]]
                mode = job["thinking"]
                prefix = {
                    **job,
                    "model": short,
                    "arm": row["arm"],
                    "state": row["state"],
                    "order": row["order"],
                }
                left = min(
                    allowance - (time.monotonic() - model_started),
                    self.deadline - time.time() - 30,
                )
                if left < self.config["per_case_seconds"] + 5:
                    result = {
                        **prefix,
                        "finish_reason": "unrun_budget",
                        "generated_tokens": 0,
                    }
                else:
                    self.status(
                        "inference",
                        model=short,
                        thinking=mode,
                        case=row["id"],
                        remaining_model_seconds=left,
                    )
                    value = self.generate(
                        model,
                        tok,
                        meta,
                        row["messages"],
                        mode,
                        job["seed"],
                        self.config["max_new_tokens"],
                        self.config["per_case_seconds"],
                    )
                    result = {**prefix, **value}
                path = self.out / short / "records" / f"{job['number']:03d}.json"
                assert not path.exists()
                write(path, result)
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
                assert before == [p._version for p in model.parameters()]
                assert all(
                    not p.requires_grad and p.grad is None for p in model.parameters()
                )
            write(
                self.out / short / "unchanged.json",
                {
                    "parameter_versions_unchanged": True,
                    "trainable_parameters": 0,
                    "gradients_absent": True,
                    "elapsed_seconds": time.monotonic() - model_started,
                },
            )
            del model, tok
            self.clear()
        write(
            self.out / "completed.json",
            {
                "records": self.results,
                "seconds": time.monotonic() - self.started,
                "calls": len(self.results),
                "all_parameters_frozen": True,
            },
        )
        self.status("complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozencapability"], required=True)
    args = parser.parse_args()
    Runner(args.root).run()


if __name__ == "__main__":
    main()
