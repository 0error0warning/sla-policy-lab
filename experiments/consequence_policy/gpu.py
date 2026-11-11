"""Capture or verify native Qwen states. Run as a module from the repository root."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiments.consequence_policy.study import MODEL, REVISION, WORDS, contexts, messages
from experiments.direct_policy.protocol import write


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenQwen:
    def __init__(self, expected, out):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        write(out / "status.json", dict(stage="download"))
        snapshot = Path(snapshot_download(MODEL, revision=REVISION, token=False,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
        weights = json.loads(expected.read_bytes())
        assert weights["id"] == MODEL and weights["revision"] == REVISION
        assert weights["files"] == {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                                    for p in snapshot.glob("*.safetensors")}
        write(out / "weights-manifest.json", weights)
        write(out / "status.json", dict(stage="load"))
        self.tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, output_loading_info=True,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        assert not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        self.model.requires_grad_(False).eval()
        word_ids = [self.tok.encode(w, add_special_tokens=False) for w in WORDS]
        assert all(len(x) == 1 for x in word_ids) and len(set(x[0] for x in word_ids)) == 4, word_ids
        self.ids = [x[0] for x in word_ids]
        self.versions = [p._version for p in self.model.parameters()]

    def encode(self, row):
        text = self.tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        ids = self.tok.encode(text, add_special_tokens=False)
        assert len(ids) < 512
        for word, token in zip(WORDS, self.ids):
            assert self.tok.encode(text + word, add_special_tokens=False) == ids + [token]
        return ids

    def batch(self, prefixes):
        torch = self.torch
        assert len(set(map(len, prefixes))) == 1
        ids = torch.tensor(prefixes, device="cuda")
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            z = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1:, :]
            scores = self.model.lm_head(z)[:, 0, self.ids]
        return z[:, 0].float().cpu().numpy(), scores.float().cpu().numpy()

    def verify_unchanged(self):
        assert self.versions == [p._version for p in self.model.parameters()]
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())


def capture(actor, out):
    rows = contexts()
    prefixes = [actor.encode(r) for r in rows]
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    hidden = None
    logits = np.zeros((len(rows), 4), dtype=np.float32)
    done = 0
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), 8):
            indices = group[start:start + 8]
            z, scores = actor.batch([prefixes[i] for i in indices])
            if hidden is None:
                hidden = np.empty((len(rows), z.shape[-1]), dtype=np.float16)
            hidden[indices] = z
            assert np.array_equal(hidden[indices].astype(np.float32), z)
            logits[indices] = scores
            done += len(indices)
            if done % 32 < len(indices):
                write(out / "status.json", dict(stage="encode", finished=done, total=len(rows)))
    single = []
    for i in np.linspace(0, len(rows) - 1, 24, dtype=int):
        _, scores = actor.batch([prefixes[i]])
        single.append(dict(context=int(i), batch_logits=logits[i].tolist(), single_logits=scores[0].tolist()))
    np.savez_compressed(out / "qwen-states.npz", hidden=hidden, logits=logits)
    write(out / "contexts.json", [dict(**r, input_ids=p) for r, p in zip(rows, prefixes)])
    write(out / "singleton-check.json", single)
    return dict(contexts=len(rows), hidden=hidden.shape[1], length_range=[min(groups), max(groups)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["capture", "verify"], default="capture")
    args = parser.parse_args()
    out = args.root / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    actor = FrozenQwen(args.root / "expected-weights.json", out)
    if args.mode == "capture":
        summary = capture(actor, out)
    else:
        from experiments.consequence_policy.live import verify
        summary = verify(actor, args.root, out)
    actor.verify_unchanged()
    torch = actor.torch
    write(out / "receipt.json", dict(mode=args.mode, summary=summary, elapsed_seconds=time.monotonic() - start,
        words=WORDS, token_ids=actor.ids, hardware=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        peak_bytes=torch.cuda.max_memory_allocated(), python=platform.python_version(),
        versions={k: version(k) for k in ("numpy", "torch", "transformers", "huggingface_hub", "safetensors")},
        fixed_parameter_elements=sum(p.numel() for p in actor.model.parameters()), backbone_freeze_verified=True))
    write(out / "status.json", dict(stage="complete"))


if __name__ == "__main__":
    main()
