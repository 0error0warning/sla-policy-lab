"""Pinned full-Qwen acquisition and ordinary closed-loop deployment."""

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.individual_life.design import rows
from experiments.internal_body.capture import DecisionState, encode
from experiments.lethal_boundary.gpu import pack_bf16


def capture(actor, root, out):
    items = rows()
    ids = [encode(actor, r) for r in items]
    proj = torch.tensor(np.load(root / "architecture.npz")["projection"], device="cuda")
    lm = actor.model.model.language_model
    keep = {}

    def skip(_, args):
        keep["skip"] = args[0][:, -1].detach()

    def down(_, args, res):
        keep["down"] = res[:, -1].detach()

    def x(_, args):
        keep["x"] = args[0][:, -1].float() @ proj

    handles = [
        lm.layers[-1].post_attention_layernorm.register_forward_pre_hook(skip),
        lm.layers[-1].mlp.down_proj.register_forward_hook(down),
        lm.layers[-1].mlp.down_proj.register_forward_pre_hook(x),
    ]
    reader = DecisionState(actor)
    data = {}
    repeat_max = 0.0
    try:
        with torch.inference_mode():
            for i, it in enumerate(ids):
                value = reader.batch([it] * 6, include_down=False)
                for k in ("n", "r", "z", "skip", "down", "x"):
                    v = (
                        value["native"]
                        if k == "n"
                        else (
                            value[k]
                            if k in ("r", "z")
                            else keep[k].float().cpu().numpy()
                        )
                    )
                    repeat_max = max(repeat_max, float(np.abs(v - v[0]).max()))
                    row = v[0].copy() if k in ("x", "n") else pack_bf16(v[0])
                    data.setdefault(k, []).append(row)
                if i % 64 == 63:
                    write(
                        out / "status.json",
                        {"stage": "frozen_text_only", "texts": i + 1},
                    )
    finally:
        for h in handles:
            h.remove()
        reader.close()
    assert repeat_max < 2e-5
    np.savez_compressed(out / "prefix.npz", **{k: np.stack(v) for k, v in data.items()})
    write(out / "requests.json", [dict(**r, ids=t) for r, t in zip(items, ids)])
    return {
        "contexts": len(ids),
        "identical_row_max_error": repeat_max,
        "contains_outcomes": False,
        "contains_health": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", required=True)
    a = p.parse_args()
    out = a.root / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    actor = FrozenQwen(a.root / "expected-weights.json", out)
    if a.mode == "lifecapture":
        result = capture(actor, a.root, out)
    elif a.mode in ("lifepilot", "lifeconfirm"):
        from experiments.individual_life.run import hardware

        result = hardware(actor, a.root, out)
    else:
        raise ValueError(a.mode)
    actor.verify_unchanged()
    write(
        out / "receipt.json",
        {
            "mode": a.mode,
            "result": result,
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "backbone_restored": True,
            "model_shards_verified": True,
            "device": torch.cuda.get_device_name(),
            "versions": {
                k: version(k) for k in ("torch", "transformers", "numpy", "scipy")
            },
        },
    )
    write(out / "status.json", {"stage": "complete"})


if __name__ == "__main__":
    main()
