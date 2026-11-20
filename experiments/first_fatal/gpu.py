"""Pinned complete model and byte-checked Qwen computation for the two locked waves."""

import argparse
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import PILOT, REPLICATION
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.first_fatal.study import execute
from experiments.social_closed_loop.core import BodyValue


def train_wave(root, out, backend, wave):
    body = BodyValue()
    write(out / "status.json", {"stage": "native_equivalence"})
    fallback = None
    try:
        backend.prepare()
    except NumericalMismatch as e:
        fallback = str(e)
        backend.fast = False
    write(
        out / "calibration.json",
        {"exact": backend.fast, "fallback": fallback, "counts": backend.stats.copy()},
    )
    write(
        out / "status.json",
        {"stage": "calibrated", "exact_cache": backend.fast, "fallback": fallback},
    )
    try:
        result = execute(out, backend, body, wave)
    except NumericalMismatch as e:
        if not backend.fast:
            raise
        fallback = str(e)
        backend.fast = False
        (out / "lives").rename(root / "superseded-cache-execution")
        write(
            out / "status.json",
            {"stage": "restarting_with_ordinary_model", "fallback": fallback},
        )
        result = execute(out, backend, body, wave)
    return result, fallback


def run(root, mode):
    wave = {"ffpilot": PILOT, "ffconfirm": REPLICATION, "ffrestore": ()}[mode]
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }
    for name, h in source_hashes.items():
        assert sha256((root / name).read_bytes()).hexdigest() == h
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    count = sum(p.numel() for p in actor.model.parameters())
    if mode == "ffrestore":
        from experiments.first_fatal.witness import run as restore

        backend.fast = False
        result, fallback = restore(backend, root, out), None
    else:
        result, fallback = train_wave(root, out, backend, wave)
    actor.verify_unchanged()
    assert count == sum(p.numel() for p in actor.model.parameters())
    write(
        out / "receipt.json",
        {
            "mode": mode,
            **result,
            "engine": "identical_native_tail_and_complete"
            if backend.fast
            else "complete_only",
            "fallback": fallback,
            "counts": backend.stats,
            "backbone_restored": True,
            "original_params": count,
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "versions": {
                k: version(k) for k in ("transformers", "torch", "numpy", "scipy")
            },
        },
    )
    write(
        out / "status.json",
        {
            "stage": "complete",
            "completed": result.get("assigned_lives", result.get("restored_lives")),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", required=True)
    a = p.parse_args()
    run(a.root, a.mode)
