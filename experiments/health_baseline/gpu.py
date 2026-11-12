"""Run one locked stage on the pinned Qwen model, retaining the parent's precision checks."""

import argparse
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.health_baseline.design import WAVES


def run(root, mode):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    for name, h in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / name).read_bytes()).hexdigest() == h
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    count = sum(p.numel() for p in actor.model.parameters())
    fallback = None
    if mode == "hbverify":
        from experiments.first_fatal.witness import run as restore

        backend.fast = False
        result = restore(backend, root, out)
    else:
        if mode in ("hbreplay", "hbrepeat"):
            from experiments.health_baseline.replay import execute

            def work():
                return execute(out, backend, root)

            directory = "replays"
        else:
            from experiments.health_baseline.live import execute as lives

            from experiments.social_closed_loop.core import BodyValue

            def work():
                return lives(out, backend, BodyValue(), WAVES[mode])

            directory = "lives"
        write(out / "status.json", {"stage": "checking_native_forward_and_gradients"})
        try:
            backend.prepare()
        except NumericalMismatch as exc:
            fallback = str(exc)
            backend.fast = False
        write(
            out / "calibration.json",
            {
                "exact": backend.fast,
                "fallback": fallback,
                "counts": backend.stats.copy(),
            },
        )
        try:
            result = work()
        except NumericalMismatch as exc:
            if not backend.fast:
                raise
            fallback = str(exc)
            backend.fast = False
            (out / directory).rename(root / "superseded-cache-execution")
            backups = root / "seed-backups"
            if backups.exists():
                backups.rename(root / "superseded-seed-backups")
            write(
                out / "status.json",
                {"stage": "ordinary_model_restart", "reason": fallback},
            )
            result = work()
    actor.verify_unchanged()
    assert count == sum(p.numel() for p in actor.model.parameters())
    write(
        out / "receipt.json",
        dict(
            mode=mode,
            **result,
            engine="equivalent_native_tail" if backend.fast else "complete_only",
            fallback=fallback,
            counts=backend.stats,
            backbone_restored=True,
            original_params=count,
            elapsed_seconds=time.monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
            versions={
                k: version(k) for k in ("transformers", "torch", "numpy", "scipy")
            },
        ),
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
    p.add_argument(
        "--mode", choices=(*WAVES, "hbreplay", "hbrepeat", "hbverify"), required=True
    )
    a = p.parse_args()
    run(a.root, a.mode)
