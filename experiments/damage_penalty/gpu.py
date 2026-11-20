"""Pinned Qwen experiment, with fresh acquisition checks and full-model reinstalls."""

import argparse
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.damage_penalty.design import ARMS, REPLAY_ROLES, RULES, WAVES
from experiments.damage_penalty.inputs import SOURCE_WAVES
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.first_fatal.witness import run as reinstall
from experiments.health_baseline.live import execute as autonomous
from experiments.health_baseline.replay import execute as replay
from experiments.social_closed_loop.core import BodyValue


def run_study(root, stage, work, result_directories, verify_snapshots):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    models = {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }
    for name, digest in models.items():
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    count = sum(p.numel() for p in actor.model.parameters())
    fallback = None
    write(out / "status.json", {"stage": "checking_full_Qwen_against_cached_prefix"})
    try:
        backend.prepare()
    except NumericalMismatch as exc:
        backend.fast = False
        fallback = str(exc)
    write(
        out / "calibration.json",
        {"exact": backend.fast, "fallback": fallback, "counts": backend.stats.copy()},
    )

    try:
        result = work(out, backend, root, stage)
    except NumericalMismatch as exc:
        if not backend.fast:
            raise
        fallback = str(exc)
        backend.fast = False
        for directory in result_directories:
            if (out / directory).exists():
                (out / directory).rename(root / ("superseded-" + directory))
        if (root / "seed-backups").exists():
            (root / "seed-backups").rename(root / "superseded-seed-backups")
        write(
            out / "status.json",
            {"stage": "restarting_with_full_model", "fallback": fallback},
        )
        result = work(out, backend, root, stage)
    engine = "checked_immutable_prefix" if backend.fast else "ordinary_only"
    if verify_snapshots:
        backend.fast = False
        result["reinstall"] = reinstall(backend, root, out)
    actor.verify_unchanged()
    assert count == sum(p.numel() for p in actor.model.parameters())
    write(
        out / "receipt.json",
        dict(
            mode=stage,
            **result,
            engine=engine,
            counts=backend.stats,
            backbone_restored=True,
            original_params=count,
            fallback=fallback,
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
        {"stage": "complete", "completed": result["assigned_lives"]},
    )


def execute(out, backend, root, stage):
    if stage in WAVES:
        return autonomous(out, backend, BodyValue(), WAVES[stage], ARMS, RULES)
    return replay(out, backend, root, SOURCE_WAVES, "local_true", RULES, REPLAY_ROLES)


def run(root, stage):
    run_study(
        root,
        stage,
        execute,
        ("lives",) if stage in WAVES else ("replays",),
        stage == "dpcheck",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=(*WAVES, "dpreplay", "dpcheck"), required=True)
    v = p.parse_args()
    run(v.root, v.mode)
