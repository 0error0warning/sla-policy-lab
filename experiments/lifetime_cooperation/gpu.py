"""Run the preregistered lives using the original Qwen backbone and private online LoRA."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.individual_life.model import derivatives, update
from experiments.social_closed_loop.core import BodyValue
from .design import ITEMS, LEVELS, PANEL, WAVES, death_matrix, schedule
from .life import run_life
from .simple import LinearGroup


class PrivateQwen:
    def __init__(self, engine, learning):
        self.engine = engine
        self.current = engine.new("population")
        self.wrong = engine.new("population")
        self.birth = self.current.snapshot()
        self.training = learning
        engine.bind(self.current, self.wrong, None)
        self.calls = 0

    def slots(self):
        real = self.current.live()
        wrong = self.wrong.live()
        return [self.birth, real, self.birth, wrong, real, wrong]

    def decision(self, i, h):
        self.calls += 1
        # Every physical action and every update uses the complete backbone.
        with torch.set_grad_enabled(self.training):
            self.forward = self.engine.ordinary(i, h, self.slots())
        return self.forward.detach().cpu().numpy()[[1, 2, 3]]

    def learn(self, i, h, a, bad, signal, behavior):
        pi = self.forward.detach().cpu().numpy()
        requests = [(1, a, self.current), (3, bad, self.wrong)]
        gradients = derivatives(self.forward, requests)
        self.forward = None
        result = []
        for (slot, label, person), g in zip(requests, gradients, strict=True):
            result.append(
                update(
                    person, g, signal, float(behavior[label]), float(pi[slot, label])
                )
            )
        return result

    def weights(self):
        def array(pair):
            return np.concatenate(
                [t.detach().cpu().numpy().ravel() for t in pair]
            ).astype("<f4", copy=False)

        return {
            "current": array(self.current.live()),
            "wrong": array(self.wrong.live()),
            "birth": array(self.birth),
        }

    @torch.no_grad()
    def probe(self):
        def get_panel():
            probabilities = np.empty(
                (3, len(PANEL), len(LEVELS), PANEL.shape[1], 4), dtype=np.float32
            )
            weights = self.slots()
            for w, indices in enumerate(PANEL):
                for j, h in enumerate(LEVELS):
                    for n, i in enumerate(indices):
                        # One complete-forward paired validation per wording and health per checkpoint.
                        p = self.engine.panel(
                            int(i),
                            h,
                            weights,
                            check=n == (self.calls + j + w) % len(indices),
                        )
                        probabilities[:, w, j, n] = p[[1, 2, 3]].cpu().numpy()
            return probabilities

        try:
            return get_panel()
        except NumericalMismatch as error:
            self.engine.fast = False
            self.engine.fallbacks.append(str(error))
            return get_panel()


def run(root, mode):
    out = root / "output"
    out.mkdir(exist_ok=True, parents=True)
    for f, h in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / f).read_bytes()).hexdigest() == h
    if mode == "lcconfirm":
        pilot = json.loads((root / "pilot-gate.json").read_bytes())
        assert pilot["passed"] and pilot["seeds"] == list(WAVES["lcpilot"])
    start = time.monotonic()
    death_matrix()
    body = BodyValue()
    actor = FrozenQwen(root / "expected-weights.json", out)
    engine = Qwen(actor, root)
    engine.fallbacks = []
    ids = sorted(set(PANEL.ravel().tolist()))
    write(
        out / "status.json",
        {"stage": "calibrating_fixed_task_pairs", "contexts": len(ids)},
    )
    try:
        engine.prepare(ids)
    except NumericalMismatch as error:
        engine.fast = False
        engine.fallbacks.append(str(error))
    rows = [
        {k: r[k] for k in ("partner", "frame", "task", "wording", "layout", "text")}
        for r in ITEMS
    ]
    write(
        out / "contexts.json", {"rows": rows, "panel": PANEL.tolist(), "health": LEVELS}
    )
    before = sum(p.numel() for p in actor.model.parameters())
    completed = 0
    for seed in WAVES[mode]:
        tape = schedule(seed)
        for world in ("stable", "changed"):
            for kind, learning in (
                ("qwen", True),
                ("qwen", False),
                ("simple", True),
                ("simple", False),
            ):
                name = kind + ("_learn" if learning else "_frozen")
                write(
                    out / "status.json",
                    dict(
                        stage="living",
                        seed=seed,
                        world=world,
                        arm=name,
                        completed=completed,
                    ),
                )
                directory = out / "lives" / f"s{seed}" / world / name
                if kind == "qwen":
                    person = PrivateQwen(engine, learning)
                    try:
                        run_life(
                            directory, person, kind, learning, seed, world, tape, body
                        )
                    finally:
                        engine.close()
                else:
                    run_life(
                        directory,
                        LinearGroup(),
                        kind,
                        learning,
                        seed,
                        world,
                        tape,
                        body,
                    )
                completed += 1
    actor.verify_unchanged()
    assert before == sum(p.numel() for p in actor.model.parameters())
    write(
        out / "receipt.json",
        dict(
            mode=mode,
            seeds=WAVES[mode],
            lives=completed,
            backbone_frozen=True,
            backbone_parameters=before,
            full_physical_decisions=True,
            gpu=torch.cuda.get_device_name(),
            counts=engine.stats,
            fallback=engine.fallbacks,
            wall_seconds=time.monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            versions={
                k: version(k) for k in ("numpy", "torch", "transformers", "scipy")
            },
        ),
    )
    write(out / "status.json", {"stage": "complete", "lives": completed})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=list(WAVES))
    args = p.parse_args()
    run(args.root, args.mode)
