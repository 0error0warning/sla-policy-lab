"""Read-only paired forecast replacements at each stage of the #21 factual chronology."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.design import PANEL, PILOT
from experiments.loreft_life.adapters import signature, vector
from experiments.partner_prediction.gpu import Participants, replay
from experiments.partner_prediction.model import Actor

from .design import (
    DENSE,
    HEALTH,
    PARENT_RUNS,
    STAGES,
    inputs,
    oracle,
    random_panel,
    turns,
)


class Probe:
    def __init__(self, actor):
        self.actor = actor
        self.comparisons = 0
        self.max_error = 0.0
        self.max_lane_error = 0.0
        self.fallback = []
        self.native_only = not actor.b.q.fast
        self.calls = 0

    @torch.no_grad()
    def read(self, model, i, h, vectors, check):
        actor = self.actor
        # Duplicate in reverse order to audit that the batch lane cannot explain a contrast.
        roles = actor.lanes([(model, v) for v in [*vectors, *reversed(vectors)]])
        self.calls += 1
        if self.native_only:
            p = actor.ordinary(i, h, roles)
        else:
            common = actor.common[i, h]
            raw = actor.inject(common.expand(6, -1), roles)
            p = actor.b.engine._read(i, raw)
            if check:
                full = actor.ordinary(i, h, roles)
                error = float((full - p).abs().max())
                self.max_error = max(self.max_error, error)
                self.comparisons += 1
                if error:
                    self.native_only = True
                    self.fallback.append([self.calls, i, h, error])
                p = full
        mirror = float((p[:3] - p[3:].flip(0)).abs().max())
        self.max_lane_error = max(self.max_lane_error, mirror)
        if mirror:
            raise RuntimeError(
                f"Identical forecast in reversed batch lane differed by {mirror}"
            )
        return p[:3].cpu().numpy()

    @torch.no_grad()
    def panel(self, model, seed, t, head, x, full):
        chances = oracle(seed)
        js = range(32) if full else DENSE
        ws = range(2) if full else (1,)
        hs = HEALTH if full else (6.0,)
        grid = np.empty((len(ws), len(hs), len(js), 3, 4), dtype=np.float32)
        f = np.empty((len(ws), len(js), 3, 3))
        for wi, w in enumerate(ws):
            for n, j in enumerate(js):
                i = int(PANEL[w, j])
                q = inputs(head.read(x, i), chances[w, j])
                f[wi, n] = q
                for hi, h in enumerate(hs):
                    check = n == (seed + t + 7 * hi + 3 * w) % len(js)
                    grid[wi, hi, n] = self.read(model, i, h, q, check)
        return grid, f


def one_history(actor, probe, folder, seed, source, previous):
    grp = Participants(actor)
    n = len(source) - 1
    chosen = turns(n)
    policy, heads, counts, health = [], [], [], []
    dense_p, dense_q, panels, forecast = [], [], [], []

    def observe(t, e, people):
        m = people.policies["true"]
        head = people.heads["true"]
        before = vector(m)
        phead = head.w.copy()
        policy.append(before)
        heads.append(phead)
        counts.append(head.updates)
        health.append(e["health"])
        full = t in chosen
        was_native = probe.native_only
        p, q = probe.panel(m, seed, t, head, actor.x, full)
        if not was_native and probe.native_only:
            p, q = probe.panel(m, seed, t, head, actor.x, full)
        if full:
            panels.append(p)
            forecast.append(q)
            dense_p.append(p[1, 0, DENSE])
            dense_q.append(q[1, DENSE])
        else:
            dense_p.append(p[0, 0])
            dense_q.append(q[0])
        assert np.array_equal(vector(m), before) and np.array_equal(head.w, phead)
        if t == 0:
            assert np.array_equal(p, np.broadcast_to(p[..., :1, :], p.shape))

    replay(actor, seed, grp, source, folder, before_decision=observe)
    panels, forecast = np.asarray(panels), np.asarray(forecast)
    weights = np.asarray(policy)
    actual = np.stack(
        [
            json.loads(line)["policy"]["true"]
            for line in (folder / "replay.jsonl").read_bytes().splitlines()
        ]
    )
    hashes = np.array([signature(w).encode() for w in weights])
    comparison = {}
    for tag, z in previous.items():
        k = f"s{seed}_"
        diff = abs(actual - z[k + "actual"])
        final = np.take(panels[-1], [0, 2], axis=-2).transpose(3, 0, 1, 2, 4)
        assert final.shape == (2, 2, 3, 32, 4)
        comparison[tag] = {
            "states": len(weights),
            "matching_parameter_states": int((hashes == z[k + "sha"]).sum()),
            "identical_last_parameters": bool(
                np.array_equal(weights[-1], z[k + "last_weights"])
            ),
            "actual_max_pp": float(diff.max() * 100),
            "actual_mean_pp": float(diff.mean() * 100),
            "terminal_panel_max_pp": float(
                100 * abs(final - z[k + "last_probs"]).max()
            ),
            "terminal_forecast_max_pp": float(
                100 * abs(forecast[-1, :, :, 0] - z[k + "last_forecast"]).max()
            ),
            "terminal_head_max_abs": float(abs(heads[-1] - z[k + "last_head"]).max()),
        }
    uniforms, body_noise = random_panel(seed)
    np.savez_compressed(
        folder / "states.npz",
        policy=weights,
        head=np.array(heads),
        health=health,
        observed=counts,
        selected=chosen,
        sha=hashes,
        source_decisions=actual,
        uniforms=uniforms,
        noise=body_noise,
    )
    np.savez_compressed(
        folder / "stages.npz",
        prob=panels,
        q=forecast,
        dense_prob=np.asarray(dense_p),
        dense_q=np.asarray(dense_q),
    )
    write(folder / "anchors.json", comparison)
    return comparison


def run(root, mode):
    assert mode == "forecaststages"
    start = time.monotonic()
    out = root / "output"
    out.mkdir(exist_ok=True)
    tapes = json.loads((root / "tapes.json").read_bytes())
    assert set(tapes) == set(map(str, PILOT))
    previous = {tag: np.load(root / (tag + "-anchors.npz")) for tag in PARENT_RUNS}
    frozen = FrozenQwen(root / "expected-weights.json", out)
    actor = Actor(frozen, root, out)
    probe = Probe(actor)
    try:
        for k, seed in enumerate(PILOT):
            folder = out / "lives" / f"s{seed}"
            folder.mkdir(parents=True)
            write(
                out / "status.json",
                {"stage": "stage_and_dense_probes", "seed": seed, "completed": k},
            )
            one_history(actor, probe, folder, seed, tapes[str(seed)], previous)
    finally:
        actor.b.close()
        for z in previous.values():
            z.close()
    frozen.verify_unchanged()
    write(
        out / "receipt.json",
        {
            "mode": mode,
            "seeds": PILOT,
            "stages": STAGES,
            "health": HEALTH,
            "device": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "full_panel_comparisons": probe.comparisons,
            "max_native_cache_error": probe.max_error,
            "max_identical_lane_error": probe.max_lane_error,
            "fallback": probe.fallback,
            "initial_fallback": actor.b.engine.fallback,
            "probe_calls": probe.calls,
            "qwen_stats": actor.b.q.stats,
            "frozen_backbone": True,
            "versions": {
                k: version(k) for k in ("torch", "numpy", "scipy", "transformers")
            },
        },
    )
    write(
        out / "status.json", {"stage": "complete", "histories": len(PILOT), "modes": 3}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["forecaststages"], required=True)
    a = p.parse_args()
    run(a.root, a.mode)
