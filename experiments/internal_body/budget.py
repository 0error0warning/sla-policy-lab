"""Separate distinct experience coverage, minibatch variability and update count."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.study import KL, TEST_H
from experiments.direct_policy.protocol import rng, write
from experiments.internal_body.depth import schedule
from experiments.internal_body.design import LR_A, LR_B, SPLIT_H
from experiments.internal_body.population import acquire
from experiments.internal_body.suffix import (
    InternalPolicy,
    StateBank,
    arr,
    distribution,
)
from experiments.social_closed_loop.core import BodyValue

SEEDS = tuple(range(64811, 64823))
ARMS = (
    "mini32",
    "balanced32",
    "pooled256",
    "full32",
    "repeat128",
    "fresh128",
    "full75",
)


def make_schedules(seed, rows):
    prefixes = [r["ids"] for r in rows[:768]]
    original = schedule(seed, prefixes)
    fresh = list(original)
    for j in range(1, 4):
        fresh.extend(schedule(seed + 10000 * j, prefixes))
    gen = rng(seed, "balanced-eight-state-steps")
    available = {
        (p, f): np.array(
            [
                i
                for i, r in enumerate(rows[:768])
                if r["partner"] == p and r["frame"] == f
            ]
        )
        for p in (0, 1)
        for f in (0, 1)
    }
    balanced = []
    for t in range(32):
        hs = gen.permutation(8)
        ix = np.array(
            [
                gen.choice(available[(int((t + h // 2) % 2), int((t // 2 + h) % 2))])
                for h in hs
            ]
        )
        balanced.append((ix, hs))
    union = (
        np.concatenate([x for x, _ in original]),
        np.concatenate([h for _, h in original]),
    )
    return {
        "mini32": original,
        "balanced32": balanced,
        "pooled256": [union] * 32,
        "full32": [None] * 32,
        "repeat128": original * 4,
        "fresh128": fresh,
        "full75": [None] * 75,
    }


def train(model, grid, scores, seq):
    opt = torch.optim.Adam(
        [{"params": [model.A], "lr": LR_A}, {"params": [model.B], "lr": LR_B}]
    )
    base = distribution(grid["n"]).detach()
    val = arr(scores, base.device)
    hist = []
    for indices in seq:
        opt.zero_grad(set_to_none=True)
        if indices is None:
            pi = distribution(model.scores(grid))
            prior = base
            feedback = val
        else:
            ix, hs = indices
            # Restrict context columns and reuse the same eight bodily basis vectors.
            subset = {
                k: (
                    v[:, ix]
                    if k in ("n", "r", "sq", "div", "dot", "x")
                    else (v[ix] if k == "partner" else v)
                )
                for k, v in grid.items()
            }
            pi = distribution(model.scores(subset))[hs, np.arange(len(ix))]
            prior = base[0, ix]
            feedback = val[hs, ix]
        gain = (pi * feedback).sum(-1).mean()
        kl = (pi * (pi / prior).log()).sum(-1).mean()
        obj = -gain + KL * kl
        obj.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()
        hist.append(
            [float(obj.detach()), float(gain.detach()), float(kl.detach()), float(gn)]
        )
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    bank = StateBank(a.cache)
    body = BodyValue()
    tr = np.arange(768)
    te = np.arange(768, 960)
    grid = bank.grid(tr, SPLIT_H)
    test = bank.grid(te, TEST_H)
    ref = [evaluation_reference(body, [bank.rows[i] for i in te], m) for m in (0, 1)]
    for s in SEEDS:
        start = time.monotonic()
        scores = acquire(s, bank, tr, body, a.output)
        seq = make_schedules(s, bank.rows)
        schedules = {}
        for k, steps in seq.items():
            if steps[0] is not None:
                schedules[k] = [[x.tolist(), h.tolist()] for x, h in steps]
        write(a.output / f"schedule-{s}.json", schedules)
        for m in (0, 1):
            folder = a.output / f"s{s}-m{m}"
            results = {}
            probes = {}
            losses = {}
            for kind in ARMS:
                model = InternalPolicy(bank, "linear")
                losses[kind] = train(model, grid, scores[m]["real"], seq[kind])
                model.save(folder / (kind + ".npz"))
                with torch.no_grad():
                    pi = distribution(model.scores(test)).cpu().numpy()
                probes[kind] = pi
                results[kind] = metrics(pi, ref[m])
            np.savez_compressed(folder / "budget-probes.npz", **probes)
            write(folder / "budget-metrics.json", results)
            write(folder / "budget-steps.json", losses)
        print("budget seed", s, round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    main()
