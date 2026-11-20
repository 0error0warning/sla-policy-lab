"""Separate singleton-state drift from BF16 tail arithmetic for the nonfatal coverage pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.study import TEST_H
from experiments.direct_policy.protocol import write
from experiments.internal_body.evidence import stat
from experiments.internal_body.quantized import Tail
from experiments.internal_body.suffix import InternalPolicy, StateBank, distribution
from experiments.social_closed_loop.core import BodyValue


def gap(a, b):
    x = np.abs(a - b).max(axis=-1)
    return {
        "mean": float(x.mean()),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def run(cache, fresh, study, out):
    tail = Tail(fresh, cache)
    bank = StateBank(cache)
    ix = np.arange(768, 960)
    ref = [
        evaluation_reference(BodyValue(), [bank.rows[i] for i in ix], m) for m in (0, 1)
    ]
    original = bank.grid(ix, TEST_H)
    with torch.no_grad():
        old = distribution(original["n"]).numpy()[0]
        new = distribution(tail.plain["native"]).numpy()[0]
    for field, state in (("x", "features"), ("r", "r"), ("native", "native")):
        getattr(bank, field)[ix] = tail.novel[state][:192]
    bank.grids.clear()
    singleton = bank.grid(ix, TEST_H)
    records = {}
    raw = {"original_frozen": old, "singleton_frozen": new}
    paired = []
    seeds = json.loads((study / "pilot.json").read_bytes())["seeds"]
    for seed in seeds:
        worlds = []
        for m in (0, 1):
            path = study / f"coverage-{seed}-m{m}"
            worldscores = {}
            with np.load(path / "probes.npz") as archived:
                for arm in ("high", "nonfatal"):
                    actor = InternalPolicy.load(bank, path / (arm + ".npz"))
                    with np.load(path / (arm + ".npz")) as w:
                        for sense, masked in (("visible", False), ("masked", True)):
                            name = arm + "_" + sense
                            with torch.no_grad():
                                first = distribution(
                                    actor.scores(original, masked=masked)
                                ).numpy()
                                second = distribution(
                                    actor.scores(singleton, masked=masked)
                                ).numpy()
                                third = tail.score("linear", w, force_mask=masked)
                            np.testing.assert_allclose(
                                first, archived[name], atol=3e-6, rtol=0
                            )
                            key = f"s{seed}-m{m}-{name}"
                            raw[key + "-singleton-smooth"] = second
                            raw[key + "-singleton-bf16"] = third
                            records[key] = {
                                "rerun_vs_saved": gap(first, archived[name]),
                                "batch_states_to_singleton_same_math": gap(
                                    first, second
                                ),
                                "singleton_smooth_to_bf16": gap(second, third),
                                "published_batch_to_singleton_bf16": gap(
                                    archived[name], third
                                ),
                            }
                            worldscores[name] = metrics(third, ref[m])
            worlds.append(worldscores)
        one = {}
        for sense in ("visible", "masked"):
            for key in ("danger_answer", "healthy_answer", "healthy_style"):
                one[sense + "_" + key] = float(
                    np.mean(
                        [
                            v["nonfatal_" + sense][key] - v["high_" + sense][key]
                            for v in worlds
                        ]
                    )
                )
            for h in (2, 6, 10):
                j = TEST_H.index(float(h))
                one[sense + f"_death{h}"] = float(
                    np.mean(
                        [
                            v["nonfatal_" + sense]["death"][j]
                            - v["high_" + sense]["death"][j]
                            for v in worlds
                        ]
                    )
                )
        paired.append(one)
    summary = {
        "seeds": seeds,
        "frozen_original_batch_vs_fresh_singleton": gap(old, new),
        "per_checkpoint": records,
        "paired_singleton_bf16_coverage_minus_high": {
            k: stat([p[k] for p in paired]) for k in paired[0]
        },
        "pairs": paired,
    }
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / "nonfatal-numerics.npz", **raw)
    write(out / "nonfatal-numerics.json", summary)
    print(
        summary["frozen_original_batch_vs_fresh_singleton"],
        {
            k: summary["paired_singleton_bf16_coverage_minus_high"][k]["mean"]
            for k in (
                "visible_danger_answer",
                "visible_healthy_answer",
                "visible_death6",
                "visible_death10",
            )
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("cache", "fresh", "study", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(a.cache, a.fresh, a.study, a.output)
