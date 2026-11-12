"""Paired nonfatal-only acquisition for the same internal LoRA and optimizer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.coverage import RANGES, score_seed
from experiments.consequence_policy.environment import draw_tape, settled_tape
from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.policy import feedback_scores
from experiments.consequence_policy.study import TEST_H
from experiments.direct_policy.protocol import write
from experiments.internal_body.evidence import stat
from experiments.internal_body.suffix import (
    InternalPolicy,
    StateBank,
    distribution,
    fit,
)
from experiments.social_closed_loop.core import BodyValue

ARMS = {k: RANGES[k] for k in ("high", "nonfatal")}
WAVES = {"pilot": range(65801, 65805), "confirm": range(65911, 65931)}


def acquire(out, seed, bank, body, indices):
    action, mu, uniform, noise = draw_tape(seed, bank.native[indices])
    np.savez_compressed(
        out / f"behavior-{seed}.npz",
        action=action,
        mu=mu,
        partner_uniform=uniform,
        body_noise=noise,
    )
    rows = [bank.rows[i] for i in indices]
    for mapping in (0, 1):
        folder = out / f"coverage-{seed}-m{mapping}"
        folder.mkdir(exist_ok=True)
        raw = {}
        audit = {}
        for name, hs in ARMS.items():
            after, reward, response = settled_tape(
                action, uniform, noise, rows, mapping, body, hs
            )
            assert np.all(after > 0)
            if name == "high":
                raw["response"] = response
            else:
                np.testing.assert_array_equal(response, raw["response"])
                for k, v in (("after", after), ("reward", reward)):
                    np.testing.assert_array_equal(v[-3:], raw["high_" + k][-3:])
            scale = np.minimum(1.0, [body.rul(h) for h in hs])
            n = action.shape[-1]
            advantage = (
                (reward - reward.mean(-1, keepdims=True))
                * n
                / (n - 1)
                / scale[:, None, None]
            )
            for k, v in (
                ("after", after),
                ("reward", reward),
                ("advantage", advantage),
                ("scores", feedback_scores(action, reward, mu, scale)),
            ):
                raw[name + "_" + k] = v
            audit[name] = []
            for j, h in enumerate(hs):
                for act in range(4):
                    hit = action[j] == act
                    audit[name].append(
                        {
                            "health": h,
                            "action": act,
                            "n": int(hit.sum()),
                            "maintained": int((response[j][hit] == 1).sum()),
                            "no_observation": int((response[j][hit] == -1).sum()),
                            "deaths": int((after[j][hit] == 0).sum()),
                            "mean_damage": float(h - after[j][hit].mean()),
                            "mean_reward": float(reward[j][hit].mean()),
                            "mean_advantage": float(advantage[j][hit].mean()),
                            "positive_advantage": int((advantage[j][hit] > 0).sum()),
                        }
                    )
        for a, b in zip(audit["high"], audit["nonfatal"], strict=True):
            assert all(
                a[k] == b[k]
                for k in ("action", "n", "maintained", "no_observation", "deaths")
            )
        np.savez_compressed(folder / "outcomes.npz", **raw)
        write(folder / "data-audit.json", audit)


def train_pair(folder, bank, train, test, ref):
    with np.load(folder / "outcomes.npz") as raw:
        targets = {k: raw[k + "_scores"] for k in ARMS}
    history, probes, records = {}, {}, {}
    test_grid = bank.grid(test, TEST_H)
    for name, hs in ARMS.items():
        model = InternalPolicy(bank, "linear")
        history[name] = fit(model, bank.grid(train, hs), targets[name])
        model.save(folder / (name + ".npz"))
        records[name] = {}
        for setting, masked in (("visible", False), ("masked", True)):
            with torch.no_grad():
                pi = distribution(model.scores(test_grid, masked=masked)).numpy()
            if masked:
                np.testing.assert_allclose(
                    pi, np.broadcast_to(pi[0], pi.shape), rtol=0, atol=3e-6
                )
            probes[name + "_" + setting] = pi
            record = metrics(pi, ref)
            record["answer300"] = float(pi[TEST_H.index(300.0), :, 2:].sum(-1).mean())
            if masked:
                record["max_same_input_probability_span"] = float(
                    np.ptp(pi, axis=0).max()
                )
            records[name][setting] = record
    write(folder / "metrics.json", {"by_arm": records})
    write(folder / "steps.json", history)
    np.savez_compressed(folder / "probes.npz", **probes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wave", choices=WAVES, required=True)
    a = p.parse_args()
    if (
        a.wave == "confirm"
        and not json.loads((a.output / "pilot.json").read_bytes())["passed"]
    ):
        raise ValueError(
            "The four-seed nonfatal coverage screen missed its locked gate."
        )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(exist_ok=True, parents=True)
    bank = StateBank(a.cache)
    train, test = np.arange(768), np.arange(768, 960)
    seeds = list(WAVES[a.wave])
    body = BodyValue()
    refs = [evaluation_reference(body, [bank.rows[i] for i in test], m) for m in (0, 1)]
    for seed in seeds:
        acquire(a.output, seed, bank, body, train)
    print(
        "Paired acquisition and distributions verified before optimization",
        seeds,
        flush=True,
    )
    paired = []
    for seed in seeds:
        for m in (0, 1):
            train_pair(a.output / f"coverage-{seed}-m{m}", bank, train, test, refs[m])
        paired.append(score_seed(seed, a.output, "coverage", ARMS, "nonfatal"))
    stats = {k: stat([v[k] for v in paired]) for k in paired[0]}
    danger = stats["visible_vs_high_fatal_answer_reduction"]
    healthy = stats["visible_vs_high_healthy_answer_change"]
    passed = (
        danger["mean"] >= 0.10
        and healthy["mean"] >= -0.05
        and danger["positive"] >= (3 if a.wave == "pilot" else 16)
    )
    if a.wave == "confirm":
        passed = passed and danger["ci95"][0] > 0
    summary = {"seeds": seeds, "paired": paired, "stats": stats, "passed": bool(passed)}
    write(a.output / (a.wave + ".json"), summary)
    print(json.dumps(summary["stats"]), flush=True)
    print("Passed", bool(passed), flush=True)


if __name__ == "__main__":
    main()
