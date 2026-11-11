"""Isolate acquisition health range while holding actions, exogenous tape and learner fixed."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.consequence_policy.environment import draw_tape, settled_tape
from experiments.consequence_policy.experiment import (
    evaluation_reference,
    metrics,
    read_inputs,
)
from experiments.consequence_policy.policy import Policy, deployed, feedback_scores, fit
from experiments.consequence_policy.study import TEST_H, TRAIN_H
from experiments.direct_policy.protocol import rng, write
from experiments.social_closed_loop.core import BodyValue

RANGES = {
    "high": (35.0, 45.0, 60.0, 75.0, 90.0, 30.0, 120.0, 350.0),
    "nonfatal": (15.0, 16.0, 18.0, 21.0, 24.0, 30.0, 120.0, 350.0),
    "wide": TRAIN_H,
}
WAVES = {"pilot": tuple(range(63201, 63205)), "confirm": tuple(range(63301, 63321))}
SPLIT = {
    "high": RANGES["high"],
    "split": (1.25, 2.5, 8.0, 75.0, 90.0, 30.0, 120.0, 350.0),
}
SPLIT_WAVES = {
    "pilot": tuple(range(63401, 63405)),
    "confirm": tuple(range(63501, 63521)),
}


def score_seed(seed, output, prefix, ranges, candidate):
    paired = []
    for m in (0, 1):
        values = json.loads(
            (output / f"{prefix}-{seed}-m{m}/metrics.json").read_bytes()
        )["by_arm"]
        row = {}
        for control in (k for k in ranges if k != candidate):
            for sensor in ("visible", "masked"):
                a = values[candidate][sensor]
                b = values[control][sensor]
                label = sensor + "_vs_" + control + "_"
                row[label + "fatal_answer_reduction"] = (
                    b["danger_answer"] - a["danger_answer"]
                )
                row[label + "healthy_answer_change"] = (
                    a["healthy_answer"] - b["healthy_answer"]
                )
                for h in (2.0, 6.0, 10.0):
                    i = TEST_H.index(h)
                    row[label + f"death_reduction_at_{int(h)}"] = (
                        b["death"][i] - a["death"][i]
                    )
        paired.append(row)
    return {k: float(np.mean([m[k] for m in paired])) for k in paired[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wave", choices=WAVES, required=True)
    parser.add_argument("--allocation", choices=("wide", "split"), default="wide")
    args = parser.parse_args()
    split = args.allocation == "split"
    prefix = "allocation" if split else "coverage"
    ranges, waves = (SPLIT, SPLIT_WAVES) if split else (RANGES, WAVES)
    if args.wave == "confirm":
        assert json.loads((args.output / f"{prefix}-pilot.json").read_bytes())["passed"]
    seeds = waves[args.wave]
    rows, native, x, partner, train = read_inputs(args.cache, args.output)
    train_rows, test_rows = [
        [r for r in rows if r["split"] == group] for group in ("train", "test")
    ]
    body = BodyValue()
    references = [evaluation_reference(body, test_rows, m) for m in (0, 1)]
    for seed in seeds:
        chosen, mu, partner_u, noise = draw_tape(seed, native[train])
        np.savez_compressed(
            args.output / f"{prefix}-{seed}-behavior.npz",
            action=chosen,
            mu=mu,
            exogenous_sha256=np.array(
                sha256(partner_u.tobytes() + noise.tobytes()).hexdigest()
            ),
        )
        for mapping in (0, 1):
            folder = args.output / f"{prefix}-{seed}-m{mapping}"
            folder.mkdir(exist_ok=True)
            summary, raw, training_log = {}, {}, {}
            outcomes = None
            for name, healths in ranges.items():
                after, r, response = settled_tape(
                    chosen, partner_u, noise, train_rows, mapping, body, healths
                )
                if name in ("high", "nonfatal"):
                    assert np.all(after > 0)
                if outcomes is None:
                    outcomes = response
                    raw["response"] = response
                else:
                    assert np.array_equal(outcomes, response)
                    assert np.array_equal(raw["high_reward"][-3:], r[-3:])
                raw[name + "_after"] = after
                raw[name + "_reward"] = r
                scale = np.minimum(1.0, [body.rul(h) for h in healths])
                scores = feedback_scores(chosen, r, mu, scale)
                policy = Policy(x.shape[1])
                training_log[name] = fit(
                    policy, native[train], x[train], partner[train], healths, scores
                )
                policy.save(folder / (name + ".npz"))
                masked = Policy(x.shape[1], blind=True)
                masked.weights[:, 0] = np.einsum(
                    "b,pbda->pda", policy.gates(TRAIN_H).mean(axis=0), policy.weights
                )
                measures = {}
                for sensor, actor in (("visible", policy), ("masked", masked)):
                    pi = deployed(
                        actor.grid_logits(
                            native[~train], x[~train], partner[~train], TEST_H
                        )
                    )
                    raw[name + "_" + sensor + "_pi"] = pi
                    measures[sensor] = metrics(pi, references[mapping])
                measures["training"] = [
                    {
                        "health": h,
                        "action": a,
                        "count": int((chosen[j] == a).sum()),
                        "maintenance": int(
                            ((chosen[j] == a) & (response[j] == 1)).sum()
                        ),
                        "deaths": int(((chosen[j] == a) & (after[j] == 0)).sum()),
                        "mean_reward": float(r[j][chosen[j] == a].mean()),
                    }
                    for j, h in enumerate(healths)
                    for a in range(4)
                ]
                summary[name] = measures
            np.savez_compressed(folder / "raw.npz", **raw)
            write(
                folder / "metrics.json",
                {"by_arm": summary, "seed": seed, "mapping": mapping},
            )
            write(folder / "fit.json", training_log)
        print("coverage seed", seed, flush=True)
    paired = [
        score_seed(s, args.output, prefix, ranges, args.allocation) for s in seeds
    ]
    bootstrap = rng(63995, "coverage-bootstrap").integers(
        len(seeds), size=(12000, len(seeds))
    )
    stats = {}
    for k in paired[0]:
        a = np.array([s[k] for s in paired])
        stats[k] = {
            "mean": float(a.mean()),
            "positive": int((a > 0).sum()),
            "ci95": np.quantile(a[bootstrap].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    danger = stats["visible_vs_high_fatal_answer_reduction"]
    healthy = stats["visible_vs_high_healthy_answer_change"]
    passed = (
        danger["mean"] >= 0.10
        and healthy["mean"] >= -0.05
        and danger["positive"] >= (3 if len(seeds) == 4 else 16)
    )
    if len(seeds) == 20:
        passed = passed and danger["ci95"][0] > 0
    write(
        args.output / f"{prefix}-{args.wave}.json",
        {"seeds": list(seeds), "paired": paired, "stats": stats, "passed": passed},
    )
    print(
        json.dumps({"danger": danger, "healthy": healthy, "passed": passed}), flush=True
    )


if __name__ == "__main__":
    main()
