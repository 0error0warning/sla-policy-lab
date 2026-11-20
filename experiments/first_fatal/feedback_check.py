"""Read-only counterfactual feedback audit on the practice histories of the true learners."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records
from experiments.first_fatal.audit import path
from experiments.first_fatal.design import FATAL
from experiments.social_closed_loop.core import BodyValue


def grid(body):
    from scipy.special import ndtr, ndtri
    from scipy.stats import qmc

    # Recover the actual training reward in expectation, independently of the
    # learner's personal baseline. Two scrambles quantify integration sensitivity.
    hs = np.linspace(FATAL + 1e-7, 6, 129)
    answers = []
    for seed in (69501, 69502):
        u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(13)
        eta = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * u)
        margins = []
        for h in hs:
            duration = (h / 48.0) ** 2
            scores = []
            for a, c in enumerate((0.0, 0.25, 1.75, 1.75)):
                peak = c * (2.0 - h / 350.0)
                action = h - duration * (peak + 4 * peak**3 / h**2) * (1 + eta[:, 0])
                after = action.copy()
                for j in range(1, 5):
                    after -= duration * (2.0 - after / 350.0) * (1.0 + eta[:, j])
                assert min(after) > 0
                q = (0.0, 0.15, 0.9, 0.1)[a]
                scores.append(
                    (
                        q * body.interpolate(action, body.reference).mean()
                        + (1 - q) * body.interpolate(after, body.reference).mean()
                    )
                    - body.rul(h)
                )
            margins.append(max(scores[:2]) - max(scores[2:]))
        answers.append(margins)
    x = np.array(answers)
    return {
        "held_health_values": len(hs),
        "noise_vectors_per_scramble": len(eta),
        "independent_scrambles": 2,
        "min_refusal_minus_best_answer_reference_reward": float(x.mean(axis=0).min()),
        "max_refusal_minus_best_answer_reference_reward": float(x.mean(axis=0).max()),
        "positive_grid_points_in_both": int((x.min(axis=0) > 0).sum()),
        "comparison_tolerance": 1e-5,
        "clear_refusal_grid_points_both": int((x.min(axis=0) > 1e-5).sum()),
        "clear_answer_grid_points_both": int((x.max(axis=0) < -1e-5).sum()),
        "near_tied_grid_points_both": int((np.abs(x).max(axis=0) <= 1e-5).sum()),
        "max_scramble_disagreement": float(np.abs(x[0] - x[1]).max()),
    }


def run(roots, out):
    body = BodyValue()
    group = defaultdict(list)
    per_life = {}
    for d in sorted(
        f.parent for root in roots for f in root.glob("s*/*/true1/life.json")
    ):
        meta = json.loads((d / "life.json").read_bytes())
        values = []
        for e in records(d):
            if not e["practice"]:
                continue
            j = e["partner"] - 2
            fav = 2 + (j ^ e["frame"] ^ meta["world"][j])
            p = [0.0, 0.15, 0.1, 0.1]
            p[fav] = 0.9
            returns = []
            for a in range(4):
                accept = None if a == 0 else e["u"][1] < p[a]
                after = path(e["health"], a, accept, e["noise"])[0][-1]
                returns.append(body.rul(after) - body.rul(e["health"]))
            a = e["action"]
            assert abs(returns[a] - e["reward"]) < 1e-9
            meandraw = np.dot(e["probs"][1], returns)
            values.append(
                (
                    a,
                    e["reward"],
                    e["baseline"],
                    e["signal"],
                    returns[a] - max(returns[:2]),
                    returns[a] - meandraw,
                )
            )
        group[meta["birth"]].extend(values)
        per_life[str(meta["seed"]) + "/" + meta["birth"]] = describe(values)
    write(
        out,
        {
            "reward_grid": grid(body),
            "true_lives": per_life,
            "practice_encounters": {k: describe(v) for k, v in group.items()},
        },
    )


def describe(rows):
    data = np.array(rows, dtype=float)
    out = {}
    for name, mask in (("answer", data[:, 0] >= 2), ("no_answer", data[:, 0] < 2)):
        x = data[mask]
        out[name] = {"n": len(x)}
        if not len(x):
            continue
        out[name].update(
            reward=float(x[:, 1].mean()),
            prior_personal_baseline=float(x[:, 2].mean()),
            signal=float(x[:, 3].mean()),
            positive_signal=int((x[:, 3] > 0).sum()),
            worse_than_best_refusal_same_draw=int((x[:, 4] < -1e-9).sum()),
            positive_despite_worse_than_best_refusal=int(
                ((x[:, 3] > 0) & (x[:, 4] < -1e-9)).sum()
            ),
            positive_despite_below_policy_mean_same_draw=int(
                ((x[:, 3] > 0) & (x[:, 5] < -1e-9)).sum()
            ),
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.lives, a.output)
