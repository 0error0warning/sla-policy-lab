"""Condition-specific opportunity costs and robustness to action presentation order."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import write
from experiments.lethal_boundary.design import CONFIRM, HEALTH, INTERIOR, QS
from experiments.lethal_boundary.evaluate import BANDS, index, policies, summarize


def healthy_training_credit(root):
    rows = contexts()[:768]
    names = ("silent", "refuse_and_appeal", "preferred_answer", "other_answer")
    sums = np.zeros((2, 4))
    counts = np.zeros((2, 4), dtype=np.int64)
    for seed in CONFIRM:
        folder = root / f"s{seed}"
        with np.load(folder / "tape.npz") as tape:
            action = tape["action"][4:]
            profile = tape["profile"]
        for mapping in (0, 1):
            with np.load(folder / f"m{mapping}/outcomes.npz") as data:
                reward = data["D_reward"][4:]
                np.testing.assert_array_equal(reward, data["R_reward"][4:])
            n = reward.shape[-1]
            advantage = (reward - reward.mean(axis=-1, keepdims=True)) * n / (n - 1)
            fav = np.array([2 + (r["partner"] ^ r["frame"] ^ mapping) for r in rows])
            category = np.where(
                action < 2, action, np.where(action == fav[None, :, None], 2, 3)
            )
            for q in (0, 1):
                for a in range(4):
                    hit = (profile[None, :, None] == q) & (category == a)
                    counts[q, a] += int(hit.sum())
                    sums[q, a] += advantage[hit].sum()
    return {
        str(q): {
            name: {
                "count": int(counts[i, a]),
                "mean_advantage": float(sums[i, a] / counts[i, a]),
            }
            for a, name in enumerate(names)
        }
        for i, q in enumerate((0.25, 0.85))
    }


def run(root, analysis, out):
    axis = np.array(json.loads((root / "axes.json").read_bytes())["health"])
    idx = index(axis, HEALTH)
    rows = contexts()[768:]
    names = {"layout": range(24), "partner": range(2), "task": (4, 5)}
    items = {
        f"{field}-{i}": np.flatnonzero([r[field] == i for r in rows])
        for field, levels in names.items()
        for i in levels
    }
    groups = {k: [] for k in items}
    mort = []
    with np.load(root / "frozen-record.npz") as z:
        f = z["panel"][:, :, :, 2:].sum(-1)
    for s in CONFIRM:
        mirrors = {k: [] for k in groups}
        by = []
        for m in (0, 1):
            folder = root / f"s{s}/m{m}"
            raw = {}
            mean = {}
            for arm in ("M", "R"):
                with np.load(folder / f"record-{arm}-probe.npz") as z:
                    raw[arm] = z["panel"][:, :, :, 2:].sum(-1)
                    allpi = policies(z["mean"], "iid")[:, idx]
                    mean[arm] = allpi[INTERIOR][:, BANDS["doomed"]].mean(axis=(0, 1, 2))
            by.append(mean)
            for key, ix in items.items():
                met = {}
                for arm, p in {**raw, "frozen": f}.items():
                    band_d = np.take(p[:2], ix, axis=2).mean(axis=(0, 2))
                    band_r = np.take(p[5], ix, axis=1).mean(axis=1)
                    met[arm] = (
                        band_d[[1, 3]].mean(),
                        (band_r[3] - band_r[1]) - (band_d[3] - band_d[1]),
                    )
                mirrors[key].append(
                    [met["R"][0] - met["M"][0], met["M"][1] - met["frozen"][1]]
                )
        for key in items:
            groups[key].append(np.mean(mirrors[key], axis=0))
        mort.append(
            {k: np.mean([x[k] for x in by], axis=0).tolist() for k in ("M", "R")}
        )
    summary = {
        k: {
            "doomed_fatal_answer_reduction": summarize(np.array(v)[:, 0]),
            "rescue_q_interaction_above_frozen": summarize(np.array(v)[:, 1]),
        }
        for k, v in groups.items()
    }
    # Cross the RUL values of the best refusal and a favored answer by varying q.
    # The other answering expression has identical bodily cost and q >= 0.10 by design.
    with np.load(analysis / "physical.npz") as z:
        v = z["conditional_rul_change"]
    phys = []
    for h, u in zip(HEALTH, v, strict=True):
        best = max(u[0, 0], 0.85 * u[1, 0] + 0.15 * u[1, 1])
        den = u[2, 1] - u[2, 0]
        phys.append(
            {
                "health": h,
                "favored_q_equal_best_refusal": float((best - u[2, 0]) / den)
                if den > 1e-12
                else None,
                "best_refusal_rul_change": float(best),
                "answer_maintained_rul_change": float(u[2, 1]),
                "answer_unmaintained_rul_change": float(u[2, 0]),
            }
        )
    with np.load(analysis / "seed-surfaces.npz") as z:
        for j, pair in enumerate(mort):
            for arm in ("M", "R"):
                actual = z["record-" + arm][j, INTERIOR][:, BANDS["doomed"], 3].mean()
                np.testing.assert_allclose(
                    1 - 0.15 * pair[arm][1], actual, atol=2e-7, rtol=0
                )
    out.mkdir(exist_ok=True, parents=True)
    write(
        out / "diagnosis.json",
        {
            "ordered_subsets": summary,
            "doomed_action_probabilities_by_seed": mort,
            "physical_q_crossovers": phys,
            "seeds": CONFIRM,
            "ecologies_per_seed": 2,
            "min_possible_doomed_mortality": 0.85,
            "high_health_training_credit": healthy_training_credit(root),
            "q": QS.tolist(),
        },
    )
    print(
        "Verified doomed-risk decomposition and",
        len(groups),
        "presentation/task/partner subsets",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "analysis", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.analysis, a.output)
