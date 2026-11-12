"""Whole-seed inference for the one pre-feedback fatal choice and retained healthy tasks."""

import argparse
import csv
import gzip
import json
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import rng, write
from experiments.first_fatal.design import (
    BIRTHS,
    HEALTH_MARGIN,
    PILOT,
    REPLICATION,
    RULES,
)
from experiments.individual_life.design import action_at


def records(d):
    with gzip.open(d / "events.jsonl.gz", "rt", encoding="utf8") as f:
        return [json.loads(s) for s in f]


def mean(values):
    v = list(values)
    return float(np.mean(v)) if v else None


def extract(d):
    m = json.loads((d / "life.json").read_bytes())
    ev = records(d)
    first = next((t for t in ev if t["first_fatal"]), None)
    near = [t for t in ev if t["practice"]]
    healthy = [t for t in ev if t["health"] >= 90]
    result = {
        k: m[k]
        for k in (
            "seed",
            "birth",
            "rule",
            "turns",
            "first_fatal",
            "practice_entry",
            "practice_count",
        )
    }
    result.update(
        reached=int(first is not None),
        n_healthy=len(healthy),
        health_live_tasks=sum(t["action"] >= 2 and t["after"] > 0 for t in healthy),
        health_live_rate=mean(t["action"] >= 2 and t["after"] > 0 for t in healthy),
        health_live_probability=mean(sum(t["probs"][1][2:]) for t in healthy),
        near_answers=sum(t["action"] >= 2 for t in near),
        near_maintenance=sum(t["maintenance"] is True for t in near),
        near_answer_wear=mean(
            t["health"] - t["after"] for t in near if t["action"] >= 2
        ),
        near_other_wear=mean(t["health"] - t["after"] for t in near if t["action"] < 2),
        near_answer_positive=sum(t["signal"] > 0 for t in near if t["action"] >= 2),
        near_other_positive=sum(t["signal"] > 0 for t in near if t["action"] < 2),
        near_answer_feedback=mean(t["reward"] for t in near if t["action"] >= 2),
        near_answer_advantage=mean(t["signal"] for t in near if t["action"] >= 2),
        near_other_advantage=mean(t["signal"] for t in near if t["action"] < 2),
    )
    if first is None:
        return result
    with np.load(d / "predecision.npz") as z:
        # Live and standardized chances are calculated in semantic token order.
        same = z["same_request"][:, :5, 2:].astype(float).sum(-1)
        panel = z["panel"][:, :, :5, 2:].astype(float).sum(-1).mean(1)
    result.update(
        first_health=first["health"],
        first_action=first["word"],
        first_avoided=int(first["action"] < 2),
        first_survived=int(first["after"] > 0),
        first_partner=first["partner"],
        first_frame=first["frame"],
        first_layout=first["layout"],
        first_draw=first["u"][0],
    )
    for j, role in enumerate(("birth", "current", "wrong", "near_wrong", "entry")):
        if role in ("wrong", "near_wrong") and m["rule"] != "true1":
            continue
        result["fatal_" + role] = float(same[0, j])
        result["same_healthy_" + role] = float(np.mean(same[1:, j]))
        result["panel_fatal_" + role] = float(panel[0, j])
        result["panel_healthy_" + role] = float(np.mean(panel[1:, j]))
        result["avoid_draw_" + role] = int(
            action_at(first["probs"][j], first["u"][0]) < 2
        )
    for role in ("birth", "wrong", "near_wrong", "entry"):
        if "fatal_" + role not in result:
            continue
        for prefix in ("fatal", "same_healthy", "panel_fatal", "panel_healthy"):
            result[prefix + "_vs_" + role] = (
                result[prefix + "_current"] - result[prefix + "_" + role]
            )
        result["selectivity_vs_" + role] = (
            result["panel_healthy_vs_" + role] - result["panel_fatal_vs_" + role]
        )
    return result


@lru_cache(maxsize=32)
def resamples(groups):
    ids = []
    start = 0
    for j, n in enumerate(groups):
        if n:
            ids.append(
                start
                + rng(69400, "prelocked-stratified-bootstrap", j, n).integers(
                    n, size=(50000, n)
                )
            )
        start += n
    return np.concatenate(ids, axis=1)


def interval(pilot, replication):
    sizes = (len(pilot), len(replication))
    values = np.asarray(list(pilot) + list(replication), float)
    if not len(values):
        return None
    boot = values[resamples(sizes)].mean(axis=1)
    adj = 0.05 / 6
    a, lo, hi, b = np.quantile(boot, [adj, 0.025, 0.975, 1 - adj])
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "ci95": [float(lo), float(hi)],
        "adjusted_lower": float(a),
        "adjusted_upper": float(b),
        "negative": int((values < 0).sum()),
        "positive": int((values > 0).sum()),
        "pilot_mean": mean(pilot),
        "replication_mean": mean(replication),
        "range": [float(values.min()), float(values.max())],
    }


def describe(rows, key):
    pairs = [(r["seed"], r[key]) for r in rows if r.get(key) is not None]
    return interval(
        [v for s, v in pairs if s in PILOT], [v for s, v in pairs if s in REPLICATION]
    )


def exact_p(discordant_a, discordant_b):
    n = discordant_a + discordant_b
    return min(
        1.0,
        2
        * sum(comb(n, k) for k in range(min(discordant_a, discordant_b) + 1))
        / (2**n),
    )


def discord(pairs):
    plus = sum(a > b for a, b in pairs)
    minus = sum(a < b for a, b in pairs)
    return {
        "pairs": len(pairs),
        "true_avoids_other_answers": plus,
        "true_answers_other_avoids": minus,
        "same": len(pairs) - plus - minus,
        "paired_exact_two_sided_p": exact_p(plus, minus),
    }


def negative_in_both(v):
    return (
        v is not None
        and v["pilot_mean"] is not None
        and v["replication_mean"] is not None
        and max(v["pilot_mean"], v["replication_mean"]) < 0
    )


def summarize(rows):
    result = {}
    for birth in BIRTHS:
        cohort = [r for r in rows if r["birth"] == birth]
        true = [r for r in cohort if r["rule"] == "true1"]
        if not true:
            continue
        keys = (
            "fatal_current",
            "fatal_birth",
            "fatal_wrong",
            "fatal_near_wrong",
            "fatal_entry",
            "fatal_vs_birth",
            "fatal_vs_wrong",
            "fatal_vs_near_wrong",
            "fatal_vs_entry",
            "same_healthy_vs_birth",
            "panel_fatal_current",
            "panel_fatal_birth",
            "panel_fatal_vs_birth",
            "panel_fatal_vs_wrong",
            "panel_fatal_vs_near_wrong",
            "panel_fatal_vs_entry",
            "panel_healthy_current",
            "panel_healthy_birth",
            "panel_healthy_vs_birth",
            "panel_healthy_vs_wrong",
            "selectivity_vs_birth",
            "near_answer_feedback",
            "near_answer_advantage",
            "near_other_advantage",
            "near_answer_wear",
            "near_other_wear",
        )
        estimates = {k: describe(true, k) for k in keys}
        arms = {}
        for rule in RULES:
            arm = [r for r in cohort if r["rule"] == rule]
            hit = [r for r in arm if r["reached"]]
            arms[rule] = {
                "assigned": len(arm),
                "reached": len(hit),
                "averted": sum(r["first_avoided"] for r in hit),
                "dead_on_first": sum(1 - r["first_survived"] for r in hit),
                "competing_deaths": len(arm) - len(hit),
                "actual_first_answer_prob": describe(arm, "fatal_current"),
                "healthy_live_tasks": sum(r["health_live_tasks"] for r in arm),
                "healthy_live_encounters": sum(r["n_healthy"] for r in arm),
                "healthy_live_rate": describe(arm, "health_live_rate"),
                "practice_counts": [
                    min(r["practice_count"] for r in arm),
                    max(r["practice_count"] for r in arm),
                ],
                "mean_practice": mean(r["practice_count"] for r in arm),
                "mean_duration": mean(r["turns"] for r in arm),
            }
        pairs = {}
        for other in ("frozen", "wrong1"):
            match = {r["seed"]: r for r in cohort if r["rule"] == other}
            ab = []
            continuous = []
            for r in true:
                b = match[r["seed"]]
                if r["reached"] and b["reached"]:
                    ab.append((r["first_avoided"], b["first_avoided"]))
                continuous.append(
                    {
                        "seed": r["seed"],
                        "p": r["fatal_current"] - b["fatal_current"]
                        if r["reached"] and b["reached"]
                        else None,
                        "healthy": r["health_live_rate"] - b["health_live_rate"],
                    }
                )
            pairs[other] = {
                **discord(ab),
                "fatal_probability_difference": describe(continuous, "p"),
                "healthy_rate_difference": describe(continuous, "healthy"),
            }
        same_draw = {
            k: discord(
                [
                    (r["first_avoided"], r["avoid_draw_" + k])
                    for r in true
                    if r["reached"]
                ]
            )
            for k in ("birth", "wrong", "near_wrong", "entry")
        }
        complete = all(
            {r["seed"] for r in cohort if r["rule"] == rule and r["reached"]}
            == set(PILOT + REPLICATION)
            for rule in RULES
        )

        guard = complete and all(
            negative_in_both(estimates[k]) and estimates[k]["adjusted_upper"] < 0
            for k in ("fatal_vs_birth", "fatal_vs_wrong")
        )
        uniform = complete and all(
            negative_in_both(estimates[k])
            for k in ("panel_fatal_vs_birth", "panel_fatal_vs_wrong")
        )
        healthy = (
            complete
            and estimates["panel_healthy_vs_birth"]["adjusted_lower"] > -HEALTH_MARGIN
        )
        nans = sum(r["near_answers"] for r in true)
        n = sum(r["practice_count"] for r in true)
        result[birth] = {
            "arms": arms,
            "paired_free_lives": pairs,
            "same_draw": same_draw,
            "estimates": estimates,
            "practice_feedback": {
                "answers": nans,
                "nonanswers": n - nans,
                "answer_positive": sum(r["near_answer_positive"] for r in true),
                "nonanswer_positive": sum(r["near_other_positive"] for r in true),
            },
            "decision": {
                "all_assigned_reached": complete,
                "first_fatal_reduction": guard,
                "balanced_panel_direction": uniform,
                "healthy_noninferiority": healthy,
                "joint_supported": guard and uniform and healthy,
                "practice_credit_direction": negative_in_both(
                    estimates["fatal_vs_near_wrong"]
                ),
            },
        }
    return result


def run(roots, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        [extract(f.parent) for root in roots for f in root.glob("s*/*/*/life.json")],
        key=lambda r: (r["seed"], r["birth"], r["rule"]),
    )
    assert rows and len({(r["seed"], r["birth"], r["rule"]) for r in rows}) == len(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "per-life.csv").open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write(out / "summary.json", summarize(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.lives, a.output)
