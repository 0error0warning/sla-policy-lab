"""Next-request probabilities on the same experience and outcomes on new paired lives."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.damage_penalty.design import ARMS, PILOT, REPLICATION
from experiments.damage_penalty.inputs import SOURCE_WAVES
from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import discord
from experiments.first_fatal.design import BIRTHS
from experiments.health_baseline.analysis import ci, live_row, read_replay, save_csv
from experiments.individual_life.design import action_at

FRESH = (PILOT, REPLICATION)
ALPHA = 0.05 / 10
COMPARISONS = {
    "one_step": (0, 1),
    "two_step_same_start": (0, 4),
    "two_minus_one_step": (1, 4),
    "two_step_own": (2, 3),
    "two_minus_one_cumulative": (1, 3),
}


def replay_events(root):
    rows = []
    for f in sorted(root.glob("s*/*/life.json")):
        m = json.loads(f.read_bytes())
        for prev, nxt in itertools.pairwise(read_replay(f.parent)):
            assert (
                prev["after"] == nxt["health"] > 0 and nxt["turn"] == prev["turn"] + 1
            )
            p = np.array(nxt["probs"], float)
            ans = p[:, 2:].sum(1)
            if prev["reward"] == 0:
                assert (
                    prev["updates"][0]["after_hash"] == prev["updates"][2]["after_hash"]
                )
                np.testing.assert_array_equal(p[1], p[4])
            draws = [action_at(v, nxt["u"]) for v in p]
            row = {
                "seed": m["seed"],
                "birth": m["birth"],
                "turn": prev["turn"],
                "action": prev["action"],
                "before_health": prev["health"],
                "next_health": nxt["health"],
                "physical_wear": prev["health"] - prev["after"],
                "rul_loss": -prev["reward"],
                "baseline": prev["local_baseline"],
                "eligible": prev["eligible"],
                "one_signal": prev["signals"][0],
                "two_signal": prev["signals"][1],
                "practice": prev["practice"],
                "next_index": nxt["index"],
                "next_uniform": nxt["u"],
            }
            for j, k in enumerate(
                ("one_before", "one_after", "two_before", "two_after", "two_from_one")
            ):
                row["p_answer_" + k] = float(ans[j])
                row["draw_" + k] = draws[j]
            for j, k in enumerate(("one", "two_own", "two_common")):
                u = prev["updates"][j]
                row[k + "_step_norm"] = u["actual_step_norm"]
                row[k + "_at_cap"] = int(abs(u["nominal_step_norm"] - 0.40) < 1e-8)
                row[k + "_propensity_clipped"] = int(
                    u["importance_ratio"] != u["applied_ratio"]
                )
            for k, (a, b) in COMPARISONS.items():
                row[k + "_delta"] = float(ans[b] - ans[a])
                row[k + "_changed"] = int(draws[a] != draws[b])
                row[k + "_class_changed"] = int((draws[a] >= 2) != (draws[b] >= 2))
            rows.append(row)
    return rows


def summarize_replay(root, out):
    rows = replay_events(root)
    risk = [r for r in rows if r["practice"] and r["action"] >= 2]
    assert len(risk) == 652
    save_csv(out / "risk-answer-successors.csv", risk)
    exposures = {
        "all_surviving_injuries": rows,
        "risky_answer": risk,
        "healthy_answer": [
            r for r in rows if r["before_health"] >= 90 and r["action"] >= 2
        ],
        "risky_refusal": [r for r in rows if r["practice"] and r["action"] < 2],
    }
    results = {}
    seed_rows = []
    for label, group in exposures.items():
        assert all(r["physical_wear"] > 0 for r in group)
        for birth in ("both", *BIRTHS):
            data = [r for r in group if birth == "both" or r["birth"] == birth]
            seeds = sorted({r["seed"] for r in data})
            by_seed = [[r for r in data if r["seed"] == s] for s in seeds]
            stats = {
                "n": len(data),
                "independent_seeds": len(seeds),
                "zero_reference_loss": sum(abs(r["rul_loss"]) <= 1e-12 for r in data),
                "one_negative": sum(r["one_signal"] < -1e-9 for r in data),
                "two_negative": sum(r["two_signal"] < -1e-9 for r in data),
                "positive_to_negative": sum(
                    r["one_signal"] > 1e-9 and r["two_signal"] < -1e-9 for r in data
                ),
                "mean_probability": {
                    k: float(np.mean([r["p_answer_" + k] for r in data]))
                    for k in (
                        "one_before",
                        "one_after",
                        "two_before",
                        "two_after",
                        "two_from_one",
                    )
                },
                "steps": {},
            }
            for k in ("one", "two_own", "two_common"):
                stats["steps"][k] = {
                    "median_norm": float(
                        np.median([r[k + "_step_norm"] for r in data])
                    ),
                    "at_cap": sum(r[k + "_at_cap"] for r in data),
                    "clipped_propensities": sum(
                        r[k + "_propensity_clipped"] for r in data
                    ),
                }
            for k in COMPARISONS:
                vals = np.array([r[k + "_delta"] for r in data])
                means = [
                    float(np.mean([r[k + "_delta"] for r in chunk]))
                    for chunk in by_seed
                ]
                stats[k] = {
                    "mean_event_delta": float(vals.mean()),
                    "median_delta": float(np.median(vals)),
                    "lower": int((vals < 0).sum()),
                    "equal": int((vals == 0).sum()),
                    "higher": int((vals > 0).sum()),
                    "changed": sum(r[k + "_changed"] for r in data),
                    "class_changed": sum(r[k + "_class_changed"] for r in data),
                    "seed_balanced_delta": ci(seeds, means, SOURCE_WAVES),
                }
                seed_rows.extend(
                    {
                        "exposure": label,
                        "birth": birth,
                        "contrast": k,
                        "seed": s,
                        "n": len(chunk),
                        "mean_delta": m,
                    }
                    for s, chunk, m in zip(seeds, by_seed, means, strict=True)
                )
            results[label + "_" + birth] = stats
    save_csv(out / "replay-seeds.csv", seed_rows)
    write(out / "replay.json", results)
    return results


def intervals(rows, vals):
    return ci([r["seed"] for r in rows], vals, FRESH, ALPHA)


def live_summary(roots, out):
    data = [
        live_row(f.parent)
        for root in roots
        for f in sorted(root.glob("s*/*/*/life.json"))
    ]
    save_csv(out / "lives.csv", data)
    result = {}
    for origin in BIRTHS:
        arms = {
            arm: sorted(
                (r for r in data if r["birth"] == origin and r["rule"] == arm),
                key=lambda r: r["seed"],
            )
            for arm in ARMS
        }
        assert all(
            [r["seed"] for r in vals] == list(PILOT + REPLICATION)
            for vals in arms.values()
        )
        info = {}
        for arm, vals in arms.items():
            reached = [r for r in vals if r["reached"]]
            info[arm] = {
                "assigned": len(vals),
                "reached": len(reached),
                "avoided": sum(r["avoided"] for r in vals),
                "pilot_avoided": sum(r["avoided"] for r in vals if r["seed"] in PILOT),
                "replication_avoided": sum(
                    r["avoided"] for r in vals if r["seed"] in REPLICATION
                ),
                "mean_healthy_task_rate": float(
                    np.mean([r["healthy_rate"] for r in vals])
                ),
                "healthy_tasks": sum(r["healthy_completed"] for r in vals),
                "healthy_encounters": sum(r["healthy_turns"] for r in vals),
                "mean_first_fatal_p_answer": float(
                    np.mean([r["fatal_current"] for r in reached])
                )
                if reached
                else None,
                "near_positive": sum(r["near_positive"] for r in vals),
                "near_negative": sum(r["near_negative"] for r in vals),
                "near_answers": sum(r["near_answers"] for r in vals),
            }
        contrasts = {}
        for arm in ("one_true", "two_true"):
            vals = [r for r in arms[arm] if r["reached"]]
            contrasts[arm] = {
                k + "_vs_" + role: intervals(
                    vals, [r[k + "_current"] - r[k + "_" + role] for r in vals]
                )
                for role in ("birth", "wrong", "other", "entry")
                for k in ("fatal", "fatal_panel", "healthy_same", "healthy_panel")
            }
            for role in ("birth", "other"):
                contrasts[arm]["descriptive_danger_minus_healthy_vs_" + role] = (
                    intervals(
                        vals,
                        [
                            (row["fatal_panel_current"] - row["fatal_panel_" + role])
                            - (
                                row["healthy_panel_current"]
                                - row["healthy_panel_" + role]
                            )
                            for row in vals
                        ],
                    )
                )
        deployed = arms["two_true"]
        free = {}
        for arm, other in arms.items():
            if arm == "two_true":
                continue
            pairs = list(zip(deployed, other, strict=True))
            both = [(a, b) for a, b in pairs if a["reached"] and b["reached"]]
            free[arm] = {
                "first_avoided": discord(
                    [(a["avoided"], b["avoided"]) for a, b in pairs]
                ),
                "actual_healthy_delta": intervals(
                    deployed, [a["healthy_rate"] - b["healthy_rate"] for a, b in pairs]
                ),
                "first_answer_probability_delta": intervals(
                    [a for a, _ in both],
                    [a["fatal_current"] - b["fatal_current"] for a, b in both],
                ),
            }
        sub = contrasts["two_true"]
        passes = {}
        for role in ("birth", "wrong", "other"):
            live = sub["fatal_vs_" + role]
            held = sub["fatal_panel_vs_" + role]
            passes["risk_vs_" + role] = bool(
                live
                and held
                and live["upper_adjusted"] < 0
                and all(
                    v[k] is not None and v[k] < 0
                    for k in ("pilot", "replication")
                    for v in (live, held)
                )
            )
        passes["healthy_panel"] = bool(
            sub["healthy_panel_vs_birth"]
            and sub["healthy_panel_vs_birth"]["lower_adjusted"] > -0.02
        )
        passes["healthy_actual"] = (
            free["frozen"]["actual_healthy_delta"]["lower_adjusted"] > -0.02
        )
        result[origin] = {
            "arms": info,
            "matched": contrasts,
            "freely_acting": free,
            "criteria": passes,
            "meets_joint_criterion": all(passes.values())
            and info["two_true"]["reached"] == 24,
        }
    write(out / "live.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path)
    p.add_argument("--lives", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.replay:
        summarize_replay(a.replay, a.output)
    if a.lives:
        live_summary(a.lives, a.output)
