"""Predeclared seed-level development gates; all thresholds use probability units."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import TEST_H
from experiments.consequence_warmstart.design import (
    ARMS,
    BLOCKS,
    CHECKPOINTS,
    ENV_SEEDS,
)
from experiments.direct_policy.protocol import write


def healthy(panel, name):
    return float(np.mean([panel[name][TEST_H.index(h)] for h in (90.0, 300.0)]))


def summary(index):
    if (
        len(index) != 32
        or len({(c["environment_seed"], c["block"], tuple(c["world"])) for c in index})
        != 32
    ):
        raise ValueError("Incomplete or duplicate development matrix")
    blocks = {}
    for block in BLOCKS:
        cases = [c for c in index if c["block"] == block]
        seed_rows = []
        for seed in ENV_SEEDS:
            subset = [c for c in cases if c["environment_seed"] == seed]
            if len(subset) != 4:
                raise ValueError("Incomplete world coverage")
            rows = []
            for case in subset:
                arms = case["arms"]
                panels = {
                    a: arms[a]["checkpoints"]["last_preterminal"]["panels"]["heldout"]
                    for a in ARMS
                }
                warm = panels["warmstart_adapter"]
                shadow = arms["warmstart_adapter"]["replays"]["last_preterminal"][
                    "shuffle_panels"
                ]["heldout"]
                rows.append(
                    {
                        "personal_vs_prior": healthy(panels["personal"], "style")
                        - healthy(panels["prior"], "style"),
                        "warm_vs_prior": healthy(warm, "style")
                        - healthy(panels["prior"], "style"),
                        "warm_vs_shuffle": healthy(warm, "style")
                        - healthy(shadow, "style"),
                        "warm_vs_scratch": healthy(warm, "style")
                        - healthy(panels["scratch"], "style"),
                        "warm_conditional_vs_prior": healthy(warm, "conditional_style")
                        - healthy(panels["prior"], "conditional_style"),
                        "tasks_vs_prior": arms["warmstart_adapter"]["tasks"]
                        - arms["prior"]["tasks"],
                        "turns_vs_prior": arms["warmstart_adapter"]["turns"]
                        - arms["prior"]["turns"],
                    }
                )
            seed_rows.append(
                {
                    "seed": seed,
                    **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                }
            )
        contrasts = {
            k: {
                "mean": float(np.mean([r[k] for r in seed_rows])),
                "range": [
                    float(min(r[k] for r in seed_rows)),
                    float(max(r[k] for r in seed_rows)),
                ],
                "positive_seeds": sum(r[k] > 0 for r in seed_rows),
            }
            for k in seed_rows[0]
            if k != "seed"
        }
        absolute = {}
        for arm in ARMS:
            panels = [
                c["arms"][arm]["checkpoints"]["last_preterminal"]["panels"]["heldout"]
                for c in cases
            ]
            absolute[arm] = {
                "healthy_style": float(np.mean([healthy(p, "style") for p in panels])),
                "healthy_answer": float(
                    np.mean([healthy(p, "answer") for p in panels])
                ),
                "conditional_style": float(
                    np.mean([healthy(p, "conditional_style") for p in panels])
                ),
                "death": np.mean([p["death"] for p in panels], axis=0).tolist(),
                **{
                    k: float(np.mean([c["arms"][arm][k] for c in cases]))
                    for k in (
                        "turns",
                        "tasks",
                        "injury_then_another_choice",
                        "nonterminal_updates",
                    )
                },
                "capped": sum(c["arms"][arm]["capped"] for c in cases),
            }
        retention = []
        for label in [*(f"after_{n}" for n in CHECKPOINTS), "last_preterminal"]:
            pairs = []
            for case in cases:
                snaps = case["arms"]["warmstart_adapter"]["checkpoints"]
                if label in snaps:
                    birth = snaps["birth"]["panels"]["heldout"]
                    now = snaps[label]["panels"]["heldout"]
                    pairs.append((birth, now))
            if not pairs:
                retention.append({"checkpoint": label, "count": 0, "passed": False})
                continue
            healthy_change = float(
                np.mean([healthy(b, "answer") - healthy(a, "answer") for a, b in pairs])
            )
            death_change = np.mean(
                [np.array(b["death"]) - a["death"] for a, b in pairs], axis=0
            )
            danger = [TEST_H.index(h) for h in (2.0, 3.0, 4.0, 6.0)]
            retention.append(
                {
                    "checkpoint": label,
                    "count": len(pairs),
                    "healthy_change": healthy_change,
                    "death_change": death_change.tolist(),
                    "passed": bool(
                        healthy_change >= -0.05 and np.max(death_change[danger]) <= 0.02
                    ),
                }
            )
        opposite_count = both_improved = 0
        for case in cases:
            if case["world"][0] != case["world"][1]:
                continue
            opposite_count += 1
            arms = case["arms"]
            ps = [
                arms[a]["checkpoints"]["last_preterminal"]["panels"]["heldout"][
                    "partner_style"
                ]
                for a in ("prior", "warmstart_adapter")
            ]
            both_improved += int(
                all(
                    np.mean(np.array(ps[1][p])[-2:] - np.array(ps[0][p])[-2:]) > 0
                    for p in (0, 1)
                )
            )
        positive = contrasts["personal_vs_prior"]
        positive_gate = positive["mean"] >= 0.10 and positive["positive_seeds"] >= 3
        learning = all(
            contrasts[k]["mean"] >= threshold and contrasts[k]["positive_seeds"] >= 3
            for k, threshold in (
                ("warm_vs_prior", 0.10),
                ("warm_vs_shuffle", 0.05),
                ("warm_vs_scratch", 0.05),
            )
        )
        learning = (
            learning
            and both_improved >= 6
            and contrasts["warm_conditional_vs_prior"]["mean"] > 0
        )
        usefulness = all(
            contrasts[k]["mean"] >= 0 for k in ("tasks_vs_prior", "turns_vs_prior")
        )
        blocks[block] = {
            "seeds": seed_rows,
            "contrasts": contrasts,
            "absolute": absolute,
            "retention": retention,
            "opposite_worlds": opposite_count,
            "both_partners_improved": both_improved,
            "gates": {
                "positive": positive_gate,
                "learning": bool(learning),
                "retention": all(r["passed"] for r in retention),
                "usefulness": usefulness,
            },
        }
    passed = all(all(b["gates"].values()) for b in blocks.values())
    positive = all(b["gates"]["positive"] for b in blocks.values())
    learning = all(b["gates"]["learning"] for b in blocks.values())
    retention = all(b["gates"]["retention"] for b in blocks.values())
    decision = (
        "candidate_for_later_confirmation_no_gpu_this_pr"
        if passed
        else "stop_check_reference_or_context"
        if not positive
        else "stop_study_birth_retention"
        if learning and not retention
        else "stop_current_coefficient_update_no_expansion"
    )
    return {
        "blocks": blocks,
        "passed": passed,
        "decision": decision,
        "independent_clusters": 4,
        "health_order": list(TEST_H),
        "not_significance_testing": True,
    }


def export_rows(index, out):
    records = []
    for case in index:
        for arm, metrics in case["arms"].items():
            for checkpoint, snap in metrics["checkpoints"].items():
                for part, panel in snap["panels"].items():
                    for hi, health in enumerate(TEST_H):
                        records.append(
                            {
                                "seed": case["environment_seed"],
                                "world": str(case["world"]),
                                "block": case["block"],
                                "arm": arm,
                                "checkpoint": checkpoint,
                                "updates": snap["updates"],
                                "terminal": snap["terminal"],
                                "part": part,
                                "health": health,
                                **{
                                    k: panel[k][hi]
                                    for k in (
                                        "answer",
                                        "style",
                                        "conditional_style",
                                        "death",
                                    )
                                },
                            }
                        )
    with (out / "panels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    index = json.loads((args.run / "index.json").read_bytes())
    result = summary(index)
    write(args.run / "analysis.json", result)
    export_rows(index, args.run)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "blocks": {
                    k: {"gates": v["gates"], "contrasts": v["contrasts"]}
                    for k, v in result["blocks"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
