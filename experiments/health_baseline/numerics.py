"""Repeatability of actual matched-tape updates and independent ordinary-Qwen snapshot installs."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import BIRTHS
from experiments.health_baseline.analysis import PAIRS, ci, read_replay
from experiments.individual_life.design import action_at


def repeats(first, second):
    drift = []
    focused = []
    steps = []
    archived = {"primary": [], "repeat": []}
    for fp in sorted(first.glob("s*/*/life.json")):
        original = json.loads(fp.read_bytes())
        repeated = json.loads((second / fp.relative_to(first)).read_bytes())
        assert all(original[k] == repeated[k] for k in ("birth", "seed", "turns"))
        x = read_replay(fp.parent)
        y = read_replay(second / fp.parent.relative_to(first))
        for i, (a, b) in enumerate(zip(x, y, strict=True)):
            assert all(
                a[k] == b[k]
                for k in (
                    "turn",
                    "health",
                    "after",
                    "action",
                    "reward",
                    "global_baseline",
                    "local_baseline",
                    "u",
                    "index",
                )
            )
            p, q = np.asarray(a["probs"]), np.asarray(b["probs"])
            delta = float(np.max(np.abs(p - q)))
            drift.append(delta)
            for label, event in (("primary", a), ("repeat", b)):
                archived[label].append(
                    (
                        event["source_global_max_error"],
                        event["source_global_hash_match"],
                    )
                )
            if not i or not (x[i - 1]["practice"] and x[i - 1]["action"] >= 2):
                continue
            u = a["u"]
            dels = {}
            for key, (j, k) in PAIRS.items():
                choices = [(action_at(v[j], u), action_at(v[k], u)) for v in (p, q)]
                flags = [t != s for t, s in choices]
                dels[key] = (
                    flags[0],
                    flags[1],
                    flags[0] != flags[1],
                    choices[0] != choices[1],
                )
            steps.append(
                [
                    [v["actual_step_norm"] for v in group[i - 1]["updates"]]
                    for group in (x, y)
                ]
            )
            focused.append((delta, dels))
    assert len(focused) == 680
    result = {}
    for key in PAIRS:
        a = np.array([f[key] for _, f in focused], int)
        result[key] = {
            "flips_first": int(a[:, 0].sum()),
            "flips_repeat": int(a[:, 1].sum()),
            "different_flip_status": int(a[:, 2].sum()),
            "different_source_or_destination_action": int(a[:, 3].sum()),
        }
    return {
        "global_vs_original_pr9": {
            run: {
                "turns": len(vals),
                "exact_semantic_distributions": sum(d == 0 for d, _ in vals),
                "exact_post_update_parameters": sum(v for _, v in vals),
                "max_probability_difference": max(d for d, _ in vals),
            }
            for run, vals in archived.items()
        },
        "median_actual_steps": {
            label: {
                "global": float(g),
                "local_own": float(l),
                "local_common": float(c),
                "common_to_global_ratio": float(c / g),
            }
            for label, (g, l, c) in zip(
                ("primary", "repeat"), np.median(steps, axis=0), strict=True
            )
        },
        "all_turns": len(drift),
        "fully_equal_panels": sum(e == 0 for e in drift),
        "p95": float(np.quantile(drift, 0.95)),
        "max_probability_error": max(drift),
        "practice_answer_successors": len(focused),
        "max_practice_answer_successor_error": max(x for x, _ in focused),
        "paired_events": result,
    }


def restoration(inputs, output):
    cases = json.loads((inputs / "witness-cases.json").read_bytes())
    with np.load(inputs / "witness-expected.npz") as z:
        logged = z["arrays"]
    with np.load(output / "restored-probabilities.npz") as z:
        fresh = z["arrays"]
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert logged.shape == fresh.shape == (sum(len(c["checks"]) for c in cases), 6, 4)
    assert receipt["training_steps"] == 0 and receipt["backbone_restored"]
    diffs = np.abs(logged - fresh).max(axis=(1, 2))
    off = 0
    risks = []
    flips = []
    for c in cases:
        n = len(c["checks"])
        a = logged[off].astype(float)
        b = fresh[off].astype(float)
        off += n
        u = c["u"]
        old, new = action_at(a[1], u), action_at(b[1], u)
        flips.append(
            {
                "kind": c["kind"],
                "birth": c["birth"],
                "rule": c["rule"],
                "seed": c["seed"],
                "changed": int(old != new),
                "class_changed": int((old >= 2) != (new >= 2)),
            }
        )
        if c["kind"] == "live" and c["rule"] == "local_true":
            r = {"seed": c["seed"], "birth": c["birth"]}
            for key, j in (("birth", 0), ("wrong", 2), ("global", 3)):
                r["observed_" + key] = float(a[1, 2:].sum() - a[j, 2:].sum())
                r["restored_" + key] = float(b[1, 2:].sum() - b[j, 2:].sum())
            risks.append(r)
    by = {}
    for origin in BIRTHS:
        group = sorted(
            (r for r in risks if r["birth"] == origin), key=lambda r: r["seed"]
        )
        by[origin] = {
            k: ci([r["seed"] for r in group], [r[k] for r in group])
            for k in group[0]
            if k not in ("birth", "seed")
        }
    return {
        "cases": len(cases),
        "panels": len(diffs),
        "exact_panels": int((diffs == 0).sum()),
        "max_absolute_difference": float(diffs.max()),
        "p95": float(np.quantile(diffs, 0.95)),
        "focal_by_kind": {
            k: {
                "n": sum(v["kind"] == k for v in flips),
                "semantic_changes": sum(v["changed"] for v in flips if v["kind"] == k),
                "answer_class_changes": sum(
                    v["class_changed"] for v in flips if v["kind"] == k
                ),
            }
            for k in ("live", "replay")
        },
        "changed_deployed_draws": [v for v in flips if v["changed"]],
        "local_fatal_contrasts": by,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("first", "repeat", "inputs", "restored", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write(
        a.output,
        {
            "replay_repeat": repeats(a.first, a.repeat),
            "restored": restoration(a.inputs, a.restored),
        },
    )
