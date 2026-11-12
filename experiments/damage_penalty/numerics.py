"""Independent re-training on unchanged histories and re-scoring of identical saved weights."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.damage_penalty.analysis import intervals
from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import BIRTHS
from experiments.health_baseline.analysis import read_replay
from experiments.individual_life.design import action_at


def retraining(primary, repeat, expected_risky_answers=652):
    drift = []
    t0 = []
    focus = []
    source = {"primary": [], "repeat": []}
    for f in sorted(primary.glob("s*/*/life.json")):
        left = read_replay(f.parent)
        right = read_replay(repeat / f.parent.relative_to(primary))
        for a, b in zip(left, right, strict=True):
            assert all(
                a[k] == b[k]
                for k in (
                    "turn",
                    "index",
                    "action",
                    "health",
                    "after",
                    "reward",
                    "local_baseline",
                    "signals",
                    "u",
                )
            )
            err = float(np.abs(np.array(a["probs"]) - np.array(b["probs"])).max())
            drift.append(err)
            if a["turn"] == 1:
                t0.append(err)
            for k, e in (("primary", a), ("repeat", b)):
                source[k].append(
                    (e["source_one_hash_match"], e["source_one_max_error"])
                )
        for (p, a), (_, b) in zip(
            itertools.pairwise(left), itertools.pairwise(right), strict=True
        ):
            if p["practice"] and p["action"] >= 2:
                pair = []
                for v in (a, b):
                    q = np.array(v["probs"])
                    diff = float(q[4, 2:].sum() - q[1, 2:].sum())
                    pair.append(
                        (diff, action_at(q[4], v["u"]) != action_at(q[1], v["u"]))
                    )
                focus.append(pair)
    x = np.asarray(focus, dtype=float)
    assert x.shape == (expected_risky_answers, 2, 2)
    return {
        "n": len(drift),
        "exact_panels": sum(v == 0 for v in drift),
        "p95": float(np.quantile(drift, 0.95)),
        "max": max(drift),
        "max_at_identical_birth": max(t0),
        "reference_reproduction": {
            k: {
                "exact_post_update_parameters": sum(h for h, _ in v),
                "exact_distributions": sum(e == 0 for _, e in v),
                "max_probability_difference": max(e for _, e in v),
            }
            for k, v in source.items()
        },
        "risk_common_fork_mean_delta": [float(v) for v in x[:, :, 0].mean(0)],
        "risk_common_fork_sample_flips": [int(v) for v in x[:, :, 1].sum(0)],
        "events_disagreeing_on_flip": int((x[:, 0, 1] != x[:, 1, 1]).sum()),
        "independent_new_seeds": 0,
    }


def reinstalls(
    inputs,
    output,
    *,
    focal_arm="two_true",
    contrasts=(("birth", 1, 0), ("mismatch", 1, 2), ("one", 1, 3)),
    estimate=intervals,
    summary_key="two_fatal_contrasts",
):
    cases = json.loads((inputs / "witness-cases.json").read_bytes())
    with np.load(inputs / "witness-expected.npz") as z:
        expected = z["arrays"]
    with np.load(output / "restored-probabilities.npz") as z:
        restored = z["arrays"]
    meta = json.loads((output / "receipt.json").read_bytes())
    assert (
        expected.shape == restored.shape == (sum(len(c["checks"]) for c in cases), 6, 4)
    )
    assert meta["backbone_restored"] and meta["reinstall"]["training_steps"] == 0
    changes = []
    risk = []
    off = 0
    for c in cases:
        a, b = expected[off].astype(float), restored[off].astype(float)
        off += len(c["checks"])
        label = action_at(a[1], c["u"])
        new = action_at(b[1], c["u"])
        if label != new:
            changes.append(
                {
                    "seed": c["seed"],
                    "birth": c["birth"],
                    "arm": c["rule"],
                    "kind": c["kind"],
                    "recorded": label,
                    "restored": new,
                    "answer_class_changed": (label >= 2) != (new >= 2),
                }
            )
        if c["kind"] == "live" and c["rule"] == focal_arm:
            row = {"seed": c["seed"], "birth": c["birth"]}
            for m, array in (("recorded", a), ("restored", b)):
                ans = array[:, 2:].sum(1)
                for key, a_idx, b_idx in contrasts:
                    row[m + "_vs_" + key] = float(ans[a_idx] - ans[b_idx])
            risk.append(row)
    diffs = np.abs(expected.astype(float) - restored.astype(float)).max(axis=(1, 2))
    grouped = {}
    for origin in BIRTHS:
        vals = [r for r in risk if r["birth"] == origin]
        grouped[origin] = {
            k: estimate(vals, [v[k] for v in vals])
            for k in vals[0]
            if k not in ("seed", "birth")
        }
    return {
        "cases": len(cases),
        "ordinary_six_row_panels": len(diffs),
        "exact_panels": int((diffs == 0).sum()),
        "max": float(diffs.max()),
        "p95": float(np.quantile(diffs, 0.95)),
        "changed_draws": changes,
        "live_first_survival_changes": sum(
            c["kind"] == "live" and c["answer_class_changed"] for c in changes
        ),
        summary_key: grouped,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("first", "repeat", "inputs", "restored", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write(
        a.output,
        {
            "replay": retraining(a.first, a.repeat),
            "snapshots": reinstalls(a.inputs, a.restored),
        },
    )
