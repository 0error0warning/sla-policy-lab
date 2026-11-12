"""Compare fresh ordinary-model decisions against the frozen first-encounter measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import describe
from experiments.first_fatal.design import BIRTHS
from experiments.individual_life.design import action_at


def run(inputs, output, dest):
    cases = json.loads((inputs / "witness-cases.json").read_bytes())
    with np.load(inputs / "witness-expected.npz") as z:
        logged = z["arrays"]
    with np.load(output / "restored-probabilities.npz") as z:
        actual = z["arrays"]
    assert actual.shape == logged.shape == (len(cases) * 6, 6, 4)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert (
        receipt["restored_lives"] == len(cases)
        and receipt["training_steps"] == 0
        and receipt["backbone_restored"]
    )
    a = actual.reshape(len(cases), 6, 6, 4)
    b = logged.reshape(a.shape)
    per_panel = np.max(np.abs(a - b), axis=(2, 3))
    changes = []
    for fresh, observed, case in zip(a, b, cases, strict=True):
        before = action_at(observed[0, 1], case["u"])
        after = action_at(fresh[0, 1], case["u"])
        if before != after:
            changes.append(
                {
                    "seed": case["seed"],
                    "birth": case["birth"],
                    "rule": case["rule"],
                    "recorded_action": before,
                    "fresh_action": after,
                    "fatal_class_changed": (before >= 2) != (after >= 2),
                }
            )
    result = {}
    for origin in BIRTHS:
        group = []
        for cur, old, rec in zip(a, b, cases, strict=True):
            if rec["birth"] != origin or rec["rule"] != "true1":
                continue
            values = {"seed": rec["seed"]}
            for t, src in (("fresh", cur), ("recorded", old)):
                p = src[0, :, 2:].astype(float).sum(-1)
                for k, idx in (("birth", 0), ("all_wrong", 2), ("practice_wrong", 3)):
                    values[t + "_vs_" + k] = float(p[1] - p[idx])
            group.append(values)
        result[origin] = {k: describe(group, k) for k in group[0] if k != "seed"}
    write(
        dest,
        {
            "restored_lives": len(cases),
            "complete_panels": int(per_panel.size),
            "exact_panels": int((per_panel == 0).sum()),
            "max_abs_probability_error": float(per_panel.max()),
            "original_common_draw_choices_changed": len(changes),
            "common_draw_fatal_class_changed": sum(
                x["fatal_class_changed"] for x in changes
            ),
            "changed_choices": changes,
            "actual_first_vs_birth_and_mismatches": result,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("inputs", "output", "result"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.output, a.result)
