"""Compare two fully completed runs, preserving paired seeds and arm labels in both."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import records
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import PILOT

from .design import ARMS, MODES


def compare(first, second, analysis, out):
    a = json.loads((analysis / "gates.json").read_bytes())
    b = json.loads((out / "gates.json").read_bytes())
    decisions = []
    for seed in PILOT:
        for phase in ("replay", "live"):
            folders = [p / "lives" / f"s{seed}" / phase for p in (first, second)]
            with (
                np.load(folders[0] / "panel.npz") as z1,
                np.load(folders[1] / "panel.npz") as z2,
            ):
                born = abs(
                    z1["policy"][MODES.index("birth"), 1]
                    - z2["policy"][MODES.index("birth"), 1]
                )
                for arm in ARMS:
                    ai = MODES.index(arm)
                    p1, p2 = (z["policy"][ai, 1] for z in (z1, z2))
                    dif = abs(p1 - p2)
                    f = abs(z1[arm][1] - z2[arm][1])
                    name = "replay.jsonl" if phase == "replay" else arm + ".jsonl"
                    t1, t2 = (records(folder / name) for folder in folders)
                    n = min(len(t1), len(t2))
                    if phase == "replay":
                        assert all(
                            x["action"] == y["action"]
                            and x["maintenance"] == y["maintenance"]
                            for x, y in zip(t1, t2, strict=True)
                        )
                        before = [r["policy"][arm] for r in t1]
                        repeat = [r["policy"][arm] for r in t2]
                    else:
                        before = [r["policy"] for r in t1]
                        repeat = [r["policy"] for r in t2]
                    divergent = next(
                        (k + 1 for k in range(n) if t1[k]["action"] != t2[k]["action"]),
                        None,
                    )
                    decisions.append(
                        {
                            "seed": seed,
                            "phase": phase,
                            "arm": arm,
                            "birth_held_max_pp": 100 * born.max(),
                            "birth_held_mean_pp": 100 * born.mean(),
                            "held_forecast_max_pp": 100 * f.max(),
                            "held_policy_max_pp": 100 * dif.max(),
                            "held_policy_mean_pp": 100 * dif.mean(),
                            "held_policy_equal_cells": int((p1 == p2).all(-1).sum()),
                            "held_top_action_disagreements": int(
                                (p1.argmax(-1) != p2.argmax(-1)).sum()
                            ),
                            "first_length": len(t1),
                            "second_length": len(t2),
                            "first_action_difference": divergent,
                            "first_turn_probability_max_pp": 100
                            * np.max(abs(np.asarray(before[0]) - repeat[0])),
                            "action_sequence_equal": len(t1) == len(t2)
                            and divergent is None,
                        }
                    )
    csv_out(out / "technical_comparison.csv", decisions)
    changes = []
    for name, one in a["gates"].items():
        two = b["gates"][name]
        changes.append(
            {
                "gate": name,
                "first_mean": one["mean"],
                "second_mean": two["mean"],
                "first_positive": one["positive"],
                "second_positive": two["positive"],
                "first_pass": one["passed"],
                "second_pass": two["passed"],
            }
        )
    csv_out(out / "technical_gates.csv", changes)
    with (out / "cross_run.csv").open(encoding="utf8", newline="") as f:
        anchor = list(csv.DictReader(f))
    with np.load(first / "features.npz") as x, np.load(second / "features.npz") as y:
        assert np.array_equal(x["ids"], y["ids"])
        feature_error = float(abs(x["x"] - y["x"]).max())
        name_err = float(abs(x["name"] - y["name"]).max())
        text_err = float(abs(x["text"] - y["text"]).max())
    live = [r for r in decisions if r["phase"] == "live"]
    result = {
        "seeds": len(PILOT),
        "birth_held_max_pp": float(max(r["birth_held_max_pp"] for r in decisions)),
        "recheck_vs_19_birth_held_pp": max(
            float(r["birth_held_max_pp"]) for r in anchor
        ),
        "recheck_vs_19_terminal_held_pp": max(
            float(r["last_held_max_pp"]) for r in anchor
        ),
        "same_autonomous_actions": sum(r["action_sequence_equal"] for r in live),
        "autonomous_total": len(live),
        "all_criteria_same": all(r["first_pass"] == r["second_pass"] for r in changes),
        "replay_held_forecast_max_pp": float(
            max(r["held_forecast_max_pp"] for r in decisions if r["phase"] == "replay")
        ),
        "all_held_policy_max_pp": float(
            max(r["held_policy_max_pp"] for r in decisions)
        ),
        "kernel_feature_max_abs": feature_error,
        "raw_block_name_max_abs": name_err,
        "raw_block_words_max_abs": text_err,
        "first_prediction": a["prediction_pass"],
        "second_prediction": b["prediction_pass"],
        "first_policy": a["policy_pass"],
        "second_policy": b["policy_pass"],
    }
    write(out / "technical_summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("first", "second", "analysis", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    print(json.dumps(compare(**vars(p.parse_args())), indent=2))
