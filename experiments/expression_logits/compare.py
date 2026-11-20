"""Check two actual native-head runs and retain each run's link to the archived switch."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import PILOT

from .design import CASE_ID, CASE_SEED, TWIN_ID


def table(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def compare(first, second, out):
    results = {}
    rows = []
    max_z = max_p = 0.0
    agree = 0
    total = 0
    for seed in PILOT:
        with (
            np.load(first / "lives" / f"s{seed}" / "scores.npz") as a,
            np.load(second / "lives" / f"s{seed}" / "scores.npz") as b,
        ):
            assert np.array_equal(a["keys"], b["keys"])
            d = abs(a["scores"].astype(float) - b["scores"])
            e = abs(a["prob"].astype(float) - b["prob"])
            same = np.all(d == 0, axis=-1)
            max_z = max(max_z, float(d.max()))
            max_p = max(max_p, float(100 * e.max()))
            agree += int(same.sum())
            total += same.size
            rows.append(
                {
                    "seed": seed,
                    "exact_score_vectors": int(same.sum()),
                    "views": int(same.size),
                    "max_score_difference": float(d.max()),
                    "max_policy_difference_pp": float(100 * e.max()),
                    "identical_hidden_words": bool(
                        np.array_equal(a["hidden_bits"], b["hidden_bits"])
                    ),
                }
            )
    csv_out(out / "technical_by_life.csv", rows)
    for tag in ("first", "second"):
        summary = json.loads((out / tag / "summary.json").read_bytes())
        audit = json.loads((out / tag / "audit.json").read_bytes())
        cases = table(out / tag / "case_views.csv")

        anchors = {}
        for prompt in (CASE_ID, TWIN_ID):
            values = [
                float(r["probability_max_drift_pp"])
                for r in cases
                if int(r["seed"]) == CASE_SEED and int(r["prompt"]) == prompt
            ]
            assert len(values) == 3
            anchors[prompt] = max(values)

        observed = summary["cases"]
        matched = observed[str(CASE_ID)] == {
            "learned": "trade",
            "oracle": "trade",
            "neutral": "help",
        } and set(observed[str(TWIN_ID)].values()) == {"help"}
        results[tag] = {
            "original_switch_and_twin": matched,
            "case_probability_drift_pp": anchors[CASE_ID],
            "twin_probability_drift_pp": anchors[TWIN_ID],
            "archive_choice_matches": summary["same_archived_choices"],
            "record_count": summary["n_rows"],
            "archive_max_pp": summary["max_drift_pp"],
            "native_cpu_error": audit["native_probability_cpu_max_error"],
        }
    result = {
        "runs": results,
        "max_native_score_difference": max_z,
        "max_between_run_probability_pp": max_p,
        "identical_vectors": agree,
        "total_vectors": total,
        "all_score_vectors_equal": bool(agree == total),
        "independent_lives": len(PILOT),
    }
    write(out / "technical.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("first", "second", "out"):
        p.add_argument("--" + n, required=True, type=Path)
    print(json.dumps(compare(**vars(p.parse_args())), indent=2))
