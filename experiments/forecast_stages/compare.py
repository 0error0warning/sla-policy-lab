"""Compare two paired technical runs, and summarize the full dense chronology."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import PILOT

from .design import PARENT_RUNS, STAGES


def concentration_sensitivity(z):
    """Match earlier to later responses on the same held request, distinguished by later concentration."""
    p = z["prob"][:, 1, 0].astype(float)
    rates = z["q"][0, 1, :, 1, 1:]
    sign = np.where(rates[:, 0] > rates[:, 1], 1.0, -1.0)
    delta = p[:, :, 1] - p[:, :, 2]
    pref = 100 * (delta[:, :, 2] - delta[:, :, 3]) * sign
    tv = 50 * abs(delta).sum(-1)
    concentrated = p[:, :, 2].max(-1) >= 0.9
    groups = {
        "already_concentrated": concentrated[1] & concentrated[3],
        "newly_concentrated": ~concentrated[1] & concentrated[3],
        "released": concentrated[1] & ~concentrated[3],
        "never_at_endpoints": ~concentrated[1] & ~concentrated[3],
    }
    records = []
    for name, mask in groups.items():
        r = {"trajectory": name, "contexts": int(mask.sum())}
        for k in (1, 3):
            r[STAGES[k] + "_tv"] = float(tv[k, mask].mean()) if mask.any() else None
            r[STAGES[k] + "_preference"] = (
                float(pref[k, mask].mean()) if mask.any() else None
            )
        records.append(r)
    return records


def summarize(first, second, analysis):
    dense, match, technical, stages, concentration = [], [], [], [], []
    for tag, root in (("first", first), ("second", second)):
        with (analysis / tag / "dense.csv").open(encoding="utf8", newline="") as f:
            rows = list(csv.DictReader(f))
        g = json.loads((analysis / tag / "summary.json").read_bytes())
        for seed in PILOT:
            with np.load(root / "lives" / f"s{seed}" / "stages.npz") as z:
                concentration.extend(
                    {"run": tag, "seed": seed, **r}
                    for r in concentration_sensitivity(z)
                )
            seq = [r for r in rows if int(r["seed"]) == seed]
            vals = np.array([float(r["oracle_preference"]) for r in seq])
            tv = np.array([float(r["oracle_tv"]) for r in seq])
            learned = np.array([float(r["learned_preference"]) for r in seq])
            sat = np.array([float(r["neutral_concentration"]) for r in seq])

            def first_hit(mask):
                hit = np.flatnonzero(mask)
                return int(hit[0]) if len(hit) else None

            max_at = int(vals.argmax())
            dense.append(
                {
                    "run": tag,
                    "seed": seed,
                    "updates": len(vals) - 1,
                    "first_distribution_effect": first_hit(tv > 1e-8),
                    "first_mean_tv_1pp": first_hit(tv >= 1),
                    "first_favorable_1pp": first_hit(vals >= 1),
                    "favorable_opportunities": int(np.sum(vals >= 1)),
                    "unfavorable_opportunities": int(np.sum(vals <= -1)),
                    "peak_favorable": float(vals[max_at]),
                    "peak_at": max_at,
                    "min_favorable": float(vals.min()),
                    "largest_mean_tv": float(tv.max()),
                    "largest_mean_tv_at": int(tv.argmax()),
                    "first_all_eight_concentrated": first_hit(sat == 100),
                    "learned_min": float(learned.min()),
                    "learned_max": float(learned.max()),
                }
            )
            caps = json.loads(
                (root / "lives" / f"s{seed}" / "anchors.json").read_bytes()
            )
            for old, entry in caps.items():
                assert old in PARENT_RUNS
                match.append(dict(run=tag, seed=seed, parent_run=old, **entry))
        for st in STAGES:
            d = g["by_stage"][st + ".oracle_neutral"]
            e = g["by_stage"][st + ".learned_neutral"]
            stages.append(
                {
                    "run": tag,
                    "stage": st,
                    "oracle_pref": d["preference"]["mean"],
                    "oracle_positive": d["preference"]["positive"],
                    "learned_pref": e["preference"]["mean"],
                    "oracle_healthy": d["healthy_difference"]["mean"],
                    "oracle_tv": d["mean_tv"]["mean"],
                    "neutral_concentration": g["by_stage"][st + ".concentration"][
                        "mean"
                    ],
                    "actual_expression_swaps": d["counts"]["sample_exchanged"],
                    "actual_action_changes": d["counts"]["sample_changed"],
                    "samples": d["counts"]["total"],
                    "useful": g["criteria"][st + ".oracle"]["useful"],
                    "early_then_locked": g["early_use_then_lock"],
                }
            )
    for seed in PILOT:
        relative = Path("lives") / f"s{seed}"
        a, b = first / relative, second / relative
        with (
            np.load(a / "states.npz") as sa,
            np.load(b / "states.npz") as sb,
            np.load(a / "stages.npz") as za,
            np.load(b / "stages.npz") as zb,
        ):
            assert np.array_equal(sa["health"], sb["health"]) and np.array_equal(
                sa["uniforms"], sb["uniforms"]
            )
            diff = 100 * abs(za["prob"].astype(float) - zb["prob"])
            technical.append(
                {
                    "seed": seed,
                    "states": len(sa["sha"]),
                    "matching_hashes": int((sa["sha"] == sb["sha"]).sum()),
                    "matching_stages": int(
                        (sa["sha"][sa["selected"]] == sb["sha"][sb["selected"]]).sum()
                    ),
                    "birth_probability_max_pp": float(diff[0].max()),
                    "early_probability_max_pp": float(diff[1].max()),
                    "middle_probability_max_pp": float(diff[2].max()),
                    "late_probability_max_pp": float(diff[3].max()),
                    "learned_forecast_max_pp": float(
                        100 * abs(za["q"][:, :, :, 0] - zb["q"][:, :, :, 0]).max()
                    ),
                }
            )
    for name, rr in (
        ("dense_by_life", dense),
        ("parent_anchors", match),
        ("between_runs", technical),
        ("stages_compared", stages),
        ("concentration_by_context", concentration),
    ):
        csv_out(analysis / (name + ".csv"), rr)
    receipt = {
        tag: json.loads((src / "receipt.json").read_bytes())
        for tag, src in (("first", first), ("second", second))
    }
    result = {
        "dense_life_states": sum(r["states"] for r in technical),
        "technical_matching_states": sum(r["matching_hashes"] for r in technical),
        "source_seed_count": len(PILOT),
        "technical_runs": 2,
        "max_between_birth_pp": max(r["birth_probability_max_pp"] for r in technical),
        "max_between_late_pp": max(r["late_probability_max_pp"] for r in technical),
        "pair_checks": sum(x["full_panel_comparisons"] for x in receipt.values()),
        "max_native_error": max(x["max_native_cache_error"] for x in receipt.values()),
        "max_lane_error": max(x["max_identical_lane_error"] for x in receipt.values()),
    }
    write(analysis / "verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("first", "second", "analysis"):
        p.add_argument("--" + k, required=True, type=Path)
    print(json.dumps(summarize(**vars(p.parse_args()))))
