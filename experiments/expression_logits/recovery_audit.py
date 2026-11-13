"""Which new head scores are anchored to the old switch and the matched non-switch?"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import write
from experiments.forecast_stages.design import VIEWS
from experiments.individual_life.design import action_at
from experiments.lifetime_cooperation.analysis import csv_out

from .design import CASE_ID, CASE_SEED, TWIN_ID


def check(src, inputs, out):
    capture = json.loads((src / "parent-focus.json").read_bytes())
    receipt = json.loads((src / "receipt.json").read_bytes())
    assert receipt["parent_preparatory_bodily_writes"] == 511
    assert (
        receipt["cache_comparisons"] == receipt["model_forward_calls"]
        and receipt["training_updates"] == 0
    )
    raw = []
    with (
        np.load(inputs / "frozen-snapshots.npz") as data,
        np.load(src / "lives" / f"s{CASE_SEED}" / "scores.npz") as saved,
    ):
        keys = data["keys"]
        for i in (CASE_ID, TWIN_ID):
            n = np.flatnonzero(
                (keys[:, 0] == CASE_SEED) & (keys[:, 1] == 31) & (keys[:, 4] == i)
            ).item()
            k = np.flatnonzero(saved["rows"] == n).item()
            hsh = data["policy_sha"][data["slots"][n]].decode()
            old = data["previous_prob"][n]
            inf = capture["readouts"][str(i)]
            ps = np.array(inf["probs"])
            zz = np.array(inf["scores"])
            f = np.array(inf["q"])
            assert np.isfinite(zz).all()
            u = data["uniforms"][n, 0]
            for m, view in enumerate(VIEWS):
                base = old[m].astype(float)
                full = saved["prob"][k, m].astype(float)
                est = saved["scores"][k, m].astype(float)
                raw.append(
                    {
                        "prompt": i,
                        "view": view,
                        "source_policy_matches": hsh == inf["sha"],
                        "source_q_max_error": float(abs(f[m] - data["q"][n, m]).max()),
                        "parent_entry_max_vs_old_pp": 100
                        * float(abs(ps[m] - base).max()),
                        "restored_full_max_vs_old_pp": 100
                        * float(abs(full - base).max()),
                        "parent_vs_restored_score_max": float(abs(zz[m] - est).max()),
                        "parent_help": float(zz[m, 2]),
                        "parent_trade": float(zz[m, 3]),
                        "restored_help": float(est[2]),
                        "restored_trade": float(est[3]),
                        "archive_choice": WORDS[action_at(base, u)],
                        "parent_choice": WORDS[action_at(ps[m], u)],
                        "restored_choice": WORDS[action_at(full, u)],
                        "archive_boundary": float(base[:3].sum()),
                        "restored_boundary": float(full[:3].sum()),
                        "u": float(u),
                    }
                )
    csv_out(out / "parent_entry_case.csv", raw)
    result = {
        "exact_replayed_checkpoints": sum(
            v["exact"] for v in capture["parameter_matches"].values()
        ),
        "checkpoints": sum(v["checked"] for v in capture["parameter_matches"].values()),
        "archived_case_probability_exact": all(
            r["restored_full_max_vs_old_pp"] == 0 for r in raw
        ),
        "past_choices_matched": all(
            r["restored_choice"] == r["archive_choice"] for r in raw
        ),
        "parent_case_exact": all(
            r["parent_entry_max_vs_old_pp"] == 0
            and r["source_policy_matches"]
            and r["source_q_max_error"] < 1e-12
            for r in raw
        ),
        "max_parent_vs_restored_score": max(
            r["parent_vs_restored_score_max"] for r in raw
        ),
        "full_cached_precision_difference": receipt["cache_max"],
        "full_cached_hidden_difference": receipt["cache_hidden_max"],
    }
    write(out / "entry_verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("src", "inputs", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(check(**vars(p.parse_args()))))
