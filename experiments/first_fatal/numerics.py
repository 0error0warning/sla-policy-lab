"""Localize cross-run numerical drift using frozen birth outputs and matched decision contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records
from experiments.first_fatal.design import BIRTHS


def distribution(errors):
    v = np.asarray(errors, float)
    return {
        "n": len(v),
        "exact": int((v == 0).sum()),
        "median": float(np.median(v)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
    }


def store(bank, index, probs, within):
    same = bank.setdefault(int(index), probs)
    within.append(float(np.max(np.abs(same - probs))))


def frozen_bank(root):
    bank = {}
    within = []
    for f in sorted(root.glob("s*/blank/*/life.json")):
        for e in records(f.parent):
            store(bank, e["index"], np.array(e["probs"][0]), within)
        with np.load(f.parent / "predecision.npz") as z:
            for i, p in zip(z["indices"], z["panel"][0, :, 0], strict=True):
                store(bank, i, p, within)
    return bank, max(within)


def compare(a, b):
    common = sorted(a.keys() & b.keys())
    return distribution([np.max(np.abs(a[i] - b[i])) for i in common])


def private_effect(q, cases, positions):
    values = q[cases][:, positions, :, 2:].astype(float).sum(-1)
    return values[:, :, 1] - values[:, :, 0]


def run(pilot, rep, inputs, output, dest):
    a, ae = frozen_bank(pilot)
    b, be = frozen_bank(rep)
    cases = json.loads((inputs / "witness-cases.json").read_bytes())
    n = len(cases)
    with np.load(inputs / "witness-expected.npz") as z:
        old = z["arrays"].reshape(n, 6, 6, 4)
    with np.load(output / "restored-probabilities.npz") as z:
        new = z["arrays"].reshape(old.shape)
    fresh = {}
    within = []
    for j, c in enumerate(cases):
        if c["birth"] == "blank":
            for k, (i, _h) in enumerate(c["checks"]):
                store(fresh, i, new[j, k, 0], within)
    estimates = {}
    for birth in BIRTHS:
        take = [
            j
            for j, c in enumerate(cases)
            if c["birth"] == birth and c["rule"] == "true1"
        ]
        part = {}
        for label, indices in (
            ("real_first", [0]),
            ("same_request_healthy", [2, 4]),
            ("sampled_heldout_healthy", [3, 5]),
        ):
            original = private_effect(old, take, indices)
            rerun = private_effect(new, take, indices)
            drift = rerun - original
            part[label] = {
                "recorded_mean": float(original.mean()),
                "fresh_mean": float(rerun.mean()),
                "mean_signed_difference": float(drift.mean()),
                "max_individual_input_difference": float(np.abs(drift).max()),
            }
        estimates[birth] = part
    write(
        dest,
        {
            "within_pilot_frozen_max": ae,
            "within_replication_frozen_max": be,
            "within_fresh_frozen_max": max(within),
            "untrained_birth_across_main_waves": compare(a, b),
            "untrained_birth_pilot_to_fresh": compare(a, fresh),
            "untrained_birth_replication_to_fresh": compare(b, fresh),
            "paired_effect_sensitivity": estimates,
            "all_panel_abs_error": distribution(
                np.abs(old - new).max(axis=(2, 3)).ravel()
            ),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("pilot", "rep", "inputs", "output", "result"):
        p.add_argument("--" + k, type=Path, required=True)
    v = p.parse_args()
    run(v.pilot, v.rep, v.inputs, v.output, v.result)
