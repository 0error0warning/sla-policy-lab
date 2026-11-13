"""Separate original-input transport, control retraining and within-allocation treatment effects."""

from io import BytesIO
from zipfile import ZipFile

import numpy as np

from experiments.order_transfer.analysis import load, write_json
from experiments.paraphrase_transfer.analysis import DANGER, MEASURES, SAFE


def differences(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    per = d.reshape((-1, d.shape[-1])).max(axis=1)
    return {
        "vectors": len(per),
        "exact_vectors": int((per == 0).sum()),
        "max_pp": float(100 * d.max()),
        "p95_pp": float(100 * np.quantile(per, 0.95)),
    }


def compare(inputs, primary, repeat, parent, dest, first_summary, repeat_summary):
    cases = load(inputs / "phrase-cases.json")
    keys = ("before_text_extension", "after_text_extension", "after_calibration")
    with (
        np.load(repeat / "legacy-interface.npz") as stages,
        np.load(inputs / "phrase-anchors.npz") as anchors,
    ):
        reference = anchors["probabilities"]
        legacy = {
            k: {
                "parent": differences(stages[k], reference),
                "before_extension": differences(stages[k], stages[keys[0]]),
            }
            for k in keys
        }
    groups = {"primary": {}, "repeat": {}}
    weights = []
    with ZipFile(parent / "gpu/ordmix/results.zip") as old:
        for label, out in (("primary", primary), ("repeat", repeat)):
            hist = {
                "pre_healthy": [],
                "pre_original_successor": [],
                "control_healthy": [],
                "restored_control_healthy": [],
                "parent_input": [],
            }
            if label == "repeat":
                hist["slot_rearrangement"] = []
            for c in cases:
                key = f"s{c['seed']}/{c['birth']}"
                d = out / "cases" / key
                with (
                    np.load(d / "probabilities.npz") as a,
                    np.load(
                        BytesIO(old.read("output/cases/" + key + "/probabilities.npz"))
                    ) as b,
                ):
                    hist["pre_healthy"].append(
                        (a["healthy"][:, :, 0], b["healthy"][:, :, 0])
                    )
                    hist["control_healthy"].append(
                        (a["healthy"][:, :, 1], b["healthy"][:, :, 2])
                    )
                    hist["restored_control_healthy"].append(
                        (a["healthy"][:, :, 3], b["healthy"][:, :, 2])
                    )
                    orig = c["original_context"]["layout"]
                    observed = np.stack(
                        (a["original"], a["own"][0, orig], a["successor"])
                    )[:, [0, 3]]
                    archived = np.stack(
                        (b["risk"][0, orig], b["risk"][1, orig], b["successor"])
                    )[:, [0, 2]]
                    hist["pre_original_successor"].append(
                        (observed[:, 0], archived[:, 0])
                    )
                    hist["parent_input"].append((observed, archived))
                if label == "repeat":
                    with np.load(d / "slot-check.npz") as slot:
                        hist["slot_rearrangement"].append(
                            (slot["alternate"], slot["current"])
                        )
                    with (
                        np.load(primary / "cases" / key / "post-updates.npz") as f,
                        np.load(d / "post-updates.npz") as t,
                    ):
                        weights.append(
                            {k: bool(np.array_equal(f[k], t[k])) for k in f.files}
                        )
            for k, pairs in hist.items():
                a, b = zip(*pairs, strict=True)
                groups[label][k] = differences(np.stack(a), np.stack(b))
    results = {}
    for endpoint in (*DANGER, *SAFE, "held_word_other_partner"):
        results[endpoint] = {}
        for metric in ("answer",) if endpoint in SAFE else MEASURES:
            a, b = [
                s["groups"]["all"][endpoint][metric]["augmented_minus_original"]
                for s in (first_summary, repeat_summary)
            ]
            results[endpoint][metric] = {
                "primary_pp": a["mean"],
                "repeat_pp": b["mean"],
                "repeat_minus_primary_pp": b["mean"] - a["mean"],
                "primary95": a["ci95"],
                "repeat95": b["ci95"],
            }
    result = {
        "same_48_experiences": True,
        "additional_independent_lives": 0,
        "legacy_interface_on_repeat": legacy,
        "drift": groups,
        "identical_retrained_weights_across_allocations": {
            k: sum(x[k] for x in weights) for k in weights[0]
        },
        "effects": results,
        "criteria_primary": first_summary["criteria"],
        "criteria_repeat": repeat_summary["criteria"],
        "practical_primary": first_summary["practical_gate"],
        "practical_repeat": repeat_summary["practical_gate"],
    }
    write_json(dest, result)
    return result
