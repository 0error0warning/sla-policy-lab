"""Recover PR #15's real experiences, calibrated radius, exact four-order control and anchors."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.internal_body.release import verify_manifest
from experiments.order_transfer.design import SEEDS, rotations
from experiments.paraphrase_transfer.design import DEFAULT_WORDS, bank

PARENT = Path("artifacts/model-studies/2026-11-13/order-transfer")


def build(parent, dest):
    verify_manifest(parent)
    dest.mkdir(parents=True, exist_ok=True)
    base = parent / "gpu/ordmix"
    weights = {}
    anchors = []
    with (
        ZipFile(base / "inputs.zip") as inputs,
        ZipFile(base / "results.zip") as results,
        np.load(BytesIO(inputs.read("order-weights.npz"))) as z,
    ):
        cases = json.loads(inputs.read("order-cases.json"))
        assert [(c["seed"], c["birth"]) for c in cases] == [
            (s, b) for s in SEEDS for b in ("blank", "population")
        ]
        for c in cases:
            path = f"output/cases/s{c['seed']}/{c['birth']}/"
            data = json.loads(results.read(path + "update.json"))
            assert (
                c["signal"] == data["fixed_signal"] < 0
                and data["common_importance_ratio"] == c["importance_ratio"]
            )
            assert (
                list(rotations(c["original_context"]["layout"])) == c["train_layouts"]
            )
            assert (
                c["original_context"]["task"],
                c["original_context"]["wording"],
            ) == (4, 3)
            c["radius"] = data["quota"]
            c["pr15_hash"] = data["rotate_hash"]
            c["pr15_multiplier"] = data["rotate_multiplier"]
            c["pr15_gradient_norm"] = data["mean_rotation_gradient_norm"]
            with np.load(BytesIO(results.read(path + "post-updates.npz"))) as posts:
                for h, v in (
                    (c["pre_hash"], z[c["pre_hash"]]),
                    (c["pr15_hash"], posts["rotate"]),
                ):
                    assert (
                        v.dtype.str == "<u4"
                        and v.shape == (13408,)
                        and sha256(v.tobytes()).hexdigest() == h
                    )
                    weights[h] = v
            with np.load(BytesIO(results.read(path + "probabilities.npz"))) as probs:
                o = c["original_context"]["layout"]
                anchors.append(
                    np.stack(
                        (probs["risk"][0, o], probs["risk"][1, o], probs["successor"])
                    )[:, [0, 2]]
                )
            c["healthy_reference"] = [
                [
                    {
                        k: e[k]
                        for k in (
                            "turn",
                            "index",
                            "health",
                            "after",
                            "action",
                            "completed",
                        )
                    }
                    for e in life
                ]
                for i, life in enumerate(
                    json.loads(results.read(path + "healthy-lives.json"))
                )
                if i in (0, 2)
            ]
        for name in ("architecture.npz", "birth-source.npz"):
            (dest / name).write_bytes(inputs.read(name))
    bank(DEFAULT_WORDS)
    (dest / "wording.json").write_bytes(DEFAULT_WORDS.read_bytes())
    write(dest / "phrase-cases.json", cases)
    np.savez_compressed(dest / "phrase-weights.npz", **weights)
    np.savez_compressed(dest / "phrase-anchors.npz", probabilities=np.array(anchors))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, default=PARENT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.parent, a.out)
