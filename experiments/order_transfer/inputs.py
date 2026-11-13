"""Prepare one immutable experience per witness; never use its successor to choose views."""

import argparse
import gzip
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import BIRTHS
from experiments.internal_body.release import verify_manifest
from experiments.order_transfer.design import SEEDS, rotations

ROOT = Path("artifacts/model-studies/2026-11-13")
SNAPSHOTS = ROOT / "snapshot-transfer"
SOURCE = ROOT / "conditional-penalty"


def build(parent, source, dest):
    verify_manifest(parent)
    verify_manifest(source)
    dest.mkdir(exist_ok=True, parents=True)
    weights = {}
    src = source / "gpu/gpreplay/results.zip"
    stage = parent / "gpu/stmap"
    with (
        ZipFile(stage / "inputs.zip") as inp,
        ZipFile(stage / "results.zip") as results,
        ZipFile(src) as orig,
        np.load(BytesIO(inp.read("transfer-weights.npz"))) as saved,
        np.load(BytesIO(results.read("output/probabilities.npz"))) as measurements,
    ):
        cases = json.loads(inp.read("transfer-cases.json"))["cases"]
        assert [(c["seed"], c["birth"]) for c in cases] == [
            (s, b) for s in SEEDS for b in BIRTHS
        ]
        anchors = measurements["primary"][:, :3, 2:4, :].copy()
        assert anchors.shape == (48, 3, 2, 4)
        for c in cases:
            key = f"output/gated/replays/s{c['seed']}/{c['birth']}/replay.jsonl.gz"
            e = json.loads(gzip.decompress(orig.read(key)).splitlines()[c["turn"] - 1])
            up = e["updates"][1]
            pre, post = c["weights"][2:4]
            for h in (pre, post):
                datum = saved[h]
                assert sha256(datum.tobytes()).hexdigest() == h
                weights[h] = datum
            assert (
                pre == e["before_hash"][1]
                and post == up["after_hash"]
                and c["chosen"] == e["action"]
                and c["signal"] == e["signals"][1] < 0
            )
            assert c["checks"][0] == [e["index"], e["health"]] and c["checks"][1] == [
                e["index"],
                e["after"],
            ]
            assert up["training_probability"] == e["probs"][3][e["action"]]
            assert abs(2 * (-c["rul_loss"]) - c["baseline"] - c["signal"]) < 1e-12
            c["pre_hash"], c["historical_post_hash"] = pre, post
            del c["weights"]
            c["behavior_probability"] = up["behavior_probability"]
            c["historical_training_probability"] = up["training_probability"]
            c["historical_step"] = up["actual_step_norm"]
            c["importance_ratio"] = up["applied_ratio"]
            c["train_layouts"] = list(rotations(c["original_context"]["layout"]))
            c["evaluation_seed"] = c["seed"] + 2000
        for k in ("architecture.npz", "birth-source.npz"):
            (dest / k).write_bytes(inp.read(k))
    write(dest / "order-cases.json", cases)
    np.savez_compressed(dest / "order-weights.npz", **weights)
    np.savez_compressed(dest / "order-anchors.npz", probabilities=anchors)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, default=SNAPSHOTS)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.parent, a.source, a.out)
