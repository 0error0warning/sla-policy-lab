"""Recover the exact signed update, two body states and semantic action of every witness."""

import argparse
import gzip
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.conditional_penalty.design import GATE_ROLES
from experiments.conditional_penalty.inputs import SOURCE_WAVES
from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import BIRTHS
from experiments.individual_life.design import rows
from experiments.internal_body.release import verify_manifest

PARENT = Path("artifacts/model-studies/2026-11-13/conditional-penalty")
CONTEXTS = (
    "original",
    "health_only",
    "successor",
    "partner_only",
    "request_only",
    "order_only",
)
FACTORS = ("partner", "frame", "task", "wording", "layout")


def row_key(record):
    return tuple(record[k] for k in FACTORS)


def checks(i, j, h, after, items, lookup):
    original, following = items[i], items[j]

    def change(fields):
        key = dict(original)
        key.update({k: following[k] for k in fields})
        return lookup[row_key(key)]

    return [
        [i, h],
        [i, after],
        [j, after],
        [change(("partner",)), after],
        [change(("frame", "task", "wording")), after],
        [change(("layout",)), after],
    ]


def build(parent, dest):
    verify_manifest(parent)
    dest.mkdir(parents=True, exist_ok=True)
    stage = parent / "gpu/gpreplay"
    items = rows()
    lookup = {row_key(r): i for i, r in enumerate(items)}
    assert len(lookup) == len(items)
    cases, weights, original, successor = [], {}, [], []
    source_sha = json.loads((stage / "receipt.json").read_bytes())["sha256"]
    assert sha256((stage / "results.zip").read_bytes()).hexdigest() == source_sha
    with (
        ZipFile(stage / "results.zip") as z,
        ZipFile(stage / "inputs.zip") as inp,
        ZipFile(BytesIO(inp.read("tapes.zip"))) as tapes,
    ):
        for seed in (*SOURCE_WAVES[0], *SOURCE_WAVES[1]):
            for birth in BIRTHS:
                short = f"s{seed}/{birth}"
                key = "output/gated/replays/" + short + "/"
                w = json.loads(z.read(key + "life.json"))["witness"]
                ev = [
                    json.loads(e)
                    for e in gzip.decompress(
                        z.read(key + "replay.jsonl.gz")
                    ).splitlines()
                ]
                source = [
                    json.loads(e)
                    for e in gzip.decompress(
                        tapes.read(short + "/events.jsonl.gz")
                    ).splitlines()
                ]
                pos = w["turn"] - 2
                e, n = ev[pos : pos + 2]
                s, t = source[pos : pos + 2]
                assert pos == next(
                    i for i, x in enumerate(ev) if x["practice"] and x["action"] >= 2
                )
                assert (
                    e["turn"] + 1 == n["turn"] == w["turn"]
                    and e["after"] == n["health"] == w["health"]
                )
                assert (
                    e["health"] > e["after"] > 0
                    and n["index"] == w["index"]
                    and n["u"] == w["u"]
                )
                for logged, factual in ((e, s), (n, t)):
                    assert all(
                        logged[k] == factual[k]
                        for k in ("turn", "health", "action", "index", "after")
                    )
                    assert all(
                        items[logged["index"]][k] == factual[k]
                        for k in ("partner", "frame", "layout")
                    )
                before, after = w["hashes"]["gate_before"], w["hashes"]["gate"]
                assert (
                    e["before_hash"][1] == before
                    and e["updates"][1]["after_hash"] == after == n["before_hash"][1]
                )
                assert (
                    abs(e["signals"][1] - (2 * e["reward"] - e["local_baseline"]))
                    < 1e-12
                )
                with np.load(
                    BytesIO(z.read(key + "successor.npz")), allow_pickle=False
                ) as arrays:
                    hashes = [w["hashes"][name] for name in GATE_ROLES]
                    for name, digest in zip(GATE_ROLES, hashes, strict=True):
                        v = arrays[name]
                        assert (
                            v.shape == (13408,)
                            and v.dtype.str == "<u4"
                            and sha256(v.tobytes()).hexdigest() == digest
                        )
                        weights[digest] = v
                    step = float(
                        np.linalg.norm(
                            weights[after].view("<f4").astype(float)
                            - weights[before].view("<f4").astype(float)
                        )
                    )
                    assert abs(step - e["updates"][1]["actual_step_norm"]) < 1e-10
                    src = arrays["probs"]
                    np.testing.assert_array_equal(src[:5], n["probs"])
                    np.testing.assert_array_equal(src[1], src[5])
                    original.append(e["probs"][3])
                    successor.append(src)
                i, j = e["index"], n["index"]
                cases.append(
                    {
                        "seed": seed,
                        "birth": birth,
                        "turn": e["turn"],
                        "wave": "pilot" if seed in SOURCE_WAVES[0] else "replication",
                        "weights": hashes,
                        "checks": checks(i, j, e["health"], n["health"], items, lookup),
                        "original_context": items[i],
                        "next_context": items[j],
                        "chosen": e["action"],
                        "next_factual_action": n["action"],
                        "original_u": e["u"],
                        "next_u": n["u"],
                        "maintenance": s["maintenance"],
                        "rul_loss": -e["reward"],
                        "baseline": e["local_baseline"],
                        "signal": e["signals"][1],
                        "parameter_step": step,
                    }
                )
    assert len(cases) == 48 and len({(c["seed"], c["birth"]) for c in cases}) == 48
    write(
        dest / "transfer-cases.json",
        {
            "contexts": CONTEXTS,
            "roles": GATE_ROLES,
            "source_sha256": source_sha,
            "cases": cases,
        },
    )
    np.savez_compressed(dest / "transfer-weights.npz", **weights)
    np.savez_compressed(
        dest / "transfer-anchors.npz",
        original_before=np.asarray(original, dtype="<f4"),
        successor=np.stack(successor),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.out)
