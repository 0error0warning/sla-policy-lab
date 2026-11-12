"""Construct a blinded reinstall set for every factual focal state and replay witness."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records

REPLAY_NAMES = (
    "global_before",
    "global",
    "local_before",
    "local",
    "local_from_global",
    "global_duplicate",
)


def build(roots, replays, dest, replay_names=REPLAY_NAMES, additional_replay=None):
    cases = []
    weights = {}
    expected = []
    dest.mkdir(parents=True, exist_ok=True)

    def remember(data, key):
        assert sha256(data.tobytes()).hexdigest() == key
        weights[key] = data
        return key

    for j, f in enumerate(
        sorted(f for root in roots for f in root.glob("s*/*/*/life.json"))
    ):
        m = json.loads(f.read_bytes())
        if m["first_fatal"] is None:
            continue
        first = records(f.parent)[m["first_fatal"] - 1]
        assert first["first_fatal"]
        yes = m["rule"].endswith("_true")
        slots = [
            "birth",
            "first",
            "first_wrong" if yes else "birth",
            "first_other" if yes else "birth",
            "first_extra" if "first_extra" in m["snapshot_hashes"] else "entry",
            "first",
        ]
        with np.load(f.parent / "parameters.npz") as z:
            keys = [remember(z[n], m["snapshot_hashes"][n]) for n in slots]
        checks = []
        with np.load(f.parent / "predecision.npz") as z:
            for i, h in enumerate(z["health"]):
                k = (j + 32 * i) % 96
                checks.extend(
                    [[first["index"], float(h)], [int(z["indices"][k]), float(h)]]
                )
                expected.extend([z["same_request"][i], z["panel"][i, k]])
        cases.append(
            {
                "kind": "live",
                "seed": m["seed"],
                "birth": m["birth"],
                "rule": m["rule"],
                "turn": first["turn"],
                "u": first["u"][0],
                "weights": keys,
                "checks": checks,
            }
        )
    sources = [(replays, replay_names)]
    if additional_replay is not None:
        sources.append(additional_replay)
    for folder, names in sources:
        for f in sorted(folder.glob("s*/*/life.json")):
            m = json.loads(f.read_bytes())
            w = m["witness"]
            with np.load(f.parent / "successor.npz") as z:
                keys = [remember(z[n], w["hashes"][n]) for n in names]
                expected.append(z["probs"])
            cases.append(
                {
                    "kind": "replay",
                    "seed": m["seed"],
                    "birth": m["birth"],
                    "rule": names[3] if additional_replay is not None else "tape",
                    "turn": w["turn"],
                    "u": w["u"],
                    "weights": keys,
                    "checks": [[w["index"], w["health"]]],
                }
            )
    assert len(cases) > 48 and sum(c["kind"] == "replay" for c in cases) == 48 * len(
        sources
    )
    write(dest / "witness-cases.json", cases)
    np.savez_compressed(dest / "witness-weights.npz", **weights)
    np.savez_compressed(dest / "witness-expected.npz", arrays=np.stack(expected))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", type=Path, nargs="+", required=True)
    p.add_argument("--replays", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.lives, a.replays, a.output)
