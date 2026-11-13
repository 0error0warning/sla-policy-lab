"""Locked stable histories, stage indices, and counterfactual decision draws."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import noises, rng
from experiments.individual_life.design import offers
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT, rules

PARENT_INPUT = "fcccd870a2ac7a55c2f791797253e072474d75aec46736d190920528db8bebf7"
PARENT_RUNS = {
    "first": "53e53b46cfba200271d27c4478fcdf9cbe63a32a12638991b898d80fc995428a",
    "repeat": "d947f89aa7c687555109bc501f6b034ea0dd261218b965c671e229dd4509caee",
}
STAGES = ("birth", "early", "middle", "late")
VIEWS = ("learned", "oracle", "neutral")
HEALTH = (6.0, 90.0, 300.0)
DENSE = np.array([j for j, i in enumerate(PANEL[1]) if ITEMS[i]["layout"] == 0])


def turns(n):
    """n is the number of nonterminal, completed bodily updates."""
    if n // 2 <= 8:
        raise ValueError("History too short for the locked stages")
    return (0, 8, n // 2, n)


def random_panel(seed):
    u = rng(seed, "forecast-stage-probes-v1").random((PANEL.shape[1], 2))
    return u, np.array([noises(seed, j + 1) for j in range(len(u))])


def oracle(seed):
    actual = np.array(
        [
            [offers(ITEMS[i], rules(seed, "stable", 0))[0][1:] for i in ix]
            for ix in PANEL
        ]
    )
    assert np.array_equal(actual[0], actual[1])
    return actual


def inputs(learned, truth):
    return np.stack((learned, truth, np.full_like(learned, 0.5)), axis=-2)


def shared_draw(p, u):
    return (u[..., None] > p.astype(float).cumsum(axis=-1)[..., :3]).sum(-1)


def shared_joint(left, right):
    """Exact pairing induced by the repository's single common uniform action draw."""

    def bounds(p):
        cut = np.clip(p.astype(float).cumsum(-1)[..., :3], 0, 1)
        return np.concatenate(
            (np.zeros_like(cut[..., :1]), cut, np.ones_like(cut[..., :1])), axis=-1
        )

    x, y = bounds(left), bounds(right)
    upper = np.minimum(x[..., 1:, None], y[..., None, 1:])
    lower = np.maximum(x[..., :-1, None], y[..., None, :-1])
    return np.maximum(upper - lower, 0)


def prepare(parent, dest):
    """Package only the shared initial state, factual tape and two previous run anchors."""
    dest.mkdir(parents=True, exist_ok=True)
    input_zip = parent / "gpu/partnerpred/inputs.zip"
    assert sha256(input_zip.read_bytes()).hexdigest() == PARENT_INPUT
    with ZipFile(input_zip) as z:
        for name in ("birth-source.npz", "architecture.npz", "tapes.json"):
            (dest / name).write_bytes(z.read(name))
    tapes = json.loads((dest / "tapes.json").read_bytes())
    limits = {}
    for tag, expected in PARENT_RUNS.items():
        path = (
            parent
            / "gpu"
            / ("partnerpred" if tag == "first" else "recheck")
            / "results.zip"
        )
        assert sha256(path.read_bytes()).hexdigest() == expected
        arrays = {}
        with ZipFile(path) as z:
            for seed in PILOT:
                key = f"s{seed}"
                pre = f"output/lives/{key}/replay/"
                records = [
                    json.loads(s) for s in z.read(pre + "replay.jsonl").splitlines()
                ]
                tape = tapes[str(seed)]
                assert len(tape) == len(records) and all(
                    all(e[k] == old[k] for k in e)
                    for e, old in zip(tape, records, strict=True)
                )
                with (
                    np.load(BytesIO(z.read(pre + "weights.npz"))) as w,
                    np.load(BytesIO(z.read(pre + "panel.npz"))) as v,
                ):
                    arrays[key + "_last_weights"] = w["policy_true"]
                    arrays[key + "_last_head"] = w["head_true"]
                    arrays[key + "_last_forecast"] = v["true"]
                    # Keep #21 learned and neutral views at the selected H values.
                    arrays[key + "_last_probs"] = v["policy"][[1, 3]][:, :, 1:, :, :]
                arrays[key + "_actual"] = np.asarray(
                    [v["policy"]["true"] for v in records], dtype=np.float32
                )
                hashes = [v["updates"]["true"]["before"] for v in records[:-1]] + [
                    records[-2]["updates"]["true"]["after"]
                ]
                arrays[key + "_sha"] = np.array(hashes, dtype="S64")
                limits[key] = turns(len(records) - 1)
        np.savez_compressed(dest / (tag + "-anchors.npz"), **arrays)
    print(json.dumps(limits))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("parent", "dest"):
        p.add_argument("--" + name, required=True, type=Path)
    prepare(**vars(p.parse_args()))
