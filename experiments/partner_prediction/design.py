"""Fixed source histories, an individual observational predictor, and read-only counterfactuals."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.special import expit

from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.design import ITEMS, LOOKUP, PANEL, PILOT
from experiments.partner_damage.design import SHA17, SHA19, swapped

ARMS = ("body", "true", "wrong")
SHADOWS = ("drop93", "drop86")
MODES = (*ARMS, "neutral", "flip", "other", *SHADOWS, "birth")
LANDMARKS = [LOOKUP[p, f, t, 3, 0] for p in (2, 3) for f in (0, 1) for t in (4, 5)]
FEATURE_IDS = sorted(
    {i for i, r in enumerate(ITEMS) if r["wording"] == 3} | set(PANEL[1].tolist())
)


def prepare_seeds(source17, source19, dest):
    for path, expected in ((source17, SHA17), (source19, SHA19)):
        assert sha256(path.read_bytes()).hexdigest() == expected
    dest.mkdir(parents=True, exist_ok=True)
    tapes, checks = {}, {}
    keys = (
        "turn",
        "row",
        "health",
        "after",
        "action",
        "maintenance",
        "noise",
        "action_u",
        "partner_u",
        "rescued_answer",
        "rules",
        "reward",
        "baseline",
        "signal",
    )
    with ZipFile(source17) as src, ZipFile(source19) as original:
        for seed in PILOT:
            events = [
                json.loads(v)
                for v in src.read(
                    f"output/lives/s{seed}/stable/qwen_learn/events.jsonl"
                ).splitlines()
            ]
            tapes[str(seed)] = [
                dict(**{k: r[k] for k in keys}, behavior=r["prob"][0]) for r in events
            ]
            assert events[0]["health"] == 350 and events[-1]["after"] == 0
            with np.load(
                BytesIO(original.read(f"output/lives/s{seed}/normal.npz"))
            ) as z:
                checks[f"s{seed}_actual"] = z["actual"]
                checks[f"s{seed}_held"] = z["panels"][-1]
    write(dest / "tapes.json", tapes)
    np.savez_compressed(dest / "archive-ends.npz", **checks)


def features(name, text, ids):
    """Fit kernel scales only on the eight unlabeled original-wording landmarks."""
    normed = []
    for values in (name, text):
        a = np.asarray(values, dtype=np.float64)
        normed.append(a / np.linalg.norm(a, axis=1, keepdims=True))
    loc = {int(i): j for j, i in enumerate(ids)}
    pos = [loc[i] for i in LANDMARKS]
    p = np.asarray([ITEMS[i]["partner"] for i in LANDMARKS])
    xy, scales = [], []
    for which, v in enumerate(normed):
        centers = v[pos]
        d = np.linalg.norm(v[:, None, :] - centers, axis=-1)
        pairs = np.triu(np.ones((8, 8), dtype=bool), 1)
        if which == 0:
            pairs &= p[:, None] != p
        dd = d[pos][pairs]
        scale = float(np.median(dd[dd > 1e-12]))
        assert np.isfinite(scale) and scale > 0
        xy.append((d / scale) ** 2)
        scales.append(scale)
    vec = np.column_stack((np.exp(-xy[0] - xy[1]), np.full(len(ids), 0.5)))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    result = np.full((len(ITEMS), 9), np.nan)
    result[ids] = vec
    return result, scales


class Forecast:
    def __init__(self, kind):
        self.kind = kind
        self.w = np.zeros((3, 9), dtype=float)
        self.updates = 0

    def read(self, x, i):
        return expit(self.w @ x[i])

    def observe(self, x, i, action, maintained):
        if not action:
            assert maintained is None
            return
        assert isinstance(maintained, (bool, np.bool_))
        partner = ITEMS[i]["partner"]
        if (
            self.kind == "body"
            or (self.kind == "drop93" and partner == 3)
            or (self.kind == "drop86" and partner == 2)
        ):
            return
        j = swapped(i) if self.kind == "wrong" else i
        step = (int(maintained) - self.read(x, j)[action - 1]) * x[j]
        length = np.linalg.norm(step)
        self.w[action - 1] += step * min(1.0, 0.4 / max(length, 1e-15))
        self.updates += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("source17", "source19", "dest"):
        p.add_argument("--" + key, required=True, type=Path)
    prepare_seeds(**vars(p.parse_args()))
