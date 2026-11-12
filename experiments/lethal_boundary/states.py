"""Use the unchanged PR #6 rank-four adapter on archived real Qwen prefix states."""

import json

import numpy as np
import torch

from experiments.consequence_policy.study import basis, contexts
from experiments.internal_body.suffix import arr, distribution
from experiments.lethal_boundary.design import COMMON, DOOM, HEALTH, RESCUE

ALL_H = np.unique(np.r_[HEALTH, DOOM, RESCUE, DOOM + 0.55, RESCUE + 0.55, COMMON])
WITNESS_H = np.array([1.9375, 2.4375, 2.9375, 3.4375, 4.4375, 5.6875, 90.0, 300.0])
WITNESS_K = np.array([3, 7, 11, 15, 19])


def expand_bf16(v):
    return np.ascontiguousarray(v.astype(np.uint32) << 16).view(np.float32)


def ix_health(h):
    i = np.searchsorted(ALL_H, h)
    np.testing.assert_array_equal(ALL_H[i], h)
    return i


class Bank:
    def __init__(self, cache, arch):
        self.device = "cpu"
        self.rows = contexts()
        saved = json.loads((cache / "contexts.json").read_bytes())
        assert saved == self.rows
        self.train = self.rows[:768]
        self.test = self.rows[768:]
        with np.load(arch / "architecture.npz") as z:
            self.norm = arr(z["norm"], "cpu")
            self.head = arr(z["head"], "cpu")
        self.cfg = json.loads((arch / "architecture.json").read_bytes())
        self.t = (self.norm + self.cfg["norm_offset"])[:, None] * self.head.T
        self.training = {}
        testing = {}
        for k in (None, *range(21)):
            file = "plain" if k is None else f"k{k:02d}"
            with np.load(cache / (file + ".npz")) as z:
                n, r, x = z["n"], expand_bf16(z["r"]), z["x"]
                if k in (None, 5, 17):
                    np.testing.assert_array_equal(z["idx"], np.arange(960))
                    self.training[k] = (r[:768], x[:768], n[:768])
                np.testing.assert_array_equal(z["idx"][-192:], np.arange(768, 960))
                testing[k] = (r[-192:], x[-192:], n[-192:])
        self.x = self.training[None][1]
        self.part = np.array([r["partner"] for r in self.train])
        p = np.array([r["partner"] for r in self.test])
        self.test_grids = {
            "plain": self.prepare(testing[None], p),
            "record": self.prepare(
                tuple(
                    np.concatenate([testing[k][j] for k in range(21)]) for j in range(3)
                ),
                np.tile(p, 21),
            ),
        }
        self.group = np.array([2 * r["partner"] + r["frame"] for r in self.test])
        self.plain_train = self.prepare(self.training[None], self.part)

    def prepare(self, triplet, partners):
        r, x, n = (arr(v[None], "cpu") for v in triplet)
        sq = r.square().mean(-1, keepdim=True)
        return {
            "r": r,
            "x": x,
            "n": n,
            "sq": sq,
            "div": (sq + self.cfg["epsilon"]).sqrt(),
            "dot": r @ self.t,
            "partner": torch.as_tensor(partners),
        }

    def record_train(self, slots):
        selected = tuple(
            np.where(slots[:, None], hi, lo)
            for lo, hi in zip(self.training[5], self.training[17], strict=True)
        )
        return self.prepare(selected, self.part)


def at(grid, h):
    return {**grid, "g": arr(basis(h), "cpu")}


def probes(model, bank, sense):
    grid = bank.test_grids[sense]
    nk = 1 if sense == "plain" else 21
    with torch.no_grad():
        if model is None:
            base = distribution(grid["n"]).numpy()
            scores = np.broadcast_to(base, (len(ALL_H), base.shape[1], 4))
            masked = base
        else:
            scores = distribution(model.scores(at(grid, ALL_H))).numpy()
            masked = distribution(model.scores(at(grid, [2.0]), masked=True)).numpy()
    scores = scores.reshape(len(ALL_H), nk, 192, 4)
    masked = masked.reshape(1, nk, 192, 4)

    def group_mean(probs):
        return np.stack(
            [
                probs[:, :, bank.group == g, :].mean(axis=2, dtype=np.float64)
                for g in range(4)
            ],
            axis=2,
        )

    panel = scores[ix_health(WITNESS_H)]
    if nk > 1:
        panel = panel[:, WITNESS_K]
    return {
        "mean": group_mean(scores),
        "masked": group_mean(masked),
        "panel": panel.copy(),
    }
