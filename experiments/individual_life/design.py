"""Private-birth definitions, paired exogenous arrivals, and actual observation credit."""

from hashlib import sha256

import numpy as np

from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import rng
from experiments.internal_body.design import SPLIT_H, all_rows

PILOT = (68011, 68012, 68013, 68014)
CONFIRM = tuple(range(68101, 68113))
BIRTHS = ("blank", "population")
VISIBLE = ("frozen", "true1", "wrong1", "true8", "shuffle8")
HIDDEN = ("frozen", "true1", "wrong1")
RATE = 0.18
RIDGE = 0.01
CAP = 0.40
NEUTRAL = basis(SPLIT_H).mean(0).astype(np.float32)
CHECKS = (8, 16, 32, 48)
PROBE_H = (2.0, 6.0, 90.0, 300.0)
PARAMS = 13408


def rows():
    return [r for r in all_rows() if r["split"] == "new_partner"]


def starting(archive, birth):
    with np.load(archive) as z:
        A = z["A"].copy()
        B = z["B"].copy()
    if birth == "blank":
        A = np.random.default_rng(64800).normal(0, 0.06, A.shape).astype(np.float32)
        B.fill(0)
    elif birth != "population":
        raise ValueError(birth)
    return np.broadcast_to(A.mean(axis=0, keepdims=True), A.shape).copy(), B


def digest(a, b):
    h = sha256()
    h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
    h.update(np.ascontiguousarray(b, dtype="<f4").tobytes())
    return h.hexdigest()


def world(seed):
    n = (seed - (68011 if seed in PILOT else 68101)) % 4
    return (n // 2, n % 2)


def offers(row, bits):
    partner = row["partner"] - 2
    favor = 2 + (partner ^ row["frame"] ^ bits[partner])
    p = np.array([0.0, 0.15, 0.1, 0.1])
    p[favor] = 0.9
    return p, favor


def tape(seed, items):
    from scipy.special import ndtr, ndtri

    # This is an upper bound from the body rule, not an administrative time limit.
    limit = 2008
    gen = rng(seed, "private-single-life-environment-v1")
    layouts = np.concatenate([gen.permutation(24) for _ in range((limit + 23) // 24)])[
        :limit
    ]
    cells = np.concatenate(
        [gen.permutation(np.tile(np.arange(4), 2)) for _ in range(limit // 8)]
    )
    lookup = {
        (r["partner"] - 2, r["frame"], r["layout"]): j
        for j, r in enumerate(items)
        if (r["task"], r["wording"]) == (4, 3)
    }
    ix = np.asarray(
        [lookup[c // 2, c % 2, l] for c, l in zip(cells, layouts, strict=True)]
    )
    uniforms = gen.random((limit, 2))
    noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * gen.random((limit, 5)))
    offsets = rng(seed, "private-miscredit-offset").integers(1, 4, size=limit)
    return ix, uniforms, noise, offsets


def permutation(seed, first, n):
    if n < 2:
        return np.arange(n)
    g = rng(seed, "private-within-block-mismatch", first, n)
    order = g.permutation(n)
    result = np.empty(n, dtype=int)
    result[order] = np.roll(order, 1)
    return result


def action_at(prob, u):
    return int(np.sum(u > np.cumsum(np.asarray(prob, dtype=float))[:3]))
