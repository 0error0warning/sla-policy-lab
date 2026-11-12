"""Replace one eighth of the paired tape with two stochastic protected-survival healths."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import rng, write
from experiments.internal_body.suffix import arr, fit
from experiments.lethal_boundary.design import (
    ANCHOR_Q,
    HEALTH,
    levels,
    prehistory,
    profiles,
)
from experiments.lethal_boundary.jitter import LocalHealthPolicy, signals
from experiments.lethal_boundary.physics import action_responses, cutoff, experience
from experiments.lethal_boundary.states import ALL_H, Bank, probes
from experiments.social_closed_loop.core import BodyValue

SEEDS = tuple(range(67301, 67313))
BORDER = cutoff(
    1.75, 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * np.array([0.25, 0.75]))
)
AXIS = np.unique(np.r_[ALL_H, BORDER])
CELLS = {
    "fixed": ("fixed", "fixed"),
    "border": ("border", "border"),
    "border_sensor_fixed": ("border", "fixed"),
    "border_shuffled": ("border", "shuffled"),
}


def assignment(seed, rows, slot):
    fixed = np.broadcast_to(levels("M", seed)[:, None], (8, len(rows))).copy()
    changed = fixed.copy()
    gen = rng(seed, "transition-state-assignment")
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(
            (r["partner"], r["frame"], r["task"], int(slot[i])), []
        ).append(i)
    for group in groups.values():
        assert len(group) == 24
        indices = gen.permutation(group)
        for j, sub in enumerate(indices.reshape(4, 6)):
            changed[j, sub[:3]] = BORDER[0]
            changed[j, sub[3:]] = BORDER[1]
    mask = fixed != changed
    np.testing.assert_array_equal(mask.sum(axis=0), np.ones(len(rows)))
    assert all(np.sum(changed == h) == 384 for h in BORDER)
    assert not np.intersect1d(HEALTH, changed).size
    return fixed, changed, mask


def permute_continuation(seed, m, rows, slot, h, a, resp, after, mask, body):
    before = body.interpolate(h, body.reference)
    scale = np.minimum(1.0, before)
    normalized = body.interpolate(after, body.reference) / scale[:, :, None]
    groups = np.array(
        [
            r["partner"] * 4 + r["frame"] * 2 + int(s)
            for r, s in zip(rows, slot, strict=True)
        ]
    )
    strata = (groups[None, :, None] * 4 + a) * 3 + (resp + 1)
    selected = np.broadcast_to(mask[:, :, None], a.shape)
    perm = np.arange(a.size).reshape(a.shape)
    gen = rng(seed, "matched-border-continuation", m)
    for label in np.unique(strata[selected]):
        loc = np.flatnonzero(selected & (strata == label))
        perm.flat[loc] = gen.permutation(loc)
        np.testing.assert_array_equal(
            np.sort(normalized.ravel()[loc]),
            np.sort(normalized.ravel()[perm.ravel()[loc]]),
        )
    shuffled = normalized.ravel()[perm]
    reward = body.interpolate(after, body.reference) - before[:, :, None]
    return np.where(
        selected, shuffled * scale[:, :, None] - before[:, :, None], reward
    ), perm.astype(np.uint16)


def run_one(seed, bank, body, root):
    d = root / f"s{seed}"
    d.mkdir(exist_ok=True, parents=True)
    a, mu, u, noise = draw_tape(seed, bank.training[None][2])
    slot = profiles(seed, bank.train)
    fixed, border, mask = assignment(seed, bank.train, slot)
    hs = {"fixed": fixed, "border": border}
    iu, ic = prehistory(seed)
    grid = bank.record_train(slot)
    np.savez_compressed(
        d / "tape.npz",
        action=a,
        mu=mu,
        uniform=u,
        noise=noise,
        profile=slot,
        prior_uniform=iu,
        prior_outcome=ic,
        **hs,
    )
    for m in (0, 1):
        folder = d / f"m{m}"
        folder.mkdir(exist_ok=True)
        resp = action_responses(bank.train, m, ANCHOR_Q[slot], a, u)
        raw = {"response": resp}
        targets = {}
        hist = {}
        for k, h in hs.items():
            y, r = experience(h, a, resp, noise, body)
            raw[k + "_after"] = y
            raw[k + "_reward"] = r
            targets[k] = signals(a, r, mu, h, body)
        rr, perm = permute_continuation(
            seed, m, bank.train, slot, border, a, resp, raw["border_after"], mask, body
        )
        raw["permutation"] = perm
        targets["shuffled"] = signals(a, rr, mu, border, body)
        np.savez_compressed(folder / "outcomes.npz", **raw)
        seen = []
        for h in BORDER:
            take = (border[:, :, None] == h) & (a >= 2) & (resp == 1)
            seen.append(
                {
                    "health": float(h),
                    "maintained_answers": int(take.sum()),
                    "survived": int((raw["border_after"][take] > 0).sum()),
                }
            )
        write(folder / "observed-transition.json", seen)
        for name, (sensor, credit) in CELLS.items():
            model = LocalHealthPolicy(bank, "linear")
            hist[name] = fit(
                model, {**grid, "g": arr(basis(hs[sensor]), "cpu")}, targets[credit]
            )
            model.save(folder / (name + ".npz"))
            np.savez_compressed(
                folder / (name + "-probe.npz"), **probes(model, bank, "record", AXIS)
            )
        write(folder / "steps.json", hist)
    write(
        d / "complete.json",
        {
            "seed": seed,
            "budget": a.size,
            "replaced": int(mask.sum() * a.shape[-1]),
            "new_anchor_healths": BORDER.tolist(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("arch", "cache", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    bank = Bank(a.cache, a.arch)
    body = BodyValue()
    for s in SEEDS:
        if (a.output / f"s{s}/complete.json").exists():
            continue
        run_one(s, bank, body, a.output)
        print("completed", s, flush=True)
