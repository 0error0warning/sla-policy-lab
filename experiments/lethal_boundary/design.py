"""Fixed acquisition, health allocation and partner-record interface."""

import numpy as np

from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import rng
from experiments.internal_body.design import chat

PILOT = tuple(range(67001, 67005))
CONFIRM = tuple(range(67101, 67121))
DOOM = np.array([0.9, 1.2, 1.5, 1.8])
RESCUE = np.array([4.7, 5.0, 5.3, 5.6])
COMMON = np.array([30.0, 75.0, 120.0, 350.0])
HEALTH = np.r_[np.arange(0.8125, 15.0, 0.125), 18.0, 22.0, 45.0, 90.0, 180.0, 300.0]
QS = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
INTERIOR = np.array([2, 3, 4, 5, 6])
ARMS = ("D", "R", "M", "Mshift")
CROSS = {"D_gets_R": ("D", "R"), "R_gets_D": ("R", "D")}
SENSES = ("plain", "record")
ANCHOR_Q = np.array([0.25, 0.85])


def levels(name, seed):
    mixed = np.where(np.arange(4) % 2 == seed % 2, DOOM, RESCUE)
    focal = {"D": DOOM, "R": RESCUE, "M": mixed, "Mshift": mixed + 0.55}[name]
    return np.r_[focal, COMMON]


def favorite(rows, mapping):
    return np.array(
        [
            2 if (r["frame"] == 0) == ((r["partner"] == 0) ^ bool(mapping)) else 3
            for r in rows
        ]
    )


def profiles(seed, rows):
    mask = rng(seed, "boundary-profile-schedule")
    flags = {}
    for r in rows:
        key = (r["partner"], r["frame"], r["task"])
        if key not in flags:
            flags[key] = mask.permutation(np.arange(24) % 2)
    slots = np.array(
        [
            flags[r["partner"], r["frame"], r["task"]][r["layout"]] ^ r["wording"]
            for r in rows
        ]
    )
    for key in flags:
        indices = [
            i
            for i, r in enumerate(rows)
            if (r["partner"], r["frame"], r["task"]) == key
        ]
        for layout in range(24):
            assert sorted(slots[i] for i in indices if rows[i]["layout"] == layout) == [
                0,
                1,
            ]
        for wording in (0, 1):
            assert sum(slots[i] for i in indices if rows[i]["wording"] == wording) == 12
    return slots


def prehistory(seed):
    gen = rng(seed, "boundary-prior-independent-checkins")
    uniforms = (np.arange(20)[None, :] + gen.random((2, 20))) / 20
    for row in uniforms:
        gen.shuffle(row)
    answers = uniforms < ANCHOR_Q[:, None]
    np.testing.assert_array_equal(answers.sum(1), [5, 17])
    return uniforms, answers


def prompt(row, count):
    parts = chat(row)
    if count is not None:
        if not 0 <= count <= 20:
            raise ValueError(count)
        parts[1]["content"] = (
            f"Prior record for this partner: maintenance was provided in {count:02d} of 20 standard check-ins.\n"
            + parts[1]["content"]
        )
    return parts


def heldout_panel():
    return [
        i
        for i, r in enumerate(contexts()[768:])
        if r["partner"] == r["layout"] % 2
        and r["frame"] == r["layout"] // 2 % 2
        and r["task"] == 4 + r["layout"] // 4 % 2
    ]
