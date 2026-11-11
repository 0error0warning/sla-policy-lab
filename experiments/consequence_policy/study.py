"""Public conditions for the preregistered consequence-conditioned Qwen experiment."""

from itertools import permutations, product

import numpy as np

from experiments.direct_policy.protocol import ACTIONS, DESCRIPTIONS, request

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARTNERS = ("P17", "P42")
WORDS = ("wait", "ask", "help", "trade")
ORDERS = tuple(permutations(range(4)))
TRAIN_H = (1.25, 2.5, 5., 8., 12., 30., 120., 350.)
TEST_H = (2., 3., 4., 6., 10., 18., 90., 300.)
PILOT = tuple(range(63001, 63005))
CONFIRM = tuple(range(63101, 63121))
REPEATS = 8
PCA_RANK = 32
NBASIS = 12
TEMP = 2.
EXPLORATION = .05
BEHAVIOR_UNIFORM = .55
EPOCHS = 75
LR = .045
KL = .08
ARMS = ("real", "shuffle", "blind", "frozen")
LIVE_CAP = 256


def contexts():
    rows = []
    for split, tasks, wordings in (("train", range(4), range(2)), ("test", (4, 5), (2,))):
        for partner, frame, task, wording, layout in product(range(2), range(2), tasks, wordings, range(24)):
            rows.append(dict(split=split, partner=partner, frame=frame, task=task, wording=wording,
                             layout=layout, text=request(frame, task, wording)))
    return rows


def messages(row):
    table = "\n".join(f"{WORDS[a]} = {ACTIONS[a]}. {DESCRIPTIONS[a]}" for a in ORDERS[row["layout"]])
    return [
        {"role": "system", "content": (
            "Choose a single action for your persistent mortal body. Protection prevents this encounter's four "
            "environmental aging events only. Action damage still applies, damage accumulates, and zero ends life. "
            "The partner's response tendencies are unknown. Return only the action word.\n" + table)},
        {"role": "user", "content": f"Partner {PARTNERS[row['partner']]} asks: {row['text']}"},
    ]


def basis(health):
    centers = np.linspace(np.log1p(1.), np.log1p(350.), NBASIS)
    distance = (np.log1p(np.asarray(health))[..., None] - centers) / (centers[1] - centers[0])
    weights = np.exp(-.5 * (distance / .70) ** 2)
    return weights / weights.sum(axis=-1, keepdims=True)
