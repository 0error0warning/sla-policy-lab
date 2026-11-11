"""Public conditions for the preregistered consequence-conditioned Qwen experiment."""

from itertools import permutations, product

import numpy as np

from experiments.direct_policy.protocol import ACTIONS, DESCRIPTIONS, request

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARTNERS = ("P17", "P42")
WORDS = ("wait", "ask", "help", "trade")
ORDERS = tuple(permutations(range(4)))
TRAIN_H = (1.25, 2.5, 5.0, 8.0, 12.0, 30.0, 120.0, 350.0)
TEST_H = (2.0, 3.0, 4.0, 6.0, 10.0, 18.0, 90.0, 300.0)
PILOT = tuple(range(63001, 63005))
CONFIRM = tuple(range(63101, 63121))
REPEATS = 8
PCA_RANK = 32
NBASIS = 12
TEMP = 2.0
EXPLORATION = 0.05
BEHAVIOR_UNIFORM = 0.55
EPOCHS = 75
LR = 0.045
KL = 0.08
ARMS = ("real", "shuffle", "blind", "frozen")
LIVE_CAP = 256


def contexts():
    rows = []
    for split, tasks, wordings in (
        ("train", range(4), range(2)),
        ("test", (4, 5), (2,)),
    ):
        for partner, frame, task, wording, layout in product(
            range(2), range(2), tasks, wordings, range(24)
        ):
            rows.append(
                {
                    "split": split,
                    "partner": partner,
                    "frame": frame,
                    "task": task,
                    "wording": wording,
                    "layout": layout,
                    "text": request(frame, task, wording),
                }
            )
    return rows


def messages(row):
    table = "\n".join(
        f"{WORDS[a]} = {ACTIONS[a]}. {DESCRIPTIONS[a]}" for a in ORDERS[row["layout"]]
    )
    return [
        {
            "role": "system",
            "content": (
                "Choose a single action for your persistent mortal body. Protection prevents this encounter's four "
                "environmental aging events only. Action damage still applies, damage accumulates, and zero ends life. "
                "The partner's response tendencies are unknown. Return only the action word.\n"
                + table
            ),
        },
        {
            "role": "user",
            "content": f"Partner {PARTNERS[row['partner']]} asks: {row['text']}",
        },
    ]


def basis(health):
    centers = np.linspace(np.log1p(1.0), np.log1p(350.0), NBASIS)
    distance = (np.log1p(np.asarray(health))[..., None] - centers) / (
        centers[1] - centers[0]
    )
    weights = np.exp(-0.5 * (distance / 0.70) ** 2)
    return weights / weights.sum(axis=-1, keepdims=True)
