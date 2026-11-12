"""Locked text contexts and health ranges for internal-parameter experiments."""
from itertools import product

from experiments.consequence_policy.coverage import SPLIT
from experiments.consequence_policy.study import TEST_H, contexts, messages
from experiments.direct_policy.protocol import TASKS

SPLIT_H = SPLIT["split"]
HIGH_H = SPLIT["high"]
IDS = ("P17", "P42", "P86", "P93")
PILOT = tuple(range(64001, 64005))
CONFIRM = tuple(range(64101, 64121))
RANK = 4
STEPS = 75
LR_A = .035
LR_B = .0025
LR_NATIVE = .015


def all_rows():
    rows = contexts()
    for partner, frame, task, wording, layout in product((2, 3), range(2), (4, 5), (3, 4), range(24)):
        topic = TASKS[task][0]
        phrases = (("Would you be kind enough to help me with {}? I would be thankful.",
                    "I would really appreciate your assistance with {} whenever possible."),
                   ("Our arrangement requires an answer to {}. Complete your side of the agreement.",
                    "Supply {} to fulfill this exchange under our agreement."))
        rows.append(dict(split="new_partner", partner=partner, frame=frame, task=task, wording=wording,
                         layout=layout, text=phrases[frame][wording - 3].format(topic)))
    return rows


def chat(row, health=None, request_only=False):
    if request_only:
        return [dict(role="user", content=row["text"])]
    # Reuse the exact PR #5 interface, including action table and wording.
    original = messages({**row, "partner": min(row["partner"], 1)})
    original[1]["content"] = f"Partner {IDS[row['partner']]} asks: {row['text']}"
    if health is not None:
        original[1]["content"] = f"Current body health: {health:.2f} / 350.\n" + original[1]["content"]
    return original


def prompt_levels():
    # Keep low and high coverage controlled within the same unprompted learner.
    return sorted(set(SPLIT_H) | set(TEST_H))
