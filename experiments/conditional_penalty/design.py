"""Conditional coefficients selected before the physical encounter."""

from experiments.damage_penalty.design import RULES as CONSTANTS
from experiments.first_fatal.design import PRACTICE_TOP
from experiments.health_baseline.design import FeedbackRule

PILOT = tuple(range(72101, 72109))
REPLICATION = tuple(range(72201, 72217))
WAVES = {"gppilot": PILOT, "gprep": REPLICATION}
ARMS = (
    "frozen",
    "one_true",
    "gate_true",
    "two_true",
    "one_wrong",
    "gate_wrong",
    "two_wrong",
)
GATE_OFF = FeedbackRule("gate", "local", 1.0)
GATE_ON = FeedbackRule("gate", "local", 2.0)


class LowHealthPenalty:
    name = "gate"

    @staticmethod
    def at_health(health):
        if not 0 < health <= 350:
            raise ValueError("The damage coefficient requires living pre-action health")
        return GATE_ON if health <= PRACTICE_TOP else GATE_OFF


GATE = LowHealthPenalty()
ONE, TWO = CONSTANTS
GATE_PAIR = (ONE, GATE)
GATE_ROLES = (
    "one_before",
    "one",
    "gate_before",
    "gate",
    "gate_from_one",
    "one_duplicate",
)
# On the gated correct tape, slot three is always 1× and slot four always 2×.
LIVE_OVERRIDES = {"gate_true": ((GATE, ONE), TWO), "gate_wrong": ((GATE, ONE), None)}
