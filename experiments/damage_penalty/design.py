"""A twofold penalty on the current RUL loss with both physical references identical."""

from experiments.health_baseline.design import FeedbackRule

PILOT = tuple(range(71101, 71109))
REPLICATION = tuple(range(71201, 71217))
WAVES = {"dppilot": PILOT, "dprep": REPLICATION}
ARMS = ("frozen", "one_true", "two_true", "one_wrong", "two_wrong")
RULES = (FeedbackRule("one", "local", 1.0), FeedbackRule("two", "local", 2.0))
REPLAY_ROLES = (
    "one_before",
    "one",
    "two_before",
    "two",
    "two_from_one",
    "one_duplicate",
)
