"""The sole training intervention is how each living individual centers its own reward."""

from dataclasses import dataclass, field

PILOT = tuple(range(70101, 70109))
REPLICATION = tuple(range(70201, 70217))
WAVES = {"hbpilot": PILOT, "hbrepa": REPLICATION[:8], "hbrepb": REPLICATION[8:]}
ARMS = ("frozen", "global_true", "local_true", "global_wrong", "local_wrong")
WINDOW = 1.25
MIN_PRIOR = 3


@dataclass
class Past:
    observations: list = field(default_factory=list)
    total: float = 0.0

    def baselines(self, h):
        global_b = (-8.0 + self.total) / (8 + len(self.observations))
        prior = [
            r for before, r in self.observations if h / WINDOW <= before <= WINDOW * h
        ]
        return (
            global_b,
            (sum(prior) / len(prior) if len(prior) >= MIN_PRIOR else global_b),
            len(prior),
        )

    def observe(self, h, reward):
        self.observations.append((h, reward))
        self.total += reward


def split_arm(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "frozen":
        return "global", "frozen"
    return tuple(arm.split("_"))


def choose(criterion, g, l):
    return {"global": g, "local": l}[criterion]
