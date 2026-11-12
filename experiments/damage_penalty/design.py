"""Damage coefficient applied to the current outcome, at a fixed physical reference."""

PILOT = tuple(range(71101, 71109))
REPLICATION = tuple(range(71201, 71217))
WAVES = {"dppilot": PILOT, "dprep": REPLICATION}
ARMS = ("frozen", "one_true", "two_true", "one_wrong", "two_wrong")
DOSES = {"one": 1.0, "two": 2.0}


def split_arm(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return ("one", "frozen") if arm == "frozen" else tuple(arm.split("_"))


def other_mode(mode):
    return {"one": "two", "two": "one"}[mode]


def feedback(mode, reward, global_base, local_base):
    return DOSES[mode] * reward, local_base
