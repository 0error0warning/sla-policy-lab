"""Verify the dose trials by recomputing both scores from physical, unweighted personal history."""

import argparse
from pathlib import Path

from experiments.direct_policy.protocol import write
from experiments.health_baseline.audit import audit_lives, audit_replay

# Explicit independent reference, separate from the training-rule objects.
SPECIFICATION = (("one", "local", 1.0), ("two", "local", 2.0))


def tapes(root, source):
    return audit_replay(root, source, SPECIFICATION, 652)


def lives(roots, birth):
    return audit_lives(roots, birth, SPECIFICATION)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("replay", "tapes", "source", "output"):
        p.add_argument("--" + k, type=Path, required=k == "output")
    p.add_argument("--lives", nargs="+", type=Path)
    a = p.parse_args()
    result = {}
    if a.replay:
        result["replay"] = tapes(a.replay, a.tapes)
    if a.lives:
        result["live"] = lives(a.lives, a.source)
    write(a.output, result)
