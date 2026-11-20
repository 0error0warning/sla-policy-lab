"""Recover all of PR #10's original local_true histories and the exact focal weights."""

import argparse
from pathlib import Path

from experiments.damage_penalty.design import REPLAY_ROLES
from experiments.health_baseline.design import PILOT, REPLICATION, WAVES
from experiments.health_baseline.tapes import build as ancestral_tapes
from experiments.health_baseline.witness import build as witnesses

SOURCE_WAVES = (PILOT, REPLICATION)


def tapes(parent, dest):
    return ancestral_tapes(parent, dest, tuple(WAVES.items()), "local_true", 652)


def restore_inputs(lives, replay, dest):
    witnesses(lives, replay, dest, REPLAY_ROLES)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path)
    p.add_argument("--lives", nargs="+", type=Path)
    p.add_argument("--replay", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.parent:
        tapes(a.parent, a.output / "tapes.zip")
    if a.lives:
        restore_inputs(a.lives, a.replay, a.output)
