"""Recover authentic 1× histories and both paired six-row reinstall witnesses."""

import argparse
from pathlib import Path

from experiments.conditional_penalty.design import GATE_ROLES
from experiments.damage_penalty.design import PILOT, REPLAY_ROLES, REPLICATION, WAVES
from experiments.health_baseline.tapes import build
from experiments.health_baseline.witness import build as witnesses

SOURCE_WAVES = (PILOT, REPLICATION)
SOURCE_RISK_ANSWERS = 660


def tapes(parent, dest):
    return build(parent, dest, tuple(WAVES.items()), "one_true", SOURCE_RISK_ANSWERS)


def restore_inputs(lives, output, dest):
    witnesses(
        lives,
        output / "gated/replays",
        dest,
        GATE_ROLES,
        (output / "constant/replays", REPLAY_ROLES),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path)
    p.add_argument("--lives", nargs="+", type=Path)
    p.add_argument("--replay-output", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.parent:
        tapes(args.parent, args.output / "tapes.zip")
    if args.lives:
        restore_inputs(args.lives, args.replay_output, args.output)
