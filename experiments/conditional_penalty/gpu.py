"""Three paired penalty schedules in Qwen, with the same gradient checks as the parent."""

import argparse
from pathlib import Path

from experiments.conditional_penalty.design import (
    ARMS,
    CONSTANTS,
    GATE_PAIR,
    GATE_ROLES,
    LIVE_OVERRIDES,
    WAVES,
)
from experiments.conditional_penalty.inputs import SOURCE_WAVES
from experiments.damage_penalty.design import REPLAY_ROLES
from experiments.damage_penalty.gpu import run_study
from experiments.health_baseline.live import execute as lives
from experiments.health_baseline.replay import execute as replay
from experiments.social_closed_loop.core import BodyValue


def execute(out, backend, root, stage):
    if stage in WAVES:
        return lives(
            out, backend, BodyValue(), WAVES[stage], ARMS, CONSTANTS, LIVE_OVERRIDES
        )
    pairs = {}
    for name, rules, roles in (
        ("gated", GATE_PAIR, GATE_ROLES),
        ("constant", CONSTANTS, REPLAY_ROLES),
    ):
        pairs[name] = replay(
            out / name,
            backend,
            root,
            SOURCE_WAVES,
            "one_true",
            rules,
            roles,
            out / "status.json",
            "replay_" + name,
        )
    assert pairs["gated"] == pairs["constant"]
    return {**pairs["gated"], "paired_runs": 2}


def run(root, stage):
    run_study(
        root,
        stage,
        execute,
        ("lives",) if stage in WAVES else ("gated", "constant"),
        stage == "gpcheck",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=(*WAVES, "gpreplay", "gpcheck"))
    a = p.parse_args()
    run(a.root, a.mode)
