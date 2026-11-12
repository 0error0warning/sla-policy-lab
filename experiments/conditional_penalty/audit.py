"""Verify dose timing, private histories, prefix identity and the original body arithmetic."""

import argparse
from pathlib import Path

import numpy as np

from experiments.conditional_penalty.inputs import SOURCE_RISK_ANSWERS
from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records
from experiments.health_baseline.analysis import read_replay
from experiments.health_baseline.audit import audit_lives, audit_replay


def reference_gate(pre_action_health):
    return 2.0 if 0 < pre_action_health <= 6.0 else 1.0


ONE = ("one", "local", 1.0)
TWO = ("two", "local", 2.0)
GATE = ("gate", "local", reference_gate)


def matched_histories(output, tapes):
    summaries = {}
    for name, rule in (("gated", GATE), ("constant", TWO)):
        summaries[name] = audit_replay(
            output / name / "replays", tapes, (ONE, rule), SOURCE_RISK_ANSWERS
        )
    stats = {
        "reference_turns": 0,
        "common_forks": 0,
        "identical_preswitch_steps": 0,
        "crossing_steps_started_above_six": 0,
        "checked_next_fork_distributions": 0,
        "low_updates": 0,
    }
    for f in sorted((output / "gated/replays").glob("s*/*/life.json")):
        a = read_replay(f.parent)
        b = read_replay(
            output / "constant/replays" / f.parent.parent.name / f.parent.name
        )
        for t, (g, c) in enumerate(zip(a, b, strict=True)):
            assert all(
                g[k] == c[k]
                for k in (
                    "turn",
                    "index",
                    "action",
                    "u",
                    "health",
                    "after",
                    "reward",
                    "local_baseline",
                    "eligible",
                )
            )
            assert (
                g["updates"][0] == c["updates"][0]
                and g["before_hash"][0] == c["before_hash"][0]
            )
            np.testing.assert_array_equal(
                np.array(g["probs"])[[0, 1]], np.array(c["probs"])[[0, 1]]
            )
            low = g["health"] <= 6
            compare = c["updates"][2 if low else 0]
            assert g["updates"][2] == compare
            assert g["signals"][1] == c["signals"][int(low)]
            if not low:
                assert (
                    g["updates"][0] == g["updates"][1]
                    and g["before_hash"][0] == g["before_hash"][1]
                )
                np.testing.assert_array_equal(g["probs"][1], g["probs"][3])
                stats["identical_preswitch_steps"] += 1
                stats["crossing_steps_started_above_six"] += int(g["after"] <= 6)
            else:
                stats["low_updates"] += 1
            if t:
                previous = a[t - 1]
                np.testing.assert_array_equal(
                    g["probs"][4], c["probs"][4 if previous["health"] <= 6 else 1]
                )
                stats["checked_next_fork_distributions"] += 1
            stats["common_forks"] += 1
            stats["reference_turns"] += 1
    return {**summaries, "three_way_checks": stats}


def live_prefixes(roots):
    summary = {
        "paired_lives": 0,
        "healthy_decisions": 0,
        "pre_switch_updates": 0,
        "first_low_actions_before_feedback": 0,
    }
    ecols = (
        "turn",
        "index",
        "health",
        "after",
        "action",
        "maintenance",
        "subevents",
        "noise",
        "u",
        "offset",
        "reward",
        "local_baseline",
    )
    for root in roots:
        for f in sorted(root.glob("s*/*/gate_*/life.json")):
            gate = f.parent
            one = gate.parent / gate.name.replace("gate_", "one_")
            left, right = records(gate), records(one)
            summary["paired_lives"] += 1
            entry = next(i for i, e in enumerate(left) if e["health"] <= 6)
            assert next(i for i, e in enumerate(right) if e["health"] <= 6) == entry
            for g, b in zip(left[: entry + 1], right[: entry + 1], strict=True):
                assert all(g[k] == b[k] for k in ecols)
                assert g["before_hash"][0] == b["before_hash"][0]
                np.testing.assert_array_equal(g["probs"][1], b["probs"][1])
                if gate.name == "gate_true":
                    assert g["before_hash"][0] == g["before_hash"][2]
                    np.testing.assert_array_equal(g["probs"][1], g["probs"][3])
                if g["health"] > 6:
                    assert (
                        g["after_hash"][0] == b["after_hash"][0]
                        and g["signal"] == b["signal"]
                    )
                    summary["pre_switch_updates"] += 1
                    summary["healthy_decisions"] += int(g["health"] >= 90)
                else:
                    assert (
                        g["update"][0]["damage_weight"] == 2.0
                        and b["update"][0]["damage_weight"] == 1.0
                    )
                    summary["first_low_actions_before_feedback"] += 1
            assert [v["action"] for v in left if v["health"] >= 90] == [
                v["action"] for v in right if v["health"] >= 90
            ]
    return summary


def autonomous(roots, birth):
    return {
        "physics": audit_lives(
            roots, birth, (ONE, TWO, GATE), {"gate_true": ("one", "two")}
        ),
        "paired_prefix": live_prefixes(roots),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("replay_output", "tapes", "source", "output"):
        p.add_argument("--" + k.replace("_", "-"), type=Path, required=k == "output")
    p.add_argument("--lives", type=Path, nargs="+")
    a = p.parse_args()
    if a.replay_output:
        write(a.output, matched_histories(a.replay_output, a.tapes))
    if a.lives:
        write(a.output, autonomous(a.lives, a.source))
