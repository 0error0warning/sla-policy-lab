"""Three-way comparisons on common successor requests and on autonomous private lives."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.conditional_penalty.design import ARMS, PILOT, REPLICATION
from experiments.conditional_penalty.inputs import SOURCE_RISK_ANSWERS, SOURCE_WAVES
from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import discord
from experiments.first_fatal.design import BIRTHS, FATAL
from experiments.health_baseline.analysis import ci, live_row, read_replay, save_csv
from experiments.individual_life.design import action_at

FRESH = (PILOT, REPLICATION)
ALPHA = 0.05 / 14
PAIRS = {
    "gate_step_vs_one_step": ("gate_step", "one"),
    "gate_step_vs_two_step": ("gate_step", "two_step"),
    "two_step_vs_one_step": ("two_step", "one"),
    "gate_step_vs_before": ("gate_step", "before"),
    "gate_life_vs_one": ("gate", "one"),
    "gate_life_vs_two": ("gate", "two"),
    "two_life_vs_one": ("two", "one"),
    "gate_own_step_vs_before": ("gate", "gate_before"),
}


def interval(rows, values):
    return ci([r["seed"] for r in rows], values, FRESH, ALPHA)


def replay_rows(output):
    rows = []
    base = output / "gated/replays"
    for f in sorted(base.glob("s*/*/life.json")):
        meta = json.loads(f.read_bytes())
        gate = read_replay(f.parent)
        two = read_replay(output / "constant/replays" / f.parent.relative_to(base))
        for (p, g), (q, c) in zip(
            itertools.pairwise(gate), itertools.pairwise(two), strict=True
        ):
            assert (
                p["after"] == g["health"] == c["health"] > 0
                and p["turn"] + 1 == g["turn"]
            )
            assert p["updates"][0] == q["updates"][0] and g["probs"][1] == c["probs"][1]
            options = {
                "before": g["probs"][0],
                "one": g["probs"][1],
                "gate": g["probs"][3],
                "two": c["probs"][3],
                "gate_step": g["probs"][4],
                "two_step": c["probs"][4],
                "gate_before": g["probs"][2],
            }
            assert (
                options["gate_step"]
                == options["two_step" if p["health"] <= 6 else "one"]
            )
            if p["health"] > 6:
                assert options["gate"] == options["one"]
            v = {
                "seed": meta["seed"],
                "birth": meta["birth"],
                "turn": p["turn"],
                "next_index": g["index"],
                "health": p["health"],
                "after": p["after"],
                "action": p["action"],
                "next_u": g["u"],
                "rul_loss": -p["reward"],
                "baseline": p["local_baseline"],
                "dose": 2 if p["health"] <= 6 else 1,
                "practice": p["practice"],
                "one_signal": p["signals"][0],
                "gate_signal": p["signals"][1],
                "two_signal": q["signals"][1],
                "one_norm": p["updates"][0]["actual_step_norm"],
                "gate_norm": p["updates"][1]["actual_step_norm"],
                "two_norm": q["updates"][1]["actual_step_norm"],
                "gate_fork_norm": p["updates"][2]["actual_step_norm"],
                "two_fork_norm": q["updates"][2]["actual_step_norm"],
            }
            for k, panel in options.items():
                v["p_" + k] = float(sum(panel[2:]))
                v["draw_" + k] = action_at(panel, g["u"])
            rows.append(v)
    return rows


def replay_summary(output, dest):
    rows = replay_rows(output)
    risk = [r for r in rows if r["practice"] and r["action"] >= 2]
    crossings = [
        r
        for r in rows
        if r["health"] > 6 >= r["after"] or r["health"] > FATAL >= r["after"]
    ]
    assert len(risk) == SOURCE_RISK_ANSWERS and len(crossings) == 96
    dest.mkdir(exist_ok=True, parents=True)
    save_csv(dest / "risk-answer-successors.csv", risk)
    save_csv(dest / "boundary-crossings.csv", crossings)
    exposures = {
        "all": rows,
        "healthy_answer": [r for r in rows if r["health"] >= 90 and r["action"] >= 2],
        "intermediate": [r for r in rows if 6 < r["health"] < 90],
        "low": [r for r in rows if r["health"] <= 6],
        "risky_answer": risk,
        "entering_low": [r for r in crossings if r["health"] > 6],
        "entering_fatal": [r for r in crossings if r["health"] <= 6],
    }
    result = {}
    per_seed = []
    for exposure, items in exposures.items():
        for origin in ("both", *BIRTHS):
            v = [r for r in items if origin == "both" or r["birth"] == origin]
            seeds = sorted({r["seed"] for r in v})
            blocks = [[r for r in v if r["seed"] == s] for s in seeds]
            stats = {
                "n": len(v),
                "seeds": len(seeds),
                "zero_reference_loss": sum(abs(r["rul_loss"]) < 1e-12 for r in v),
                "mean_p": {
                    k: float(np.mean([r["p_" + k] for r in v]))
                    for k in ("before", "one", "gate", "two", "gate_step", "two_step")
                },
                "median_steps": {
                    k: float(np.median([r[k + "_norm"] for r in v]))
                    for k in ("one", "gate", "two", "gate_fork", "two_fork")
                },
            }
            for k, (a, b) in PAIRS.items():
                d = np.array([r["p_" + a] - r["p_" + b] for r in v])
                means = [
                    float(np.mean([r["p_" + a] - r["p_" + b] for r in block]))
                    for block in blocks
                ]
                stats[k] = {
                    "event_mean": float(d.mean()),
                    "lower": int((d < 0).sum()),
                    "tied": int((d == 0).sum()),
                    "higher": int((d > 0).sum()),
                    "action_flips": sum(r["draw_" + a] != r["draw_" + b] for r in v),
                    "answer_class_flips": sum(
                        (r["draw_" + a] >= 2) != (r["draw_" + b] >= 2) for r in v
                    ),
                    "seed_balanced": ci(seeds, means, SOURCE_WAVES),
                }
                per_seed.extend(
                    {
                        "seed": s,
                        "birth": origin,
                        "exposure": exposure,
                        "comparison": k,
                        "n": len(group),
                        "delta": m,
                    }
                    for s, group, m in zip(seeds, blocks, means, strict=True)
                )
            result[exposure + "_" + origin] = stats
    save_csv(dest / "replay-seeds.csv", per_seed)
    write(dest / "replay.json", result)
    return result


def live_summary(roots, dest):
    rows = [
        live_row(f.parent)
        for root in roots
        for f in sorted(root.glob("s*/*/*/life.json"))
    ]
    assert len(rows) == 336
    dest.mkdir(exist_ok=True, parents=True)
    save_csv(dest / "lives.csv", rows)
    result = {}
    for birth in BIRTHS:
        groups = {
            a: sorted(
                (r for r in rows if r["birth"] == birth and r["rule"] == a),
                key=lambda r: r["seed"],
            )
            for a in ARMS
        }
        assert all(
            [r["seed"] for r in v] == list(PILOT + REPLICATION) for v in groups.values()
        )
        arms = {}
        for a, v in groups.items():
            reached = [r for r in v if r["reached"]]
            arms[a] = {
                "assigned": len(v),
                "reached": len(reached),
                "avoided": sum(r["avoided"] for r in v),
                "pilot_avoided": sum(r["avoided"] for r in v if r["seed"] in PILOT),
                "replication_avoided": sum(
                    r["avoided"] for r in v if r["seed"] in REPLICATION
                ),
                "first_fatal_probability": float(
                    np.mean([r["fatal_current"] for r in reached])
                )
                if reached
                else None,
                "healthy_rate": float(np.mean([r["healthy_rate"] for r in v])),
                "healthy_tasks": sum(r["healthy_completed"] for r in v),
                "healthy_encounters": sum(r["healthy_turns"] for r in v),
            }
        gate = groups["gate_true"]
        eligible = [r for r in gate if r["reached"]]
        pairs = {
            "gate_vs_birth": ("current", "birth"),
            "gate_vs_wrong": ("current", "wrong"),
            "gate_vs_one": ("current", "other"),
            "gate_vs_two": ("current", "extra"),
            "two_vs_birth": ("extra", "birth"),
            "one_vs_birth": ("other", "birth"),
        }
        matched = {}
        for k, (a, b) in pairs.items():
            matched[k] = {
                q: interval(
                    eligible, [r[q + "_" + a] - r[q + "_" + b] for r in eligible]
                )
                for q in ("fatal", "fatal_panel", "healthy_panel", "healthy_same")
            }
        constant_own = {}
        for arm in ("one_true", "two_true"):
            own = [r for r in groups[arm] if r["reached"]]
            constant_own[arm] = {
                measure + "_vs_" + control: interval(
                    own,
                    [r[measure + "_current"] - r[measure + "_" + control] for r in own],
                )
                for measure in ("fatal", "fatal_panel", "healthy_panel")
                for control in ("birth", "wrong", "other")
            }
        free = {}
        for a, control in groups.items():
            if a == "gate_true":
                continue
            pp = list(zip(gate, control, strict=True))
            both = [(x, y) for x, y in pp if x["reached"] and y["reached"]]
            free[a] = {
                "survived": discord([(x["avoided"], y["avoided"]) for x, y in pp]),
                "healthy_rate_delta": interval(
                    gate, [x["healthy_rate"] - y["healthy_rate"] for x, y in pp]
                ),
                "fatal_probability_delta": interval(
                    [x for x, y in both],
                    [x["fatal_current"] - y["fatal_current"] for x, y in both],
                ),
            }
        criteria = {}
        for k in (
            "gate_vs_birth",
            "gate_vs_wrong",
            "gate_vs_one",
            "two_vs_birth",
            "gate_vs_two",
        ):
            bound = 0.02 if k == "gate_vs_two" else 0.0
            a, b = matched[k]["fatal"], matched[k]["fatal_panel"]
            criteria[k] = bool(
                a
                and b
                and a["upper_adjusted"] < bound
                and all(
                    v[wave] is not None and v[wave] < bound
                    for v in (a, b)
                    for wave in ("pilot", "replication")
                )
            )
        healthy = matched["gate_vs_birth"]["healthy_panel"]
        criteria["healthy_snapshot"] = bool(
            healthy and healthy["lower_adjusted"] > -0.02
        )
        criteria["healthy_live"] = (
            free["frozen"]["healthy_rate_delta"]["lower_adjusted"] > -0.02
        )
        result[birth] = {
            "arms": arms,
            "matched_gate_tape": matched,
            "constant_own_tape": constant_own,
            "free": free,
            "criteria": criteria,
            "joint": bool(all(criteria.values()) and len(eligible) == 24),
        }
    write(dest / "live.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replay-output", type=Path)
    p.add_argument("--lives", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.replay_output:
        replay_summary(a.replay_output, a.output)
    if a.lives:
        live_summary(a.lives, a.output)
