"""Human-readable actual life, seed effects and within-life diagnostic contrasts."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import describe, records


def emit(path, data):
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def individual(events):
    fields = []
    for i, e in enumerate(events):
        info = {
            "turn": e["turn"],
            "partner": e["partner"],
            "frame": e["frame"],
            "layout": e["layout"],
            "health_at_choice": e["health_before"],
            "actual_action": e["word"],
            "maintenance": {
                None: "no partner response",
                False: "declined",
                True: "maintained",
            }[e["maintenance"]],
            "health_settled": e["health_after"],
            "feedback": e["reference_change"],
            "personal_signal": e["centered_signal"],
            "loRA_version_used": e["before_learning_version"],
            "steps_released_afterward": e["updates_after_settlement"],
            "terminal": e["terminal"],
            "preceding_experience_turn": events[i - 1]["turn"] if i else "",
            "previous_snapshot_choice": ("wait", "ask", "help", "trade")[
                e["counterfactual_previous_choice"]
            ]
            if i
            else "",
            "same_draw_as_previous_snapshot_flips": int(
                e["counterfactual_previous_choice"] != e["action"]
            )
            if i
            else "",
        }
        for label, name in (
            ("before_prior_update", "prob_previous"),
            ("deployed", "prob_current"),
            ("own_birth", "prob_birth"),
            ("wrong_shadow", "prob_wrong_shadow"),
        ):
            for a, p in zip(("wait", "ask", "help", "trade"), e[name], strict=True):
                info[label + "_" + a] = p
        fields.append(info)
    return fields


def feedback_links(events):
    counts = {}
    for previous, now in itertools.pairwise(events):
        response = (
            "silent"
            if previous["maintenance"] is None
            else "maintained"
            if previous["maintenance"]
            else "declined"
        )
        cell = (
            "same_partner" if now["partner"] == previous["partner"] else "other_partner"
        )
        cell += "_same_frame" if now["frame"] == previous["frame"] else "_other_frame"
        change = (np.array(now["prob_current"]) - now["prob_previous"])[
            previous["action"]
        ]
        counts.setdefault(response + "/" + cell, []).append(change)
    return {
        k: {"n": len(a), "mean_change_on_preceding_action": float(np.mean(a))}
        for k, a in counts.items()
    }


def run(root, analysis, out):
    out.mkdir(parents=True, exist_ok=True)
    lives = json.loads((analysis / "lives.json").read_bytes())
    seed_rows = []
    links = {}
    for tag, entry in sorted(lives.items()):
        sense, seed, birth, rule = tag.split("/")
        m = entry["metrics"]
        a = {"sensor": sense, "seed": int(seed[1:]), "birth": birth, "rule": rule}
        a.update(
            {
                k: m.get(k, "")
                for k in (
                    "turns",
                    "next_living_decisions",
                    "main_steps",
                    "terminal_steps",
                    "tasks",
                    "safe_tasks",
                    "received_maintenance",
                    "actual_doomed_encounters",
                    "actual_doomed_answers",
                    "actual_rescuable_encounters",
                    "G",
                    "G_raw",
                    "C",
                    "C_raw",
                    "L",
                    "L_raw",
                    "TV",
                    "changed_next_choices",
                    "vs_birth_flip",
                    "birth_death_change",
                    "birth_safe_task_change",
                    "birth_preferred_style_change",
                )
            }
        )
        seed_rows.append(a)
        if rule != "true1":
            continue
        events = records(root / tag / "life.jsonl")
        links[tag] = feedback_links(events)
        if seed in ("s68011", "s68101") and sense == "visible":
            emit(out / ("life-" + seed + "-" + birth + ".csv"), individual(events))
    emit(out / "per-seed.csv", seed_rows)
    summary = {}
    for tag, vals in links.items():
        sense, seed, birth, rule = tag.split("/")
        arm = summary.setdefault(sense + "/" + birth, {})
        for key, v in vals.items():
            arm.setdefault(key, []).append(v)
    result = {
        arm: {
            k: {
                "individuals_with_observations": len(v),
                "transitions": sum(t["n"] for t in v),
                "previous_action_probability_change": describe(
                    [t["mean_change_on_preceding_action"] for t in v]
                ),
            }
            for k, v in vals.items()
        }
        for arm, vals in summary.items()
    }
    write(out / "feedback.json", {"seeds": links, "descriptive_conditionals": result})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "analysis", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.analysis, a.output)
