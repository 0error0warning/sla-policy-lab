"""Readable whole lives and a figure for the prespecified first-fatal contrasts."""

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.first_fatal.analysis import records
from experiments.first_fatal.design import BIRTHS, FATAL, PILOT, PRACTICE_TOP
from experiments.individual_life.design import action_at

WORDS = ("wait", "ask", "help", "trade")


def show_lives(root, out):
    out.mkdir(exist_ok=True, parents=True)
    for b in BIRTHS:
        es = records(root / f"s{PILOT[0]}" / b / "true1")
        with (out / f"s{PILOT[0]}-{b}.csv").open(
            "w", encoding="utf8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "turn",
                    "partner",
                    "frame",
                    "layout",
                    "health_before",
                    "health_after",
                    "practice",
                    "first_fatal",
                    "action",
                    "maintenance",
                    "birth_choice_same_draw",
                    "wrong_choice_same_draw",
                    "near_only_wrong_choice_same_draw",
                    "p_answer_current",
                    "p_answer_birth",
                    "p_answer_wrong",
                    "p_answer_near_wrong",
                    "p_answer_entry",
                    "reward",
                    "personal_baseline",
                    "centered_signal",
                    "prior_updates",
                ]
            )
            for e in es:
                p = e["probs"]
                u = e["u"][0]
                writer.writerow(
                    [
                        e["turn"],
                        ("P86", "P93")[e["partner"] - 2],
                        e["frame"],
                        e["layout"],
                        e["health"],
                        e["after"],
                        e["practice"],
                        e["first_fatal"],
                        e["word"],
                        {None: "no response", True: "maintained", False: "declined"}[
                            e["maintenance"]
                        ],
                        *[WORDS[action_at(p[i], u)] for i in (0, 2, 3)],
                        *[float(sum(p[i][2:])) for i in (1, 0, 2, 3, 4)],
                        e["reward"],
                        e["baseline"],
                        e["signal"],
                        e["version_at_choice"],
                    ]
                )


def figure(root, data, out):
    from matplotlib.figure import Figure

    file = root / f"s{PILOT[0]}" / "blank" / "true1"
    e = records(file)
    first = next(x for x in e if x["first_fatal"])
    fig = Figure(figsize=(10.5, 4.5), layout="constrained")
    a, b = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.14]})
    t = np.arange(1, len(e) + 1)
    h = np.array([x["health"] for x in e])
    a.plot(t, h, lw=1.8, color="#205b77", label="Living body")
    a.axhspan(FATAL, PRACTICE_TOP, color="#d3eadf", label="Survivable practice")
    a.scatter(
        [first["turn"]],
        [first["health"]],
        s=62,
        color="#ab442b",
        zorder=3,
        label="First fatal-answer request",
    )
    a.set(
        yscale="log",
        xlabel="Actual request",
        ylabel="Health at decision",
        title=f"One life · seed {PILOT[0]} · no prior body training",
    )
    a.legend(frameon=False, fontsize=8)
    with (data / "per-life.csv").open(encoding="utf8", newline="") as f:
        allrows = list(csv.DictReader(f))
    for birth, color, marker, label in (
        ("blank", "#156c77", "o", "No prior body training"),
        ("population", "#ac572d", "^", "Population-initialized"),
    ):
        rows = [
            r
            for r in allrows
            if r["birth"] == birth and r["rule"] == "true1" and r["reached"] == "1"
        ]
        x = 100 * np.array([float(r["panel_fatal_vs_birth"]) for r in rows])
        y = 100 * np.array([float(r["panel_healthy_vs_birth"]) for r in rows])
        b.scatter(x, y, c=color, marker=marker, alpha=0.72, s=28, label=label)
        b.scatter(
            x.mean(),
            y.mean(),
            marker="X",
            s=115,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    b.axvline(0, color="#505050", lw=0.8)
    b.axhline(-2, color="#666666", ls="--", lw=0.9, label="Healthy retention margin")
    b.set(
        xlabel="Change in fatal answering (percentage points)",
        ylabel="Change in healthy completion (percentage points)",
        title="First-fatal snapshot versus its own birth",
    )
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.19)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=155, facecolor="white")
    fig.clear()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    show_lives(a.pilot, a.output)
    figure(a.pilot, a.analysis, a.output / "first-fatal.png")
