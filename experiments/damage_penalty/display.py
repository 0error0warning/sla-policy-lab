"""Show paired effects per independent seed and uncertainty at three points in the life."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(analysis, out):
    def rows(name):
        with (analysis / name).open(encoding="utf8", newline="") as f:
            return list(csv.DictReader(f))

    replay = rows("primary/replay-seeds.csv")
    lives = rows("lives.csv")
    numbers = json.loads((analysis / "live.json").read_bytes())
    tape = json.loads((analysis / "primary/replay.json").read_bytes())
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for j, (title, ylabel) in enumerate(
        (
            (
                "After a survivable answering injury",
                "Next P(answer)\n2× step minus 1× step (pp)",
            ),
            (
                "At the first fatal request",
                "P(fatal answer)\n2× learner minus birth (pp)",
            ),
            (
                "While healthy in the actual life",
                "Task completion rate\n2× learner minus frozen (pp)",
            ),
        )
    ):
        ax = axs[j]
        for i, (birth, color) in enumerate(
            (("blank", "#356e90"), ("population", "#aa5b37"))
        ):
            if j == 0:
                vs = [
                    float(r["mean_delta"])
                    for r in replay
                    if r["birth"] == birth
                    and r["exposure"] == "risky_answer"
                    and r["contrast"] == "two_minus_one_step"
                ]
                est = tape["risky_answer_" + birth]["two_minus_one_step"][
                    "seed_balanced_delta"
                ]
            else:
                two = sorted(
                    (
                        r
                        for r in lives
                        if r["birth"] == birth and r["rule"] == "two_true"
                    ),
                    key=lambda r: r["seed"],
                )
                frozen = sorted(
                    (r for r in lives if r["birth"] == birth and r["rule"] == "frozen"),
                    key=lambda r: r["seed"],
                )
                if j == 1:
                    vs = [
                        float(r["fatal_current"]) - float(r["fatal_birth"]) for r in two
                    ]
                    est = numbers[birth]["matched"]["two_true"]["fatal_vs_birth"]
                else:
                    vs = [
                        float(a["healthy_rate"]) - float(b["healthy_rate"])
                        for a, b in zip(two, frozen, strict=True)
                    ]
                    est = numbers[birth]["freely_acting"]["frozen"][
                        "actual_healthy_delta"
                    ]
            y = np.array(vs) * 100
            ax.scatter(
                i + np.linspace(-0.11, 0.11, len(y)),
                y,
                s=18,
                c=color,
                alpha=0.55,
                zorder=2,
            )
            lo, hi = np.array(est["ci95"]) * 100
            ax.plot([i, i], [lo, hi], lw=2.4, c="black", zorder=3)
            ax.scatter(
                i,
                100 * est["mean"],
                s=40,
                marker="D",
                c=color,
                edgecolor="black",
                zorder=4,
            )
        ax.axhline(0, c="#39464d", lw=0.8)
        if j == 2:
            ax.axhline(-2, c="#717b80", lw=0.8, ls=":")
        ax.set_title(title, size=10, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1], ["No body\ntraining", "Population-\ntrained"])
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        "How does doubling the damage penalty change Qwen’s decisions?",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(out, dpi=165, metadata={"Software": "SLA finite-life study"})
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("png", type=Path)
    a = p.parse_args()
    draw(a.analysis, a.png)
