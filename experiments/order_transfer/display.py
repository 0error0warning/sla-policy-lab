"""Absolute changes of training arms versus their common pre-update snapshot."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(analysis, dest):
    s = json.loads((analysis / "summary.json").read_bytes())["groups"]["all"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.05), layout="constrained")
    specifications = (
        (
            "Danger after one injury",
            [
                ("injured_held", "chosen"),
                ("injured_held", "answer"),
                ("actual_next", "chosen"),
                ("actual_next", "answer"),
            ],
            [
                "Held orders\nchosen action",
                "Held orders\nall answers",
                "Next request\nchosen action",
                "Next request\nall answers",
            ],
        ),
        (
            "Safe task completion",
            [("healthy_panel", "answer"), ("healthy_rollout", "answer")],
            ["Reserved panel\nexpected tasks", "Healthy rollout\nobserved tasks"],
        ),
    )
    for ax, (title, pairs, labels) in zip(axes, specifications, strict=True):
        pos = np.arange(len(pairs))
        for off, which, name, color in (
            (-0.11, "repeat", "Original order", "#306987"),
            (0.11, "rotate", "Four orders", "#ae4f2a"),
        ):
            stats = [s[k][m][which + "_minus_pre"] for k, m in pairs]
            mean = np.array([x["mean"] for x in stats])
            bounds = np.array([x["ci95"] for x in stats])
            ax.errorbar(
                pos + off,
                mean,
                yerr=np.maximum(
                    0, np.stack((mean - bounds[:, 0], bounds[:, 1] - mean))
                ),
                fmt="o",
                ms=6,
                capsize=3,
                color=color,
                label=name,
            )
        ax.axhline(0, color="#4d5361", lw=0.85)
        ax.set_xticks(pos, labels=labels, fontsize=8)
        ax.set_xlim(-0.4, len(pos) - 0.6)
        ax.set_title(title)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Post-update minus pre-update (pp)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(-2, ls=":", color="#747a82", lw=1.1)
    axes[1].set_ylabel("Completion-rate change (pp)")
    fig.savefig(dest, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    draw(a.analysis, a.dest)
