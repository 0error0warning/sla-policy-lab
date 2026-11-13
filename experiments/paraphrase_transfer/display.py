"""Display paired wording-training effects and the observed partner strata."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(analysis, dest):
    data = json.loads((analysis / "summary.json").read_bytes())["groups"]["all"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    endpoints = (
        "held_new_wording",
        "actual_next",
        "actual_next_same_partner",
        "actual_next_changed_partner",
    )
    pos = np.arange(len(endpoints))
    for off, m, label, color in (
        (-0.09, "chosen", "Penalized action", "#216186"),
        (0.09, "answer", "Either answer", "#ad502c"),
    ):
        points = [data[e][m]["augmented_minus_original"] for e in endpoints]
        line(axes[0], pos + off, points, label, color)
    axes[0].set_xticks(
        pos,
        labels=[
            "Unseen\nwordings",
            "All actual\nsuccessors",
            "Same\npartner",
            "Changed\npartner",
        ],
    )
    axes[0].set_title("Transfer at post-injury health")
    axes[0].set_ylabel("Paraphrases minus original-only (pp)")
    axes[0].legend(fontsize=8, frameon=False)
    pos = np.arange(2)
    for off, key, label, color in (
        (-0.09, "augmented_minus_original", "Versus original-only", "#216186"),
        (0.09, "augmented_minus_pre", "Versus pre-update", "#ad502c"),
    ):
        line(
            axes[1],
            pos + off,
            [data[e]["answer"][key] for e in ("healthy_panel", "healthy_rollout")],
            label,
            color,
        )
    axes[1].set_title("Healthy task completion")
    axes[1].set_ylabel("Completion-rate difference (pp)")
    axes[1].set_xticks(pos, labels=["Reserved\nquestions", "Observed\nhealthy tasks"])
    axes[1].axhline(-2, color="#7f8189", ls=":", lw=1)
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.axhline(0, color="#676b74", lw=0.8)
        ax.spines[["right", "top"]].set_visible(False)
        ax.margins(x=0.14)
    fig.savefig(dest, dpi=155)
    plt.close(fig)


def line(ax, x, points, label, color):
    mean = np.array([p["mean"] for p in points])
    interval = np.array([p["ci95"] for p in points])
    err = np.maximum(0, np.stack((mean - interval[:, 0], interval[:, 1] - mean)))
    ax.errorbar(x, mean, yerr=err, fmt="o", color=color, capsize=3, ms=6, label=label)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    draw(a.analysis, a.dest)
