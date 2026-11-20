"""Show zeros and per-update differences without treating repeated inputs as independent."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(analysis, dest):
    with (analysis / "paired-updates.csv").open(encoding="utf8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    labels = [
        "Original request\noriginal health",
        "Original request\npost-injury health",
        "Next request\npost-injury health",
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, 3.8), layout="constrained", sharey=True
    )
    idx = np.arange(3)
    for ax, m, title in zip(
        axes,
        ("chosen", "answer"),
        ("Originally chosen answer action", "Either answer action"),
        strict=True,
    ):
        for birth, color, offset, name in (
            ("blank", "#286486", -0.09, "No prior body training"),
            ("population", "#ad582d", 0.09, "Population-trained"),
        ):
            subset = [r for r in cases if r["birth"] == birth]
            x = np.array(
                [
                    [
                        float(r[c + "_" + m + "_pp"])
                        for c in ("original", "health_only", "successor")
                    ]
                    for r in subset
                ]
            )
            for j, row in enumerate(x):
                xloc = idx + offset + (j % 7 - 3) * 0.009
                ax.plot(xloc, row, color=color, lw=0.6, alpha=0.13, zorder=1)
                ax.scatter(xloc, row, s=11, color=color, alpha=0.40, zorder=2)
            ax.scatter(
                idx + offset,
                x.mean(axis=0),
                s=62,
                color=color,
                edgecolor="#222222",
                linewidth=0.9,
                marker="D",
                label=name,
                zorder=3,
            )
        ax.axhline(0, color="#4f5860", ls="--", lw=0.7)
        ax.set_xticks(idx, labels=labels)
        ax.set_xlim(-0.35, 2.35)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Post-update minus pre-update (pp)")
    axes[1].legend(loc="best", fontsize=8, frameon=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("dest", type=Path)
    a = parser.parse_args()
    draw(a.analysis, a.dest)
