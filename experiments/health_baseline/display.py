"""Seed-level risk and healthy decisions under the two correct-credit baselines."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(table, out):
    with table.open(encoding="utf8", newline="") as f:
        data = list(csv.DictReader(f))
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 5.8), sharex=True, sharey="row", layout="constrained"
    )
    colors = ("#a65a2e", "#156c72")
    titles = ("No prior body training", "Population-trained birth")
    for col, birth in enumerate(("blank", "population")):
        for row, field in enumerate(("fatal", "healthy_panel")):
            ax = axes[row, col]
            values = []
            for rule in ("global_true", "local_true"):
                group = sorted(
                    (r for r in data if r["birth"] == birth and r["rule"] == rule),
                    key=lambda r: int(r["seed"]),
                )
                values.append(
                    np.array(
                        [
                            100
                            * (
                                float(r[field + "_current"])
                                - float(r[field + "_birth"])
                            )
                            for r in group
                        ]
                    )
                )
            g, l = values
            for a, b in zip(g, l, strict=True):
                ax.plot([0, 1], [a, b], color="#a8adb2", lw=0.75, alpha=0.55, zorder=1)
            for i, v in enumerate(values):
                ax.scatter(
                    np.repeat(i, len(v)), v, s=15, c=colors[i], alpha=0.64, zorder=2
                )
                ax.scatter(
                    i,
                    v.mean(),
                    marker="D",
                    s=60,
                    color=colors[i],
                    edgecolor="black",
                    lw=0.75,
                    zorder=3,
                )
            ax.axhline(0, color="#303b42", lw=0.85, zorder=0)
            if row == 1:
                ax.axhline(-2, color="#788382", ls=":", lw=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.22, 1.22)
            ax.set_xticks([0, 1], ["Lifetime baseline", "Similar-health baseline"])
            ax.grid(axis="y", color="#e3e6e7", lw=0.5)
            if not row:
                ax.set_title(titles[col], loc="left", size=12)
            if not col:
                ax.set_ylabel(
                    ("First-fatal: P(answer)", "Healthy panel: P(complete)")[row]
                    + "\nchange from own birth (pp)",
                    fontsize=9,
                )
    fig.suptitle(
        "Within-life change in Qwen decisions",
        x=0.065,
        ha="left",
        weight="bold",
        fontsize=16,
    )
    fig.savefig(out, dpi=165, metadata={"Software": "SLA finite-life study"})
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("lives_csv", type=Path)
    p.add_argument("output_png", type=Path)
    a = p.parse_args()
    draw(a.lives_csv, a.output_png)
