"""Seed-paired thresholds and physical opportunities in the stable ecology."""

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(folder, dest):
    with (folder / "panels.csv").open(encoding="utf8") as f:
        panel = [
            r
            for r in csv.DictReader(f)
            if r["world"] == "stable"
            and r["family"] == "qwen"
            and r["checkpoint"] == "last_living"
            and r["partner"] == "all"
        ]
    with (folder / "lives.csv").open(encoding="utf8") as f:
        lives = sorted(
            [
                r
                for r in csv.DictReader(f)
                if r["world"] == "stable"
                and r["kind"] == "qwen"
                and r["learning"] == "True"
            ],
            key=lambda r: r["seed"],
        )
    seeds = [r["seed"] for r in lives]
    data = {(r["seed"], r["split"], r["role"]): r for r in panel}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), layout="constrained")
    endpoints = [
        ("cooperation", 1, 5, "Answer and survive · H=6"),
        ("death", -1, 2, "Reduce total death · H=2"),
        ("healthy", 1, -2, "Complete task · H=90/300"),
    ]
    colors = ("#28697A", "#BE6D35")
    for ax, (metric, sign, threshold, title) in zip(axes, endpoints, strict=True):
        ax.axhline(0, color="#67727C", lw=0.9)
        ax.axhline(threshold, color="#596C47", lw=1.1, ls="--")
        for j, split in enumerate(("familiar", "held")):
            for k, ref in enumerate(("birth", "wrong")):
                x = 2 * j + k
                change = np.array(
                    [
                        100
                        * sign
                        * (
                            float(data[s, split, "current"][metric])
                            - float(data[s, split, ref][metric])
                        )
                        for s in seeds
                    ]
                )
                ax.scatter(
                    x + np.linspace(-0.14, 0.14, len(seeds)),
                    change,
                    color=colors[j],
                    alpha=0.68,
                    s=19,
                )
                ax.plot(
                    [x - 0.18, x + 0.18], [change.mean()] * 2, color="black", lw=2.3
                )
        ax.set_xticks(range(4), ["Birth", "Wrong", "Birth", "Wrong"])
        ax.set_xlabel("Original wording       Held wording")
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.10)
    axes[0].set_ylabel("Paired benefit (percentage points)")
    fig.suptitle(
        "One point per independent seed. Bars indicate means. Dashed lines show original-wording thresholds.",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("png", type=Path)
    a = p.parse_args()
    render(a.analysis, a.png)
