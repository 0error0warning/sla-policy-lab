"""Seed-level feedback and transfer, and separately reported identity decodability."""

import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(analysis, dest):
    def table(name):
        with (analysis / name).open(encoding="utf8", newline="") as f:
            return list(csv.DictReader(f))

    cases = [r for r in table("cases.csv") if r["ordinary"] == "True"]
    sites = [
        r
        for r in table("positions.csv")
        if r["ordinary"] == "True" and r["site"] in ("early", "terminal")
    ]
    readout = table("identity-readout.csv")
    seeds = sorted({r["seed"] for r in cases})
    fig, ax = plt.subplots(1, 3, figsize=(11.7, 3.7), layout="constrained")
    colors = ["#246D79", "#B56737"]
    for x, (key, label) in enumerate(
        (("body_tv_source", "Source"), ("body_tv_other", "Other"))
    ):
        y = [np.mean([float(r[key]) for r in cases if r["seed"] == s]) for s in seeds]
        ax[0].scatter(x + np.linspace(-0.10, 0.10, len(y)), y, color=colors[x], s=20)
        ax[0].plot([x - 0.2, x + 0.2], [np.mean(y)] * 2, color="black", lw=2)
    ax[0].set_xticks((0, 1), ("Source ID", "Other ID"))
    ax[0].set_xlim(-0.55, 1.55)
    ax[0].set_ylabel("Action distribution change (TV, pp)")
    ax[0].set_title("Light vs heavy damage\nFour feedback cells")
    for j, stage in enumerate(("early_id", "early_decision", "terminal_decision")):
        v = [r for r in readout if r["layer"] == stage]
        ax[1].plot(
            range(4),
            [100 * float(r["accuracy"]) for r in v],
            marker=["o", "s", "^"][j],
            color=["#71905E", *colors][j],
            linewidth=1.3,
            label=(
                "ID token at block 4",
                "Decision at block 4",
                "Decision at last layer",
            )[j],
        )
    ax[1].axhline(50, color="#777777", lw=0.9, ls="--")
    ax[1].set_xticks(range(4), ["2", "6", "90", "300"])
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("Held-task health")
    ax[1].set_ylabel("Held-task accuracy (%)")
    ax[1].legend(fontsize=7, frameon=False, loc="best")
    ax[1].set_title("Can the partner ID be decoded?")
    for j, layer in enumerate(("early", "terminal")):
        chosen = [r for r in sites if r["site"] == layer]
        y = [
            np.mean([float(r["specificity"]) for r in chosen if r["seed"] == s])
            for s in seeds
        ]
        ax[2].scatter(j + np.linspace(-0.1, 0.1, len(y)), y, s=20, color=colors[j])
        ax[2].plot([j - 0.2, j + 0.2], [np.mean(y)] * 2, color="black", lw=2)
    ax[2].axhline(0, color="#777777", lw=0.9)
    ax[2].set_xticks((0, 1), ("Block 4", "Last layer"))
    ax[2].set_xlim(-0.55, 1.55)
    ax[2].set_title("One identical-budget write\nSource minus other ID")
    ax[2].set_ylabel("Difference in TV (pp)")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(axis="y", alpha=0.14)
        x.tick_params(labelsize=8)
    fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("image", type=Path)
    a = p.parse_args()
    draw(a.analysis, a.image)
