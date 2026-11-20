"""Stage-wise paired causal contrasts and changes in action concentration."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def draw(analysis, out):
    names = ("early", "middle", "late")
    labels = ["Early\n8 updates", "Middle\n29–34", "Late\n59–69"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    colors = ("#176e74", "#934564")
    for run, (tag, c) in enumerate(zip(("first", "second"), colors, strict=True)):
        g = json.loads((analysis / tag / "summary.json").read_bytes())["by_stage"]
        for ax, metric in zip(
            axes, ("preference", "concentration", "healthy_difference"), strict=True
        ):
            vals = [
                g[st + ".concentration"]
                if metric == "concentration"
                else g[st + ".oracle_neutral"][metric]
                for st in names
            ]
            means = np.array([v["mean"] for v in vals])
            left, right = np.array([v["ci95"] for v in vals]).T
            xs = np.arange(3) + (run - 0.5) * 0.12
            ax.errorbar(
                xs,
                means,
                yerr=[means - left, right - means],
                color=c,
                fmt="o-",
                capsize=4,
                ms=5,
                lw=1.6,
            )
    for ax, title, ylab in zip(
        axes,
        (
            "Does true information help?",
            "How concentrated is the policy?",
            "Can a healthy agent answer?",
        ),
        (
            "Effective expression advantage (pp)",
            "Contexts with top action ≥90% (%)",
            "Answer-rate difference (pp)",
        ),
        strict=True,
    ):
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-0.35, 2.35)
        ax.grid(axis="y", color="#e2e7e8", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].axhline(1, color="#777", alpha=0.8, linestyle=":", lw=1)
    for ax in (axes[0], axes[2]):
        ax.axhline(0, lw=0.8, color="#888")
    axes[1].set_ylim(-3, 103)
    axes[2].axhline(-2, color="#777", alpha=0.8, linestyle=":", lw=1)
    axes[1].legend(
        [Line2D([0], [0], color=c, marker="o", lw=1.4) for c in colors],
        ["First run", "Technical repeat"],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Reading maintenance forecasts over one life",
        x=0.01,
        ha="left",
        fontsize=15,
        weight="semibold",
    )
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    draw(a.analysis, a.out)
