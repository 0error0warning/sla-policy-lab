"""Export paired behavior and causal effects. One dot per environment seed."""

import argparse, csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    with p.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def draw(folder, dest):
    behavior = read(folder / "behavior-by-seed.csv")
    causal = read(folder / "steering-by-seed.csv")
    methods = read(folder / "methods.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained")

    def dot(ax, x, a, color):
        vals = np.array(a, dtype=float)
        ax.scatter(
            x + np.linspace(-0.13, 0.13, len(vals)), vals, s=22, color=color, alpha=0.8
        )
        ax.plot([x - 0.21, x + 0.21], [vals.mean()] * 2, color="black", lw=2.0)

    colors = ["#427A83", "#AC6A41", "#6C65A0"]
    birth = [float(r["held_social"]) for r in methods if r["method"] == "birth"]
    for j, role in enumerate(("native_lora", "residual_lora", "loreft")):
        v = [float(r["held_social"]) for r in methods if r["method"] == role]
        dot(axes[0], j, np.asarray(v) - birth, colors[j])
    axes[0].set_xticks([0, 1, 2], ["Original\nLoRA", "Matched\nLoRA", "LoReFT"])
    axes[0].set_title(
        "Partner-matched expression\nHeld words, relative to birth", fontsize=11
    )
    axes[0].axhline(3, color="#70815F", lw=1, ls="--")
    for j, k in enumerate(("adapt_S", "other_spill_L", "adapt_S_minus_L")):
        dot(axes[1], j, [r[k] for r in behavior], colors[j])
    axes[1].set_xticks(
        [0, 1, 2], ["Switched\npartner", "Other partner\nabsolute", "Difference"]
    )
    axes[1].set_title(
        "Reversing one partner's responses\nHeld words, trained LoReFT", fontsize=11
    )
    for j, k in enumerate(
        ("positive_minus_negative_half", "remove_drop", "random_drop_abs", "swap_drop")
    ):
        dot(axes[2], j, [r[k] for r in causal], colors[j % 3])
    axes[2].set_xticks(
        [0, 1, 2, 3],
        ["Add ±\nhalf-diff", "Remove\ndrop", "Random\nabs. drop", "Swap\ndrop"],
    )
    axes[2].set_title(
        "Activation interventions\nHeld seed, wording and task", fontsize=11
    )
    for ax in axes:
        ax.axhline(0, color="#66727A", lw=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.13)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Percentage points", fontsize=9)
    fig.savefig(dest, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    draw(a.folder, a.dest)
