"""Three views of learned forecasts, action selection and healthy physical participation."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.individual_life.analysis import describe
from experiments.lifetime_cooperation.design import PILOT


def rows(path):
    with path.open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def draw(analysis, output):
    pred = rows(analysis / "prediction.csv")
    policy = rows(analysis / "policy.csv")
    lives = rows(analysis / "life.csv")
    colors = {"same": "#237b84", "opposite": "#93516b"}
    rules = {int(r["seed"]): r["rules"] for r in pred}

    def metric(data, mode, partner, key):
        d = {
            int(r["seed"]): float(r[key])
            for r in data
            if r["phase"] == "replay"
            and r["split"] == "held"
            and r["mode"] == mode
            and r["partner"] == partner
        }
        return np.array([d[s] for s in PILOT])

    def healthy(arm):
        d = {
            int(r["seed"]): 100 * int(r["healthy_tasks"]) / int(r["healthy_rounds"])
            for r in lives
            if r["phase"] == "live" and r["arm"] == arm and r["partner"] == "both"
        }
        return np.array([d[s] for s in PILOT])

    blocks = [
        (
            "P93 experience and forecast accuracy",
            "Decrease in forecast squared error",
            [
                (
                    "P93",
                    metric(pred, "drop93", "P93", "error")
                    - metric(pred, "true", "P93", "error"),
                ),
                (
                    "P86",
                    metric(pred, "drop93", "P86", "error")
                    - metric(pred, "true", "P86", "error"),
                ),
            ],
        ),
        (
            "Effective expression at health 6",
            "Change in expression preference (pp)",
            [
                (
                    "vs body only",
                    metric(policy, "true", "both", "advantage")
                    - metric(policy, "body", "both", "advantage"),
                ),
                (
                    "vs wrong ID",
                    metric(policy, "true", "both", "advantage")
                    - metric(policy, "wrong", "both", "advantage"),
                ),
                (
                    "vs flipped forecast",
                    metric(policy, "true", "both", "advantage")
                    - metric(policy, "flip", "both", "advantage"),
                ),
            ],
        ),
        (
            "Tasks completed while healthy",
            "Change in actual completion rate (pp)",
            [
                ("vs body only", healthy("true") - healthy("body")),
                ("vs wrong ID", healthy("true") - healthy("wrong")),
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), layout="constrained")
    for ax, (title, label, groups) in zip(axes, blocks, strict=True):
        ax.axvline(0, color="#777e86", lw=1, ls="--")
        for y, (name, values) in enumerate(groups):
            dy = np.linspace(-0.18, 0.18, len(values))
            for offset, seed, value in zip(dy, PILOT, values, strict=True):
                ax.scatter(
                    value,
                    y + offset,
                    s=26,
                    facecolors=colors[rules[seed]],
                    edgecolors="white",
                    lw=0.4,
                    zorder=3,
                )
            ci = describe(values)["ci95"]
            ax.plot(ci, [y + 0.32] * 2, color="#1d2330", lw=1.6)
            ax.scatter(
                values.mean(), y + 0.32, color="#1d2330", marker="D", s=29, zorder=4
            )
        ax.set_yticks(np.arange(len(groups)), [name for name, _ in groups])
        ax.set_ylim(len(groups) - 0.33, -0.5)
        ax.set_title(title, loc="left", fontsize=11, pad=12, weight="semibold")
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(labelsize=8, length=2)
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b8bec2")
        ax.grid(axis="x", color="#e9e9e9", lw=0.6)
    fig.suptitle(
        "Eight paired lifetimes  |  forecasts, Qwen choices and healthy tasks",
        x=0.01,
        ha="left",
        fontsize=15,
    )
    fig.savefig(output, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    draw(a.analysis, a.output)
