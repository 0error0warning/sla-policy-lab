"""Plot a prespecified individual, exposing its real sequence and matched counterfactual choices."""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from experiments.individual_life.analysis import records


def plot(root, analysis, out, birth="blank", seed=68011):
    name = f"visible/s{seed}/{birth}/true1"
    events = records(root / name / "life.jsonl")
    with (analysis / "next-living.csv").open(encoding="utf8", newline="") as f:
        contrasts = [r for r in csv.DictReader(f) if r["life"] == name]
    turns = np.arange(1, len(events) + 1)
    h = np.array([e["health_before"] for e in events])
    seen = np.array([e["maintenance"] is True for e in events])
    title = (
        "Without prior body training"
        if birth == "blank"
        else "With prior body training"
    )
    fig = Figure(figsize=(9, 5.1), layout="constrained")
    top, low = fig.subplots(2, 1, sharex=True)
    fig.suptitle(
        f"One Qwen life · {title} · seed {seed}", fontsize=14, fontweight="bold"
    )
    top.plot(
        np.r_[turns, turns[-1] + 1],
        np.r_[h, 0],
        color="#163c57",
        lw=1.8,
        label="Health at each request",
    )
    top.scatter(
        turns[seen],
        h[seen],
        s=14,
        color="#137c66",
        label="Maintenance received",
        zorder=3,
    )
    top.scatter(
        [turns[-1] + 1],
        [0],
        color="#9e3e26",
        s=36,
        marker="x",
        zorder=3,
        label="Body reaches zero after final action",
    )
    top.set_ylabel("Body health")
    top.set_ylim(-15, 375)
    top.legend(loc="upper right", frameon=False, fontsize=8)
    t = np.array([int(r["turn"]) for r in contrasts])
    assert np.array_equal(t, turns[1:])
    for k, label, c in (
        ("individual_vs_birth_quality", "Since its own birth", "#18558b"),
        (
            "individual_vs_previous_quality",
            "From the immediately preceding update",
            "#aa4d25",
        ),
    ):
        y = 100 * np.array([float(r[k]) for r in contrasts])
        low.plot(t, y, color=c, lw=1.3, label=label)
    flips = t[[bool(int(float(r["actual_choice_would_change"]))) for r in contrasts]]
    if len(flips):
        low.scatter(
            flips,
            np.full(len(flips), 0.025),
            transform=low.get_xaxis_transform(),
            marker="|",
            s=70,
            color="#242424",
            label="Same draw changes the next action",
        )
    low.axhline(0, color="#555555", lw=0.6)
    low.set_ylabel("Next-decision value change\n(% of four-action range)")
    low.set_xlabel("Actual request in this individual’s life")
    low.legend(loc="upper left", fontsize=8, frameon=False, ncols=1)
    for ax in (top, low):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, color="#d9e0e5")
        ax.set_axisbelow(True)
    low.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165, facecolor="white")
    fig.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    plot(a.root, a.analysis, a.output)
