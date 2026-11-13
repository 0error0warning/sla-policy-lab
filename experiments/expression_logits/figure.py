"""Native early score changes and the two trials sharing the same action draw."""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .design import CASE_ID, TWIN_ID


def draw(analysis, out):
    base = analysis / "first"
    with gzip.open(
        base / "native_contrasts.csv.gz", "rt", encoding="utf8", newline=""
    ) as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["completed"] == "8"
            and r["wording"] == "held"
            and r["target"] == "oracle"
            and r["reference"] == "neutral"
        ]
    # The old sampling intervals are identified by the archived probabilities even
    # on hardware where a new full forward falls on the other side of the draw.
    with (base / "archived_case_boundaries.csv").open(encoding="utf8", newline="") as f:
        case = {(int(r["prompt"]), r["view"]): r for r in csv.DictReader(f)}
    xy = np.array([[float(r["d_help"]), float(r["d_trade"])] for r in rows])
    points, counts = np.unique(xy, axis=0, return_counts=True)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 4.6), layout="constrained", width_ratios=(1, 1.3)
    )
    lim = max(0.08, 1.22 * np.abs(points).max())
    left.plot(
        [-lim, lim], [-lim, lim], linestyle="--", lw=0.9, color="#7e8995", zorder=0
    )
    left.axhline(0, color="#dce2e6", lw=0.8)
    left.axvline(0, color="#dce2e6", lw=0.8)
    left.scatter(
        points[:, 0],
        points[:, 1],
        s=20 + 17 * np.sqrt(counts),
        color="#207373",
        edgecolor="white",
        lw=0.65,
        alpha=0.88,
    )
    for (x, y), count in zip(points, counts, strict=True):
        if count >= 10:
            left.annotate(
                str(count),
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
                color="#1e4448",
            )
    left.set_aspect("equal", adjustable="box")
    left.set_xlim(-lim, lim)
    left.set_ylim(-lim, lim)
    left.xaxis.set_major_locator(MaxNLocator(5))
    left.yaxis.set_major_locator(MaxNLocator(5))
    left.set_xlabel("Change in appeal logit")
    left.set_ylabel("Change in reciprocal-offer logit")
    left.set_title(
        "After 8 updates  |  true rate minus 0.5",
        loc="left",
        fontsize=11,
        weight="semibold",
    )
    colors = {"learned": "#4a6791", "oracle": "#237b70", "neutral": "#aa5965"}
    orders = ("learned", "oracle", "neutral")
    for j, i in enumerate((CASE_ID, TWIN_ID)):
        for k, mode in enumerate(orders):
            r = case[i, mode]
            y = j + (0.16 * (k - 1))
            right.scatter(
                100 * float(r["upper_help"]), y, color=colors[mode], s=52, zorder=4
            )
    u = 100 * float(case[CASE_ID, "neutral"]["u"])
    right.axvline(
        u, color="#262e37", lw=1.2, linestyle="--", label="Shared action draw"
    )
    right.set_yticks([0, 1], ["Held wording\n#369", "Matched wording\n#345"])
    right.set_ylim(1.45, -0.45)
    right.set_title(
        "P93 at midlife  |  archived sampling boundary",
        loc="left",
        fontsize=11,
        weight="semibold",
    )
    right.set_xlabel("Cumulative probability through appeal (%)")
    right.grid(axis="x", color="#e6e9eb", lw=0.7)
    right.legend(
        [Line2D([0], [0], marker="o", linestyle="", color=colors[k]) for k in orders]
        + [Line2D([0], [0], color="#262e37", linestyle="--")],
        ["Learned", "True", "0.5", "Shared draw"],
        loc="lower right",
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, length=3)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    draw(a.analysis, a.out)
