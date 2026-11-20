"""Compare irreversible health loss and action probabilities in the prior Qwen records."""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.first_fatal.design import FATAL
from experiments.first_fatal.design import settle as trial_settle
from experiments.social_closed_loop.core import settle as paper_settle


def draw(root, dest):
    with gzip.open(root / "rounds.csv.gz", "rt", newline="", encoding="utf8") as stream:
        risky = [
            r
            for r in csv.DictReader(stream)
            if r["band"] == "practice" and int(r["action"]) >= 2
        ]
    signal = np.array([float(r["gate_signal"]) for r in risky])
    change = np.array([100 * float(r["gate_next_delta"]) for r in risky])
    fig, (body, policy) = plt.subplots(1, 2, figsize=(10, 3.65), layout="constrained")
    h = np.unique(np.r_[np.geomspace(1, 350, 500), FATAL, np.nextafter(FATAL, np.inf)])
    paper = [x - paper_settle(x, 2, True, [0] * 5)[0] for x in h]
    trial = [x - trial_settle(x, 2, True, [0] * 5)[0] for x in h]
    body.plot(h, paper, color="#315d8a", label="Paper, Eq. 4", lw=2)
    body.plot(h, trial, color="#ac4b25", label="PR 9–12", lw=2)
    body.axvspan(FATAL, 6, color="#e7c77e", alpha=0.32)
    body.axvline(FATAL, color="#796c53", lw=0.8, ls=":")
    body.set(
        xscale="log",
        yscale="log",
        xlabel="Pre-action health",
        ylabel="Health lost",
        title="Same maintained answer, zero noise",
    )
    body.set_xticks(
        [2, 5, 12, 48, 175, 350], labels=["2", "5", "12", "48", "175", "350"]
    )
    body.legend(frameon=False, loc="lower right")
    body.grid(axis="y", color="#e0e4e7", lw=0.6)
    groups = [signal < -1e-12, signal > 1e-12]
    bottom = np.zeros(2, dtype=int)
    for label, color, mask in (
        ("P(answer) lower", "#315d8a", change < 0),
        ("Unchanged", "#d7dce0", change == 0),
        ("Higher", "#ac4b25", change > 0),
    ):
        counts = np.array([np.sum(mask & group) for group in groups])
        policy.bar([0, 1], counts, bottom=bottom, label=label, color=color, width=0.55)
        for x, (b, n) in enumerate(zip(bottom, counts, strict=True)):
            if n >= 20:
                policy.text(
                    x,
                    b + n / 2,
                    f"{n}",
                    ha="center",
                    va="center",
                    color="#161d23" if label == "Unchanged" else "white",
                    fontsize=10,
                )
        bottom += counts
    policy.set_xticks([0, 1], labels=["Negative signal", "Positive signal"])
    policy.set(
        ylabel="Surviving risky answers",
        title="One LoRA update on the next request",
        xlim=(-0.55, 1.8),
    )
    policy.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in (body, policy):
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(dest, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    draw(args.root, args.dest)
