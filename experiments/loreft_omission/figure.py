"""Targets across partners after single and triple update omissions, one row per source life."""

import argparse, csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(folder, dest):
    def read(name):
        with (folder / name).open(encoding="utf8", newline="") as f:
            return list(csv.DictReader(f))

    lives = read("lives.csv")
    effects = read("event-effects.csv")
    codes = [int(s["seed"]) for s in lives]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.4, 4.7), layout="constrained", sharey=True
    )
    names = (
        "true anchor · omit last",
        "true anchor · omit last 3",
        "ordinary window · omit last 3",
    )
    for ax, title, (win, arm) in zip(
        axes,
        names,
        (("event", "skip1"), ("event", "skip3"), ("ordinary", "skip3")),
        strict=True,
    ):
        ax.axvline(0, color="#5C6A75", lw=0.9)
        ax.axvline(5, color="#73865B", lw=1, ls="--")
        for j, s in enumerate(codes):
            selected = [
                r
                for r in effects
                if int(r["seed"]) == s
                and r["family"] == "actual"
                and r["kind"] == "onset"
                and r["window"] == win
                and r["arm"] == arm
                and r["time"] == "at"
            ]
            if not selected:
                continue
            row = selected[0]
            a, b = (float(row[k]) for k in ("delta_p86", "delta_p93"))
            ax.plot([a, b], [j - 0.08, j + 0.08], color="#A3ADB1", lw=1)
            ax.scatter(
                a,
                j - 0.08,
                c="#20697A",
                s=30,
                marker="o",
                label="P86" if j == 0 else None,
                zorder=3,
            )
            ax.scatter(
                b,
                j + 0.08,
                c="#BB6432",
                s=30,
                marker="s",
                label="P93" if j == 0 else None,
                zorder=3,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xlim(-101, 101)
        ax.grid(axis="y", color="#D1D9DA", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Normal minus omission (pp)", fontsize=9)
    axes[0].set_yticks(range(len(codes)), [str(c) for c in codes])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Environment seed")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle(
        "Probability of the initial preferred action on identical healthy requests",
        fontsize=14,
    )
    fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("image", type=Path)
    a = p.parse_args()
    render(a.analysis, a.image)
