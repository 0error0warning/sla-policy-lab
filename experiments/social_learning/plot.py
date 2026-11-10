"""Render continuous-body trajectories from the archived social pilot (optional matplotlib)."""

import argparse
from pathlib import Path
import zipfile

from score import cpu_baseline, read, verify_archive


def draw(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(output)
    data = read(root / "protocol.json")
    colors = {"aligned": "#2364aa", "frozen": "#69717f", "shuffled": "#db7835", "rule": "#2c8c81"}
    names = {"aligned": "Real feedback LoRA", "frozen": "Frozen", "shuffled": "Shuffled feedback LoRA", "rule": "Experience-based rule"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.18, top=.89, hspace=.23, wspace=.06)
    for ax, life in zip(axes.flat, data["lives"]):
        prefix = life["trace"]
        ax.plot([0] + [r["after"]["age"] for r in prefix], [.08] + [r["after"]["damage"] for r in prefix],
                color="#353b43", linewidth=2, linestyle="--")
        x0, y0 = prefix[-1]["after"]["age"], prefix[-1]["after"]["damage"]
        for arm in ("frozen", "shuffled", "aligned", "rule"):
            rows = (cpu_baseline(life, "empirical_reference", data["datasets"][life["id"]]["aligned"]["posterior"])
                    if arm == "rule" else read(root / "output" / (life["id"] + "-" + arm + "-rollout.json"))["trace"])
            ax.plot([x0] + [r["after"]["age"] for r in rows], [y0] + [r["after"]["damage"] for r in rows],
                    color=colors[arm], label=names[arm], linewidth=2, linestyle=":" if arm == "rule" else "-")
        ax.axvline(32, color="#b7bdc4", linewidth=1)
        ax.axhline(1, color="#c4c9cf", linewidth=1)
        ax.set_title(f"Seed {life['seed']}, partner mapping {life['mapping']}", fontsize=11)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015), ncol=4, frameon=False, fontsize=9)
    fig.supxlabel("Encounter (weights updated once at turn 32)", y=.085)
    fig.supylabel("Cumulative body damage", x=.01)
    fig.suptitle("One persistent body, then matched policy continuations", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with verify_archive(args.archive) as archive:
        draw(zipfile.Path(archive), args.output)
