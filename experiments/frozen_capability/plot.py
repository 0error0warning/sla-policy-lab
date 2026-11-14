"""Plot all registered presentations, keeping missing outputs visible."""

import argparse
import gzip
import json
from pathlib import Path

from .design import HEALTHS

COLORS = ["#E4E7EB", "#99A5B5", "#E9B872", "#CD6C70", "#59AA95"]
LABELS = [
    "No archived output",
    "No final (budget)",
    "Format / channel",
    "Wrong choice",
    "Correct choice",
]


def plot(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    with gzip.open(source, "rt", encoding="utf8") as file:
        records = [json.loads(line) for line in file]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.4), sharey=True)
    mode_pairs = [(m, t) for m in ("4B", "9B") for t in (False, True)]
    for i, arm in enumerate(("known_rules", "given_outcomes")):
        for j, (model, thinking) in enumerate(mode_pairs):
            ax = axes[i, j]
            rows = [
                r
                for r in records
                if r["model"] == model and r["thinking"] == thinking and r["arm"] == arm
            ]
            assert len(rows) == 32
            matrix = [[0] * 4 for _ in HEALTHS]
            for r in rows:
                y = HEALTHS.index(r["health"])
                order = int(r["id"].rsplit("-order", 1)[1][0])
                x = (0 if r["favorite"] == "help" else 2) + order
                value = (
                    0
                    if not r["executed"]
                    else 1
                    if r["unfinished"]
                    else 2
                    if not r["choice_valid"]
                    else 4
                    if r["choice_correct"]
                    else 3
                )
                matrix[y][x] = value
                word = (
                    r["choice"]
                    if r["choice_valid"]
                    else (
                        "-"
                        if not r["executed"]
                        else "time"
                        if r["unfinished"]
                        else "fmt"
                    )
                )
                ax.text(x, y, word, ha="center", va="center", fontsize=8.5)
            ax.imshow(
                matrix, cmap=ListedColormap(COLORS), vmin=0, vmax=4, aspect="auto"
            )
            ax.set_xticks(range(4), ["H/0", "H/1", "T/0", "T/1"], fontsize=8)
            ax.set_yticks(range(len(HEALTHS)), [f"{h:g}" for h in HEALTHS], fontsize=9)
            ax.set_xticks([x - 0.5 for x in range(5)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(9)], minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", left=False, bottom=False)
            ax.set_title(f"{model} {'thinking' if thinking else 'direct'}", fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    ("Known rules" if i == 0 else "Given correct outcomes") + "\nHealth"
                )
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Frozen model diagnostic: decision and completion by condition", fontsize=15
    )
    fig.legend(
        [Patch(facecolor=c) for c in COLORS],
        LABELS,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "H/T = higher maintenance rate for help/trade; 0/1 = reversed presentation orders.\n"
        "Colours evaluate the choice only; numerical prediction errors are reported separately.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plot(a.source, a.output)
