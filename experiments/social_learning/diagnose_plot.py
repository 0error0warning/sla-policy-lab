"""Optional matplotlib figure of saved-weight fitting and input sensitivity."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np


def draw(scores, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(scores.read_bytes())
    rows = [r for r in result["by_model"] if r["feedback"] == "aligned"]
    tuned = [r for r in rows if r["model"] == "saved_adapter"]
    base = {r["run_id"]: r for r in rows if r["model"] == "frozen"}
    colors = dict(base="#7c8592", tuned="#2364aa", constant="#2c8c81", target="#cf7934")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    fig.subplots_adjust(left=.07, right=.985, bottom=.23, top=.80, wspace=.28)
    x = np.arange(4)
    for offset, name, color, numbers in (
        (-.23, "Frozen Qwen", colors["base"], [base[r["run_id"]]["groups"]["train_original"]["full"]["mean_kl"] for r in tuned]),
        (0., "Saved aligned LoRA", colors["tuned"], [r["groups"]["train_original"]["full"]["mean_kl"] for r in tuned]),
        (.23, "Input-blind target mean", colors["constant"], [r["groups"]["train_original"]["full"]["context_free_mean_kl"] for r in tuned])):
        axes[0].bar(x + offset, numbers, width=.22, label=name, color=color)
    axes[0].set_xticks(x, [r["life"].replace("s", "").replace("-m", "\nmapping ") for r in tuned])
    axes[0].set_ylabel("KL to teacher target (lower is better)")
    axes[0].set_title("Fit on the original training prompts")
    groups = ("partner_only", "order_only", "question_only")
    x = np.arange(3)
    for offset, models, color in ((-.17, [base[r["run_id"]] for r in tuned], colors["base"]), (.17, tuned, colors["tuned"])):
        axes[1].bar(x + offset, [mean(r["pairs"][g]["mean_model_tv"] for r in models) for g in groups], width=.32, color=color)
    axes[1].scatter(x, [mean(r["pairs"][g]["mean_teacher_tv"] for r in tuned) for g in groups],
                    marker="D", s=35, color=colors["target"], label="Teacher-target change", zorder=4)
    axes[1].set_xticks(x, ["Partner", "Definition order", "Task question"])
    axes[1].set_ylabel("Mean paired probability change (TV)")
    axes[1].set_title("Change one input; hold the body fixed")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(handles + h, labels + l, loc="lower center", bbox_to_anchor=(.5, .015), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Read-only diagnostic of the four aligned-feedback adapters", y=.96, fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    draw(args.scores, args.output)
