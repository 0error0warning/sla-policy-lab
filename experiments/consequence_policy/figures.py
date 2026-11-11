"""Plot confirmatory seed means and paired uncertainty from the delivered raw results."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.consequence_policy.study import CONFIRM, TEST_H
from experiments.direct_policy.protocol import rng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.results
    font = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = font.exists()
    if chinese:
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font))
        mpl.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font)
        ).get_name()
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.35), layout="constrained")
    boots = rng(63994, "figure").integers(len(CONFIRM), size=(4000, len(CONFIRM)))
    colors = ["#02796f", "#727882", "#ce6c31", "#4865a8"]
    names = (
        ["真实后果", "冻结", "打乱后果", "无身体传感"]
        if chinese
        else ["True outcome", "Frozen", "Shuffled", "Blind"]
    )
    for j, arm in enumerate(("real", "frozen", "shuffle", "blind")):
        a = (
            np.array(
                [
                    np.mean(
                        [
                            json.loads(
                                (root / f"s{seed}-m{m}/metrics.json").read_bytes()
                            )["metrics"][arm]["answer"]
                            for m in (0, 1)
                        ],
                        axis=0,
                    )
                    for seed in CONFIRM
                ]
            )
            * 100
        )
        lo, hi = np.quantile(a[boots].mean(axis=1), [0.025, 0.975], axis=0)
        axes[0].plot(
            TEST_H,
            a.mean(axis=0),
            marker="o",
            ms=3,
            color=colors[j],
            label=names[j],
            lw=2 if j == 0 else 1.3,
        )
        axes[0].fill_between(TEST_H, lo, hi, color=colors[j], alpha=0.1)
    axes[0].set_xscale("log")
    axes[0].set_xticks([2, 6, 18, 90, 300], ["2", "6", "18", "90", "300"])
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel(
        "当前身体健康值（对数轴）" if chinese else "Current health (log scale)"
    )
    axes[0].set_ylabel("选择回答的概率（%）" if chinese else "Answer probability (%)")
    axes[0].set_title(
        "同一请求下，身体改变决策" if chinese else "Choice across body states"
    )
    axes[0].legend(fontsize=8, loc="lower right")
    for i, series in enumerate(("coverage", "allocation")):
        stats = json.loads((root / f"{series}-confirm.json").read_bytes())["stats"]
        for j, key in enumerate(
            (
                "visible_vs_high_fatal_answer_reduction",
                "visible_vs_high_healthy_answer_change",
            )
        ):
            obs = stats[key]
            val = 100 * obs["mean"]
            lo, hi = np.asarray(obs["ci95"]) * 100
            y = 2.2 - i * 1.4 - 0.35 * j
            axes[1].errorbar(
                val,
                y,
                xerr=[[val - lo], [hi - val]],
                fmt="o",
                color=colors[j * 2],
                capsize=3,
            )
    axes[1].set_yticks(
        [2.2, 1.85, 0.8, 0.45],
        [
            "5/8 · 危险回答减少",
            "5/8 · 健康回答变化",
            "3/8 · 危险回答减少",
            "3/8 · 健康回答变化",
        ]
        if chinese
        else [
            "5/8 · fatal reduction",
            "5/8 · healthy change",
            "3/8 · fatal reduction",
            "3/8 · healthy change",
        ],
    )
    axes[1].axvline(0, color="#aaa", lw=1)
    axes[1].axvline(-5, color="#c87843", lw=0.8, ls="--")
    axes[1].set_ylim(0, 2.62)
    axes[1].set_xlabel(
        "相对高健康训练组（百分点）" if chinese else "Difference from high-only (pp)"
    )
    axes[1].set_title(
        "相同预算，训练区间的取舍" if chinese else "Allocation at fixed budget"
    )
    study = json.loads((root / "singleton-lives/life-confirm.json").read_bytes())[
        "absolute"
    ]
    arms = ("prior", "personal", "scratch")
    labels = (
        ["共同起点冻结", "一生内更新", "从零更新"]
        if chinese
        else ["Fixed prior", "Personal", "From scratch"]
    )
    pos = np.arange(len(arms))
    axes[2].bar(
        pos - 0.17,
        [study[a]["turns"] for a in arms],
        0.32,
        color="#a8bdcc",
        label="生存回合" if chinese else "Encounters",
    )
    axes[2].bar(
        pos + 0.17,
        [study[a]["tasks"] for a in arms],
        0.32,
        color=colors[0],
        label="完成任务" if chinese else "Tasks",
    )
    axes[2].set_xticks(pos, labels, fontsize=9)
    axes[2].set_ylabel("每条完整生命的均值" if chinese else "Mean per full life")
    axes[2].set_title("独立单条推理下的完整生命" if chinese else "Singleton deployment")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#ddd", lw=0.5)
        ax.set_axisbelow(True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
