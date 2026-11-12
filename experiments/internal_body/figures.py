"""Plot paired independent-seed estimates, with simulated bodies and four actual action words."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def read(p):
    return json.loads(p.read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--credit", type=Path, required=True)
    p.add_argument("--routing", type=Path)
    p.add_argument("--depth", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    font = Path("C:/Windows/Fonts/msyh.ttc")
    zh = font.is_file()
    if zh:
        font_manager.fontManager.addfont(str(font))
        mpl.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font)
        ).get_name()
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    evidence = read(a.evidence)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), layout="constrained")
    green = "#17766e"
    orange = "#ba512f"
    blue = "#315b82"
    gray = "#657680"
    ax = axes[0, 0]

    def dots(ax, sets, names):
        y = np.arange(len(sets))[::-1]
        for j, (k, color, label) in enumerate(
            (
                ("danger_answer", orange, "健康 2" if zh else "Health 2"),
                ("healthy_answer", green, "健康 90/300" if zh else "Health 90/300"),
            )
        ):
            for idx, data in enumerate(sets):
                v = data[k]
                mean = v["mean"] * 100
                lo, hi = np.array(v["ci95"]) * 100
                ax.errorbar(
                    mean,
                    y[idx] + (0.1 if j else -0.1),
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    capsize=2,
                    color=color,
                    label=label if idx == 0 else None,
                )
        ax.set_yticks(y, names)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color="#dedede", lw=0.5)
        ax.set_xlabel("回答概率（%）" if zh else "Answer probability (%)")
        ax.legend(fontsize=8, loc="lower right")

    v = evidence["population"]["absolute"]
    dots(
        ax,
        [
            v[k]
            for k in (
                "frozen",
                "reference5",
                "linear",
                "nonlinear",
                "prompt_lora",
                "maintain",
            )
        ],
        [
            "冻结",
            "PR #5 输出适配器",
            "末层内部 LoRA",
            "末层非线性适配器",
            "只在提示里写健康",
            "只奖励获得保护",
        ]
        if zh
        else [
            "Frozen",
            "PR #5 head",
            "Internal LoRA",
            "Nonlinear adapter",
            "Prompt-only LoRA",
            "Maintenance credit",
        ],
    )
    ax.set_title("哪些条件产生身体区分" if zh else "Body-dependent choice", loc="left")
    ax = axes[0, 1]
    v = evidence["native"]["absolute"]
    dots(
        ax,
        [
            v[k]
            for k in (
                "frozen",
                "text",
                "sensor",
                "shuffle",
                "carrier_exchanged",
                "carrier_removed",
            )
        ],
        [
            "冻结",
            "原权重＋健康提示",
            "原权重＋固定身体信号",
            "打乱身体后果",
            "交换固定信号",
            "移除固定信号",
        ]
        if zh
        else [
            "Frozen",
            "Trained norm + text",
            "Trained norm + sensor",
            "Shuffled outcomes",
            "Exchanged sensor",
            "Removed sensor",
        ],
    )
    ax.set_title(
        "只训练原有的 2,560 个参数" if zh else "Only 2,560 native parameters",
        loc="left",
    )
    ax = axes[1, 0]
    cr = read(a.credit)["stats"]
    order = ["fixed", "scalar", "scalar3", "predict", "predict1", "fisher"]
    names = (
        [
            "固定起点",
            "普通梯度",
            "三倍梯度",
            "预测后果 ×3",
            "预测后果 ×1",
            "Fisher 更新",
        ]
        if zh
        else [
            "Fixed",
            "Gradient",
            "3× gradient",
            "Prediction ×3",
            "Prediction ×1",
            "Fisher",
        ]
    )
    y = np.arange(len(order))[::-1]
    for i, k in enumerate(order):
        v = cr["0_" + k + "_style"]
        mean = 100 * v["mean"]
        lo, hi = np.array(v["ci95"]) * 100
        ax.errorbar(
            mean, y[i], xerr=[[mean - lo], [hi - mean]], fmt="o", color=blue, capsize=2
        )
    ax.set_yticks(y, names)
    ax.set_xlim(0, 65)
    ax.grid(axis="x", color="#dedede", lw=0.5)
    ax.set_title(
        "单生学习后，对陌生问法选对求助表达"
        if zh
        else "Personal learning on new wording",
        loc="left",
    )
    ax.set_xlabel(
        "健康时选中伙伴偏好表达的概率（%）" if zh else "Preferred healthy answer (%)"
    )
    ax = axes[1, 1]
    if a.routing:
        rr = read(a.routing)["stats"]
        ys = [2.5, 2.0, 1.0, 0.5]
        for j, k in enumerate(
            (
                "opposite_style_live",
                "opposite_style_fixed_tape",
                "same_style_live",
                "same_style_fixed_tape",
            )
        ):
            s = rr[k]
            m = 100 * s["mean"]
            lo, hi = np.array(s["ci95"]) * 100
            ax.errorbar(
                m,
                ys[j],
                xerr=[[m - lo], [hi - m]],
                color=green if j < 2 else gray,
                fmt="o",
                capsize=3,
            )
        ax.set_yticks(
            ys,
            [
                "偏好相反 · 闭环",
                "偏好相反 · 同经历",
                "偏好相同 · 闭环",
                "偏好相同 · 同经历",
            ]
            if zh
            else [
                "Opposite · live",
                "Opposite · same tape",
                "Same · live",
                "Same · same tape",
            ],
        )
        ax.axvline(0, color="#999", lw=1)
        ax.grid(axis="x", color="#dedede", lw=0.5)
        ax.set_title(
            "两个同容量存储槽，按身份或按布局分配"
            if zh
            else "Partner routing vs matched-capacity nuisance",
            loc="left",
        )
        ax.set_xlabel(
            "按真实身份分路的增益（百分点）" if zh else "True identity advantage (pp)"
        )
    else:
        ax.axis("off")
    fig.savefig(a.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    if a.depth:
        depth = read(a.depth)
        if "placement" not in depth:
            return
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), layout="constrained")
        ax = axes[0]
        x = np.arange(3)
        v = depth["placement"]["stats"]
        for name, c, off in (
            ("delta_danger", orange, -0.11),
            ("delta_healthy", green, 0.11),
        ):
            for j, site in enumerate((3, 15, 31)):
                s = v[f"d{site}-r{site}-real-x1"][name]
                m = 100 * s["mean"]
                lo, hi = np.array(s["ci95"]) * 100
                lab = (
                    ("致死回答减少" if name == "delta_danger" else "健康回答变化")
                    if zh
                    else name
                )
                ax.errorbar(
                    x[j] + off,
                    m,
                    yerr=[[m - lo], [hi - m]],
                    color=c,
                    fmt="o",
                    capsize=3,
                    label=lab if j == 0 else None,
                )
        ax.axhline(0, lw=0.8, color="#999")
        ax.set_xticks(x, ["3", "15", "31"])
        ax.set_xlabel("真实注入的解码层（从 0 计）" if zh else "Actual injection block")
        ax.set_ylabel(
            "相较同轮冻结模型（百分点）" if zh else "Difference from frozen (pp)"
        )
        ax.legend(fontsize=8)
        ax.set_title(
            "整模型执行，12 个独立种子" if zh else "Full-model execution, 12 seeds",
            loc="left",
        )
        mat = np.array(
            [
                [
                    v[f"d{donor}-r{recv}-real-x1"]["delta_danger"]["mean"] * 100
                    for recv in (3, 15, 31)
                ]
                for donor in (3, 15, 31)
            ]
        )
        # Diagonal for the transplant panel uses its four donors only, like all off-diagonal cells.
        for i, l in enumerate((3, 15, 31)):
            mat[i, i] = 100 * np.mean(
                [
                    depth["placement"]["per_world"][f"s{s}-m{m}"][f"d{l}-r{l}-real-x1"][
                        "delta_danger"
                    ]
                    for s in range(65001, 65005)
                    for m in (0, 1)
                ]
            )
        lim = max(1, np.max(np.abs(mat)))
        im = axes[1].imshow(mat, vmin=-lim, vmax=lim, cmap="BrBG")
        for i in range(3):
            for j in range(3):
                axes[1].text(
                    j,
                    i,
                    f"{mat[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    color="white" if abs(mat[i, j]) > 0.62 * lim else "#17242c",
                )
        axes[1].set_xticks(x, ["3", "15", "31"])
        axes[1].set_yticks(x, ["3", "15", "31"])
        axes[1].set_xlabel("实际接入层" if zh else "Receiving block")
        axes[1].set_ylabel("训练位置" if zh else "Donor block")
        axes[1].set_title(
            "原样移植后，致死回答减少多少点"
            if zh
            else "Unchanged transplant, fatal reduction (pp)",
            loc="left",
        )
        fig.colorbar(im, ax=axes[1], shrink=0.75)
        fig.savefig(a.output.with_name("placement.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
