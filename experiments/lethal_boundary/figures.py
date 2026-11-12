"""Standalone research figures and human-readable held-out point estimates."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from experiments.lethal_boundary.evaluate import MI, bootstrap


def exported_points(values, h, q, file):
    n = len(values)
    indices = bootstrap(n)
    weights = np.stack([(indices == j).mean(1) for j in range(n)], axis=1)
    keys = (
        "answer",
        "intrinsic_fatal_answer",
        "answer_and_die",
        "total_death",
        "safe_task",
    )
    means = values.mean(0)
    bounds = {}
    for qi in range(len(q)):
        for k in keys:
            across = weights @ values[:, qi, :, MI[k]]
            bounds[qi, k] = np.quantile(across, [0.025, 0.975], axis=0)
    with file.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "current_health",
                "preferred_style_maintenance_q",
                *[s for k in keys for s in (k, k + "_low95", k + "_high95")],
            ]
        )
        for qi, rate in enumerate(q):
            for hi, value in enumerate(h):
                w.writerow(
                    [
                        value,
                        rate,
                        *[
                            round(float(v), 8)
                            for k in keys
                            for v in (means[qi, hi, MI[k]], *bounds[qi, k][:, hi])
                        ],
                    ]
                )
    return weights


def run(analysis, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    path = Path("C:/Windows/Fonts/msyh.ttc")
    zh = path.is_file()
    if zh:
        font_manager.fontManager.addfont(str(path))
        mpl.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(path)
        ).get_name()

    def label(cn, en):
        return cn if zh else en

    mpl.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    with np.load(analysis / "seed-surfaces.npz") as z:
        h, q = z["health"], z["q"]
        data = {
            k: z[k]
            for k in (
                "record-D",
                "record-R",
                "record-M",
                "record-Mshift",
                "record-frozen",
                "plain-M",
            )
        }
    with np.load(analysis / "physical.npz") as z:
        physical = z["conditional_death"]
    weights = exported_points(
        data["record-M"], h, q, out / "mixed-policy-by-health-and-partner.csv"
    )
    colors = {
        "record-D": "#91541f",
        "record-R": "#366c9d",
        "record-M": "#137863",
        "record-Mshift": "#845597",
        "record-frozen": "#6b7480",
        "plain-M": "#947853",
    }
    names = {
        "record-D": label("只见不可救", "Doomed experiences"),
        "record-R": label("只见可救", "Rescuable experiences"),
        "record-M": label("两侧混合", "Mixed experiences"),
        "record-Mshift": label("训练档平移 +0.55", "Shifted training values"),
        "record-frozen": label("冻结 Qwen", "Frozen Qwen"),
        "plain-M": label("混合，无伙伴记录", "Mixed, no partner record"),
    }

    def plot(ax, x, arr, color, title, linestyle="-", band=True):
        ax.plot(x, 100 * arr.mean(0), color=color, label=title, lw=2, ls=linestyle)
        if band:
            low, high = np.quantile(weights @ arr, [0.025, 0.975], axis=0) * 100
            ax.fill_between(x, low, high, color=color, alpha=0.12, lw=0)

    def boundary(ax):
        ax.axvspan(2.4414549, 4.520616, alpha=0.11, color="#ad876b", zorder=0)
        ax.set_xlabel(label("行动前健康", "Health before action"))

    def polish(ax, title, ylabel):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.17)
        ax.set_ylim(-2, 102)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), layout="constrained")
    ax = axes[0, 0]
    short = h < 15
    ax.plot(
        h[short],
        100 * physical[short, 2, 1],
        color="#137863",
        lw=2.5,
        label=label("得到维护", "Maintained"),
    )
    ax.plot(
        h[short],
        100 * physical[short, 2, 0],
        color="#952f30",
        lw=2.5,
        label=label("未得到维护", "Unmaintained"),
    )
    boundary(ax)
    polish(
        ax,
        label("同一次回答，维护能否保住身体", "Conditional mortality of an answer"),
        label("回答后的死亡率（%）", "Death after answering (%)"),
    )
    ax.legend()
    ax = axes[0, 1]
    short = h < 8
    qi = 6
    for k in ("record-R", "record-D", "record-M"):
        plot(ax, h[short], data[k][:, qi, short, MI["answer"]], colors[k], names[k])
    boundary(ax)
    polish(
        ax,
        label("未见健康点的选择，维护率 0.75", "Unseen healths at q = 0.75"),
        label("回答概率（%）", "Answer probability (%)"),
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for bounds, color, name in (
        (h <= 2.44, "#a94340", label("维护也救不了", "Answer always fatal")),
        (
            (h >= 4.521) & (h <= 6.5),
            "#137863",
            label("得到维护能存活", "Rescuable with care"),
        ),
    ):
        plot(
            ax,
            q,
            data["record-M"][:, :, :, MI["answer"]][:, :, bounds].mean(2),
            color,
            name,
        )
    polish(
        ax,
        label("混合组如何使用伙伴记录", "Mixed policy with a partner record"),
        label("回答概率（%）", "Answer probability (%)"),
    )
    ax.set_xlabel(
        label(
            "伙伴对偏好表达的实际维护率",
            "True maintenance probability for favored style",
        )
    )
    ax.legend()
    ax = axes[1, 1]
    healthy = np.isin(h, [90, 300])
    for k in ("record-M", "plain-M", "record-frozen"):
        plot(
            ax,
            q,
            data[k][:, :, :, MI["safe_task"]][:, :, healthy].mean(2),
            colors[k],
            names[k],
            linestyle="--" if k == "plain-M" else "-",
        )
    polish(
        ax,
        label("健康 90/300 时完成任务的代价", "Safe tasks at health 90/300"),
        label("回答且存活（%）", "Answer and survive (%)"),
    )
    ax.set_xlabel(
        label(
            "伙伴对偏好表达的实际维护率",
            "True maintenance probability for favored style",
        )
    )
    ax.legend(fontsize=9)
    fig.savefig(out / "boundary-transfer.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 4.2), layout="constrained", sharey=True
    )
    choose = h < 15
    for ax, (key, cn, en) in zip(
        axes,
        [
            ("answer_and_die", "回答且在当轮死亡", "Answer and die"),
            ("total_death", "当轮死亡，含拒绝", "Total mortality"),
            ("safe_task", "回答且存活", "Answer and survive"),
        ],
        strict=True,
    ):
        grid = data["record-M"].mean(0)[:, choose, MI[key]] * 100
        im = ax.pcolormesh(
            h[choose], q, grid, shading="nearest", vmin=0, vmax=100, cmap="viridis"
        )
        for edge in (2.4414549, 4.520616):
            ax.axvline(edge, color="white", lw=0.85, ls="--", alpha=0.8)
        ax.set_title(label(cn, en), loc="left", fontweight="bold")
        ax.set_xlabel(label("行动前健康", "Health before action"))
        ax.set_yticks(q[::2])
    axes[0].set_ylabel(label("实际维护率", "True maintenance probability"))
    fig.colorbar(im, ax=axes, label=label("概率（%）", "Probability (%)"), shrink=0.77)
    fig.savefig(out / "pointwise-risks.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def export_transition(study, out):
    with np.load(study / "seed-surfaces.npz") as z:
        h, q = z["health"], z["q"]
        fixed = z["fixed"]
        border = z["border"]
    keys = ("answer_and_die", "total_death", "safe_task")
    rows = np.flatnonzero((h > 2.44) & (h < 4.521))
    sample = bootstrap(len(fixed))
    weights = np.stack([(sample == j).mean(1) for j in range(len(fixed))], axis=1)
    with (out / "transition-pointwise.csv").open(
        "w", encoding="utf8", newline=""
    ) as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "health",
                "q",
                *[
                    k + "_" + suffix
                    for k in keys
                    for suffix in (
                        "fixed",
                        "border",
                        "difference",
                        "difference_low95",
                        "difference_high95",
                    )
                ],
            ]
        )
        for i, hh in enumerate(h):
            if i not in rows:
                continue
            for qi, prob in enumerate(q):
                row = [hh, prob]
                for k in keys:
                    a = fixed[:, qi, i, MI[k]]
                    b = border[:, qi, i, MI[k]]
                    ci = np.quantile(weights @ (b - a), [0.025, 0.975])
                    row.extend(
                        round(float(t), 8)
                        for t in (a.mean(), b.mean(), (b - a).mean(), *ci)
                    )
                w.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--transition", type=Path)
    a = p.parse_args()
    run(a.analysis, a.output)
    if a.transition:
        export_transition(a.transition, a.output)
