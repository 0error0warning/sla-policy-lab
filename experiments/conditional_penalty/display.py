"""Three decision outcomes for each birth, paired within environmental seed."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(root, dest):
    with (root / "lives.csv").open(encoding="utf8", newline="") as f:
        data = list(csv.DictReader(f))
    summary = json.loads((root / "live.json").read_bytes())
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), layout="constrained")
    specs = (
        (
            "First fatal request",
            "P(fatal answer), conditional minus birth (pp)",
            "gate_vs_birth",
        ),
        (
            "Retention at first fatal request",
            "P(fatal answer), conditional minus 2× (pp)",
            "gate_vs_two",
        ),
        (
            "During actual healthy encounters",
            "Task rate, conditional minus frozen (pp)",
            "healthy",
        ),
    )
    for ax, (title, label, key) in zip(axes, specs, strict=True):
        for j, (origin, color) in enumerate(
            (("blank", "#42718f"), ("population", "#ad633b"))
        ):
            group = sorted(
                (r for r in data if r["birth"] == origin and r["rule"] == "gate_true"),
                key=lambda r: int(r["seed"]),
            )
            controls = sorted(
                (r for r in data if r["birth"] == origin and r["rule"] == "frozen"),
                key=lambda r: int(r["seed"]),
            )
            if key == "healthy":
                vals = [
                    float(a["healthy_rate"]) - float(b["healthy_rate"])
                    for a, b in zip(group, controls, strict=True)
                ]
                est = summary[origin]["free"]["frozen"]["healthy_rate_delta"]
            else:
                role = "birth" if key == "gate_vs_birth" else "extra"
                vals = [
                    float(a["fatal_current"]) - float(a["fatal_" + role]) for a in group
                ]
                est = summary[origin]["matched_gate_tape"][key]["fatal"]
            ax.scatter(
                j + np.linspace(-0.1, 0.1, len(vals)),
                100 * np.asarray(vals),
                s=17,
                color=color,
                alpha=0.56,
            )
            ax.plot([j, j], 100 * np.array(est["ci95"]), c="black", lw=2)
            ax.scatter(
                j,
                100 * est["mean"],
                c=color,
                marker="D",
                s=36,
                edgecolor="black",
                zorder=4,
            )
        ax.axhline(0, color="#47525a", lw=0.8)
        if key != "gate_vs_birth":
            ax.axhline(-2 if key == "healthy" else 2, color="#73808a", ls=":", lw=0.85)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], ["No body\ntraining", "Population-\ntrained"])
        ax.set_title(title, loc="left", size=10)
        ax.set_ylabel(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Damage feedback doubled only at low health", fontsize=15, weight="bold"
    )
    fig.savefig(dest, dpi=160, metadata={"Software": "SLA finite-life study"})
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis", type=Path)
    p.add_argument("png", type=Path)
    a = p.parse_args()
    draw(a.analysis, a.png)
