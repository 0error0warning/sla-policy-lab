"""Plot the paired measurements without treating prompts as independent lives."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "lora":"LoRA, layers 0–3", "ia3":"IA³, LR 0.0001", "ia3_high":"IA³, LR 0.001",
    "edit":"Output weight edit", "edit_light":"Output edit + weak anchors",
    "edit_anchored":"Output edit + strong anchors", "fast":"Partner-routed fast weights",
    "frozen_history":"Frozen + observed counts", "frozen":"Frozen", "beta":"Per-partner Beta",
    "hierarchical_bins":"Hierarchical Beta, style bins",
}
COLORS = {
    "lora":"#3677aa", "ia3":"#8876a3", "ia3_high":"#725695", "edit":"#b55a38",
    "edit_light":"#c97342", "edit_anchored":"#d48758",
    "fast":"#208677", "frozen_history":"#83734f", "frozen":"#999a9e", "beta":"#8b958c",
    "hierarchical_bins":"#739478",
}


def plot(data,out):
    methods = [m for m in LABELS if m in data["metrics"]]
    fig,axes = plt.subplots(1,4,figsize=(14,6.7),layout="constrained",sharey=True)
    panels = [
        ("New-wording probability error",lambda b:b["focus_heldout_mae"]),
        ("History effect on current partner",lambda b:b["by_stage_effect"]["60"][b["focus"]]["directional"]),
        ("History effect on unseen P86",lambda b:b["by_stage_effect"]["60"]["P86"]["directional"]),
        ("Old partner drift after learning next",lambda b:b["anchor_drift"]),
    ]
    for ax,(title,measure) in zip(axes,panels,strict=True):
        for i,m in enumerate(methods):
            x = [100*measure(block) for block in data["metrics"][m].values()]
            ax.barh(i,mean(x),color=COLORS[m],height=.65,alpha=.8)
            ax.scatter(x,[i-.10,i+.10],s=14,color="#222222",zorder=3)
        ax.axvline(0,color="#6d6d6d",linewidth=.65)
        ax.grid(axis="x",alpha=.20)
        ax.set_title(title,fontsize=10,loc="left")
        ax.set_xlabel("Percentage points")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(list(range(len(methods))),[LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    fig.suptitle("Paired experiences under alternative parameter updates",fontsize=15)
    fig.supxlabel("Bars average two matched life pairs. Dots show each pair. Error is relative to the per-partner Beta teacher.",fontsize=9)
    fig.savefig(out,dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    plot(json.loads(args.scored.read_bytes()),args.output)
