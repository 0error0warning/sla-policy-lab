"""Create a standalone figure from verified tiny-fitting results."""

import argparse
import json
from pathlib import Path


def plot(scored, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout="constrained")
    colors={0:"#2563a6",1:"#b35326"}
    for result in scored["jobs"].values():
        points=sorted((int(k),v["train"]["mean_kl"]) for k,v in result["checkpoints"].items())
        x,y=zip(*points)
        if result["task"]=="policy":
            label=f"bank {result['mapping']}, lr={result['learning_rate']:g}"
            axes[0].plot(x,y,marker="o",color=colors[result["mapping"]],
                         linestyle="-" if result["learning_rate"]==.001 else "--",label=label)
        else:
            axes[1].plot(x,y,marker="o",color=colors[result["mapping"]],
                linestyle="-" if result["task"]=="social" else "--",
                label=f"bank {result['mapping']}, {result['task']}")
    for ax in axes[:2]:
        ax.set_xlabel("Optimizer steps")
        ax.set_ylabel("Mean KL to training teacher (lower is better)")
        ax.set_xticks([0,16,48,96])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y",alpha=.16)
        ax.legend(fontsize=8,frameon=False)
        ax.axhline(.02,color="#555555",linewidth=.8,linestyle=":")
    axes[0].set_title("A. Combined policy: fixed 16 prompts",loc="left",fontsize=11)
    axes[1].set_title("B. Split tasks: lower learning rate",loc="left",fontsize=11)
    bodies=[r for r in scored["jobs"].values() if r["task"]=="body"]
    groups=[]
    for step,label in (("0","Frozen start"),("96","Trained end")):
        selected=sum(r["checkpoints"][step]["provided_returns"]["teacher_action_correct"] for r in bodies)
        selection_n=sum(r["checkpoints"][step]["provided_returns"]["n"] for r in bodies)
        computed=sum(r["checkpoints"][step]["numeric_returns"]["all_four_within_tolerance"] for r in bodies)
        compute_n=sum(r["checkpoints"][step]["numeric_returns"]["n"] for r in bodies)
        groups.append((label,selected,selection_n,computed,compute_n))
    x=np.arange(2)
    for i,(field,denom,color,label) in enumerate(((1,2,"#2563a6","Choose supplied returns"),
                                                 (3,4,"#b35326","Compute all four returns"))):
        pos=x+(i-.5)*.36
        vals=[100*g[field]/g[denom] for g in groups]
        bars=axes[2].bar(pos,vals,width=.33,color=color,label=label)
        for bar,g in zip(bars,groups):
            axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                         f"{g[field]}/{g[denom]}",ha="center",fontsize=9)
    axes[2].set_xticks(x,[g[0] for g in groups])
    axes[2].set_ylim(0,119)
    axes[2].set_yticks([0,25,50,75,100])
    axes[2].set_ylabel("Correct (%)")
    axes[2].set_title("C. Selection vs. arithmetic probes",loc="left",fontsize=11)
    axes[2].legend(fontsize=8,frameon=False,loc="upper right")
    axes[2].grid(axis="y",alpha=.16)
    fig.suptitle("Crossed tiny-example fitting assay | Qwen3.5-4B, last-four-MLP LoRA",fontsize=14)
    fig.supxlabel("Descriptive engineering checks: one initialization; two reused feedback banks share one source seed. No independent efficacy test.",fontsize=9)
    fig.savefig(output,dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    plot(json.loads(args.scored.read_bytes()),args.output)
