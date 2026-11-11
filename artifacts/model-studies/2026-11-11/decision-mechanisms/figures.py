"""Plot held-out seed results as points, with paired conditions connected."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_figure(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout='constrained')
    cases = data['matched_feedback']
    names = ['social', 'light', 'shuffled', 'heavy']
    labels = ['Social only', 'Light damage', 'Shuffled damage', 'Heavy damage']
    for case in cases:
        color = f"C{case['seed']-61151}"
        for axis, metric in zip(axes, ('low_fatal_answer', 'high_minus_low_answer')):
            values = [case['conditions'][n]['end'][metric]*100 for n in names]
            axis.plot(range(4), values, 'o-', color=color, alpha=.8, label=case['seed'])
        rank = case['hidden_damage_readout']
        axes[2].plot(range(3), [rank[n]*100 for n in ('3','15','31')], 'o-', color=color, alpha=.8)
    for axis in axes[:2]:
        axis.set_xticks(range(4), labels, rotation=20, ha='right')
    axes[0].set(title='Fatal answer at health 2', ylabel='Action probability (%)')
    axes[0].legend(title='Environment seed', fontsize=8, frameon=False)
    axes[1].set(title='Does high health increase answering?', ylabel='Health 300 minus health 2 (pp)')
    axes[1].axhline(0, color='grey', linestyle=':')
    axes[2].set(title='Damage history readout with health hidden', ylabel='Correctly ranked matched pairs (%)',
                ylim=(-3, 103))
    axes[2].set_xticks(range(3), ['Layer 3', 'Layer 15', 'Layer 31'])
    axes[2].axhline(50, color='grey', linestyle=':')
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix('.pdf'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('summary', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    make_figure(json.loads(a.summary.read_bytes()), a.output)
