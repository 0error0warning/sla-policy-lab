"""Matched seed differences and the held-out layer-3 readout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(data, destination):
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for index, stage in enumerate(('original','new')):
        contrast = data['comparisons'][stage]
        offset = np.linspace(-.12,.12,4)
        for ax, metric in zip(axes[0],('low_answer','high_answer')):
            points = np.array(contrast[metric]['values'])*100
            ax.scatter(index+offset,points,label=stage,color=f'C{index}',s=40)
            ax.scatter(index,points.mean(),marker='D',s=55,color='black')
            ax.axhline(0,color='grey',linestyle=':')
            ax.set_xticks([0,1],['Original four seeds','New four seeds'])
            ax.set_ylabel('Independent baseline minus zero (pp)')
    axes[0,0].set_title('Fatal answering at health 2')
    axes[0,1].set_title('Answering at health 300')
    for index,stage in enumerate(('original','new')):
        points = np.array(data['comparisons'][stage]['interaction']['values'])*100
        axes[1,0].scatter(index+np.linspace(-.12,.12,4),points,color=f'C{index}',s=40)
        axes[1,0].scatter(index,points.mean(),marker='D',s=55,color='black')
    axes[1,0].axhline(0,color='grey',linestyle=':')
    axes[1,0].set_xticks([0,1],['Original four seeds','New four seeds'])
    axes[1,0].set(title='Change in health-conditioned answering',
                  ylabel='Change in high minus low answer gap (pp)')
    for index,method in enumerate(('zero','independent')):
        values = [100*r['paired_rank'] for r in data['features'] if r['method']==method]
        axes[1,1].scatter(index+np.linspace(-.12,.12,4),values,s=40,color=f'C{index}')
    axes[1,1].axhline(50,color='grey',linestyle=':')
    axes[1,1].axhline(75,color='grey',linestyle='--')
    axes[1,1].set_xticks([0,1],['Zero baseline','Independent baseline'])
    axes[1,1].set(title='Layer 3 readout on new seeds',ylabel='Matched damage pairs ranked correctly (%)',
                  ylim=(-3,103))
    fig.savefig(destination,dpi=170)
    fig.savefig(destination.with_suffix('.pdf'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('summary',type=Path)
    p.add_argument('destination',type=Path)
    a=p.parse_args()
    draw(json.loads(a.summary.read_bytes()),a.destination)
