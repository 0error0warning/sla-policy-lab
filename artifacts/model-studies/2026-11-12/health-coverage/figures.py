"""Seed-level paired coverage effects and the actual health distribution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(summary, root, target):
    fig, axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    seeds=sorted(summary['by_seed'])
    for i,seed in enumerate(seeds):
        row=summary['by_seed'][seed]
        axes[0,0].plot([0,1],[100*row[f'{v}_danger_death'] for v in ('visible','hidden')],
                       'o-',color=f'C{i}',label=seed)
        axes[0,1].plot([0,1],[100*row[f'{v}_normal_answer'] for v in ('visible','hidden')],
                       'o-',color=f'C{i}')
    for ax in axes[0]:
        ax.axhline(0,color='grey',linestyle=':')
        ax.set_xticks([0,1],['Health visible','Health hidden'])
        ax.set_ylabel('Low-health replay minus original (pp)')
    axes[0,0].set_title('Change in death risk at health 6 and 10')
    axes[0,0].legend(title='Independent seed',fontsize=8,frameon=False)
    axes[0,1].set_title('Change in answering at health 300')
    for regime,color,label in [('original','#61717d','Original health range'),
                               ('covered','#b66c23','Low-health coverage')]:
        for seed in seeds:
            case=json.loads((root/f'{seed}-{regime}.json').read_bytes())
            axes[1,0].plot(range(1,9),[r['health_before'] for r in case['buffer']],
                           color=color,alpha=.65,label=label if seed==seeds[0] else None)
        values=[]
        for health in (2,6,10,15,90,300):
            values.append(100*np.mean([summary['cases'][f'{seed}-{regime}']['doses']['24']
                                     [f'visible-h{health}']['death_risk'] for seed in seeds]))
        axes[1,1].plot(range(6),values,'o-',color=color,label=label)
    axes[1,0].axhline(15,color='grey',linestyle=':')
    axes[1,0].set(title='Actual health before training actions',xlabel='Replay encounter',ylabel='Health')
    axes[1,0].legend(fontsize=8,frameon=False)
    axes[1,1].set_xticks(range(6),['2','6','10','15','90','300'])
    axes[1,1].set(title='Expected one-step death, health visible',
                  xlabel='Evaluation health',ylabel='Mean death probability (%)')
    axes[1,1].legend(fontsize=8,frameon=False)
    fig.savefig(target,dpi=170)
    fig.savefig(target.with_suffix('.pdf'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    draw(json.loads(a.summary.read_bytes()),a.root,a.output)
