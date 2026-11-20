"""Standalone scientific figures from verified records, without GPU inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PRIMARY=['frozen','social','body','body_blind']
COLORS=dict(zip(PRIMARY,['#69747c','#bc8125','#216bad','#984f92']))
LABELS=dict(zip(PRIMARY,['Frozen','Social feedback','Body feedback','Body feedback\n(health hidden)']))


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def seed_values(rows,arm,field):
    chosen=[r for r in rows if r['arm']==arm]
    return [np.mean([r[field] for r in chosen if r['seed']==seed]) for seed in sorted({r['seed'] for r in chosen})]


def create(data,roots,output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'figure.facecolor':'white','axes.titleweight':'bold'})
    rows=data['results'];pairs=data['reward_prefix_interventions']['pairs']
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    for ax,optimizer in zip(axes[0],('adam','sgd')):
        for signal,color,style,label in [('body_high','#216bad','-','Body reward, health 300'),
                ('body_low','#b84040','-','Body reward, health 3'),
                ('social_high','#6b8141','--','Social reward, both healths'),('zero','#777777',':','Zero reward')]:
            case=next(r for r in data['smoke']['reward_interventions'] if r['optimizer']==optimizer and r['signal']==signal)
            ax.plot([0,1,4,16],[0]+[100*c['selected_change'] for c in case['checkpoints']],
                marker='o',label=label,color=color,linestyle=style)
        ax.axhline(0,color='#cccccc',linewidth=.8);ax.set_xticks([0,1,4,16])
        ax.set(xlabel='Optimizer steps (forced diagnostic action)',ylabel='Selected-action probability change (pp)',
            title=f'{optimizer.upper()}: same input, different realised injury')
    axes[0,0].legend(fontsize=8)
    positions=np.arange(len(pairs));tv=[100*r['turn9_tv'] for r in pairs]
    axes[1,0].bar(positions,tv,color='#216bad',width=.65)
    axes[1,0].set_xticks(positions,[f"{r['seed']}\nm{r['mapping']}" for r in pairs],fontsize=8)
    axes[1,0].set(title='First decision after the reward update',
        ylabel='Policy total variation at turn 9 (%)',xlabel='Same first 8 transitions and turn-9 input')
    divergences=[r['first_actual_action_divergence']['turn'] if r['first_actual_action_divergence'] else np.nan for r in pairs]
    axes[1,1].scatter(positions,divergences,color='#216bad',s=40)
    axes[1,1].set_xticks(positions,[f"{r['seed']}\nm{r['mapping']}" for r in pairs],fontsize=8)
    axes[1,1].set(ylim=(0,74),ylabel='First different actual action (turn)',
        title='First actual choice that differed across rewards',xlabel='Both agents were alive at the time of choice')
    save(fig,output,'01-feedback-to-action')

    traces={}
    for root in roots:
        for path in (root/'output').glob('*/result.json'):
            r=json.loads(path.read_bytes())
            if r['arm'] in PRIMARY:traces[r['id']]=r
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    for arm in PRIMARY:
        selected=[r for r in traces.values() if r['arm']==arm];health=[];answers=[]
        for r in selected:
            h=[350.]+[x['health_after'] for x in r['trace']]
            h+=([h[-1]]*(73-len(h)))
            a=[0]+list(np.cumsum([x['action']>=2 for x in r['trace']]))
            a+=([a[-1]]*(73-len(a)))
            health.append(h);answers.append(a)
        axes[0,0].plot(range(73),np.mean(health,axis=0),color=COLORS[arm],label=LABELS[arm].replace('\n',' '))
        axes[0,1].plot(range(73),np.mean(answers,axis=0),color=COLORS[arm],label=LABELS[arm].replace('\n',' '))
    for ax in axes[0]:
        ax.axvline(20,color='#dddddd',linestyle=':');ax.axvline(50,color='#dddddd',linestyle=':')
        ax.set_xlabel('Encounter (partners switch after 20 and 50)')
    axes[0,0].set(title='Body health by encounter',ylabel='Mean health (zero after death)');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Answers given over the lifetime',ylabel='Mean cumulative answers')
    for i,arm in enumerate(PRIMARY):
        reward=seed_values(rows,arm,'body_reward_total');unsafe=np.array(seed_values(rows,arm,'final_low_fatal'))*100
        for ax,values in [(axes[1,0],reward),(axes[1,1],unsafe)]:
            ax.bar(i,float(np.mean(values)),color=COLORS[arm],alpha=.65,width=.65)
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color=COLORS[arm],edgecolors='white',s=32,zorder=3)
    for ax in axes[1]:ax.set_xticks(range(4),[LABELS[arm] for arm in PRIMARY],fontsize=9)
    axes[1,0].set(title='Reward in the simulated lives',ylabel='Total engineering body reward')
    initial=[r['probes']['0']['metrics']['joint']['low_health_fatal_answer_probability'] for r in rows if r['arm']=='frozen']
    axes[1,1].axhline(100*np.mean(initial),color='#69747c',linestyle='--',linewidth=1,label='Frozen initial')
    axes[1,1].set(title='Analytical final probe at health 2',ylabel='Probability of an immediately fatal answer (%)',ylim=(0,100))
    axes[1,1].legend(fontsize=8)
    fig.supxlabel('Eight paired environments from four seeds. Dots show means by seed.',fontsize=9)
    save(fig,output,'02-primary-outcomes')

    contrasts=['body-minus-social','body_blind-minus-body','myopic-minus-body',
        'shuffled-minus-body','tail-minus-body','more_steps-minus-body']
    labels=['Body reward vs social','Hide current health','One-step return','Shuffle advantages','Tail LoRA','Double update epochs']
    fig,axes=plt.subplots(1,3,figsize=(13.2,5.4),layout='constrained')
    fields=['body_reward_total','final_low_fatal','final_style_margin']
    for ax,field,title,mult in zip(axes,fields,['Actual life reward difference','Fatal-answer probability difference',
        'Correct social-style margin difference'],[1,100,100]):
        for i,key in enumerate(contrasts):
            effect=data['paired_effects'][key]
            env=np.array([r[field] for r in effect['rows']])*mult
            seed=np.array([r[field] for r in effect['seed_means']])*mult
            ax.scatter(env,np.full(len(env),i),s=16,color='#aaaaaa',alpha=.5)
            ax.scatter(seed,i+np.linspace(-.09,.09,len(seed)),s=32,color='#216bad')
            ax.scatter(np.mean(seed),i,marker='D',s=40,color='#153f60',zorder=4)
        ax.axvline(0,color='#aaaaaa',linestyle='--',linewidth=1)
        ax.set(title=title,yticks=range(6));ax.invert_yaxis();ax.set_yticklabels(labels if ax is axes[0] else ['']*6)
        ax.set_xlabel('Difference (reward)' if field=='body_reward_total' else 'Difference (percentage points)')
    fig.supxlabel('Gray dots show environments. Blue dots show seed means. Diamonds show the mean. Four seeds for the first two rows, two for the rest.',fontsize=9)
    save(fig,output,'03-paired-factors')

    if 'numerics' in data:
        result = data['numerics']
        steps = np.array([row['parameter_step_l2'] for row in result['scales']])
        differences = np.array([row['difference_over_autograd'] for row in result['scales']])
        fig, ax = plt.subplots(figsize=(7.6, 4.4), layout='constrained')
        ax.axhline(0, color='#cccccc', linewidth=.8)
        ax.axhline(1, color='#69747c', linestyle='--', linewidth=1, label='Agreement with autograd')
        ax.plot(steps, differences, color='#216bad', marker='o', label='BF16 forward, FP32 LoRA')
        ax.set(xscale='log', yscale='symlog', xlabel='LoRA perturbation size (parameter L2 norm)',
               ylabel='Central finite difference / autograd derivative',
               title='Finite-scale gradient check on the same Qwen input')
        ax.set_ylim(min(-.08, differences.min()-.05), max(2., differences.max()*1.45))
        ax.legend(fontsize=9)
        save(fig, output, '04-precision-diagnostic')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--root',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();create(json.loads(args.analysis.read_bytes()),args.root,args.output)
