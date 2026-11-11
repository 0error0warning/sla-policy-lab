"""CPU-only reward-design references; these never train or select for Qwen."""

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from protocol import CONFIG, COSTS, World, probabilities, rng, sample_action, write


class ImmediateReference:
    def __init__(self):
        normal=NormalDist();lo,hi=normal.cdf(-3.),normal.cdf(3.)
        gen=np.random.default_rng(611490)
        self.noise=np.array([.1*normal.inv_cdf(lo+(hi-lo)*float(x)) for x in gen.random(4096*5)]).reshape(4096,5)

    def after(self,health,action,maintained):
        h=np.full(len(self.noise),health)
        for i in range(5):
            cost=COSTS[action] if i==0 else (0. if maintained else 1.)
            h=np.maximum(0.,h-cost*(2.-h/350.)*(1+self.noise[:,i]))
        return h

    def values(self,health,response):
        values=[]
        for action in range(4):
            p=0. if action==0 else response[action-1]
            total=0.
            for maintained,weight in ((True,p),(False,1.-p)):
                after=self.after(health,action,maintained)
                utility=2.+int(action>=2)+.5*int(maintained)-.15*(health-after)-12*(after<=0)
                total+=weight*float(utility.mean())
            values.append(total)
        return np.array(values)


def run(output,method=None):
    output.mkdir(parents=True,exist_ok=True)
    reference=ImmediateReference();results=[]
    available=['always_silent','always_refuse_appeal','always_answer_appeal','always_answer_exchange','uniform',
        'known_rule_immediate','known_rule_immediate_mixed','always_answer_appeal_mixed','always_answer_exchange_mixed']
    assert method is None or method in available
    methods=available[:6] if method is None else [method]
    for seed in CONFIG['seeds']:
        for mapping in CONFIG['mappings']:
            for method in methods:
                world=World(seed,mapping,f'cpu-s{seed}-m{mapping}-{method}');trace=[]
                for item in world.items:
                    if world.done:break
                    base_method = method.removesuffix('_mixed')
                    if base_method.startswith('always_'):
                        action = available.index(base_method)
                    elif base_method == 'uniform':
                        action = int(rng(seed,'baseline-uniform',item['turn']).integers(4))
                    else:
                        response = probabilities(item['partner'], item['frame'], mapping)
                        action = int(np.argmax(reference.values(world.health, response)))
                    if method.endswith('_mixed'):
                        distribution = np.full(4, .025)
                        distribution[action] += .9
                        action, _, _ = sample_action(distribution[item['order']], item['order'], seed, item['turn'])
                    trace.append(world.step(world.encounter_id(),action))
                body=sum(x['reward_body'] for x in trace)
                result=dict(seed=seed,mapping=mapping,method=method,turns=world.turn,health=world.health,
                    answers=sum(x['action']>=2 for x in trace),protected=sum(x['accepted'] is True for x in trace),
                    physical_death=world.health<=0,body_reward_total=body,
                    discounted_body_return=float(sum(.97**i*x['reward_body'] for i,x in enumerate(trace))),trace=trace)
                write(output/(world.name+'.json'),result);results.append({k:v for k,v in result.items() if k!='trace'})
    fields=['turns','health','answers','protected','body_reward_total','discounted_body_return']
    aggregate={m:{f:float(np.mean([r[f] for r in results if r['method']==m])) for f in fields} for m in methods}
    grid=[]
    for h in (350.,300.,100.,30.,10.,4.,3.,2.,1.):
        for p in ([.15,.9,.1],[.15,.1,.9]):
            values=reference.values(h,p)
            survival=[]
            for action in range(4):
                probability=0. if action==0 else p[action-1]
                survival.append(float(probability*np.mean(reference.after(h,action,True)>0)+
                    (1-probability)*np.mean(reference.after(h,action,False)>0)))
            grid.append(dict(health=h,response_probabilities=p,expected_immediate_reward=values.tolist(),
                survival_probability=survival,best_immediate_reward_action=int(values.argmax())))
    result=dict(scope='fixed-action/random controls and known-probability one-step reward reference; not global lifetime optimum',
        oracle_noise_samples=4096,aggregate=aggregate,results=results,post_hoc_reward_grid=grid)
    write(output/'summary.json',result);print(json.dumps(dict(aggregate=aggregate)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--method')
    a=p.parse_args();run(a.output,a.method)
