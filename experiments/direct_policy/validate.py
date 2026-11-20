"""Independent checks for bodily consequences, returns and PPO reward credit."""

from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
import argparse
import json
from pathlib import Path

import numpy as np

from protocol import (CONFIG, COSTS, World, canonical, comprehension, jobs, noises,
                      policy, probes, returns, reward, schedule, settle, write)


def validate():
    generator=np.random.default_rng(611410)
    # A global conservative certificate, separate from the random Decimal checks.
    # Fragility <= 2 and (1+noise) <= 1.3 at every event.
    live_lower_bounds=[]
    for a,cost in enumerate(COSTS):
        for maintained in ((False,) if a==0 else (False,True)):
            maximum_damage=(cost+(0 if maintained else 4))*2*1.3
            lower=2+int(a>=2)+.5*int(maintained)-.15*maximum_damage
            assert lower>0
            live_lower_bounds.append(dict(action=a,maintained=maintained,nonterminal_reward_lower_bound=lower))
    maximum=0.;events=0
    for _ in range(1200):
        h=float(generator.uniform(.001,350));a=int(generator.integers(4))
        y=bool(generator.integers(2)) if a else None
        noise=generator.uniform(-.3,.3,5).tolist()
        after,steps=settle(h,a,y,noise)
        health=Decimal(str(h));independent=[float(health)]
        for i,eta in enumerate(noise):
            cost=Decimal(str(COSTS[a])) if i==0 else Decimal(0 if y else 1)
            health=max(Decimal(0),health-cost*(Decimal(2)-health/Decimal(350))*(Decimal(1)+Decimal(str(eta))))
            independent.append(float(health))
        maximum=max(maximum,max(abs(a-b) for a,b in zip(steps,independent)));events+=5
        assert after<=h and steps[-1]==after
        social,components=reward(h,after,a,y,False)
        full,_=reward(h,after,a,y,True)
        assert abs(social-(2+int(a>=2)+.5*int(y is True)))<1e-12
        assert abs(full-(social-.15*(h-after)-12*int(after<=0)))<1e-12
        if after>0:assert full>0, "primary live-step reward must not create a negative living treadmill"
    assert maximum<1e-10
    for gamma in [0.,.5,.97,1.]:
        x=generator.normal(size=8)
        explicit=np.array([sum(gamma**(j-i)*x[j] for j in range(i,len(x))) for i in range(len(x))])
        assert np.allclose(returns(x,gamma),explicit,atol=1e-12)
    # Discounted pure potential differences do not measure lifetime at gamma=1.
    short=returns([-4.,-6.],1.)[0];long=returns([-2.,-2.,-2.,-2.,-2.],1.)[0]
    assert short==long==-10.
    positive_short=returns([2.,2.,-10.],.97)[0]
    positive_long=returns([2.]*6+[-10.],.97)[0]
    assert positive_long>positive_short
    # Check the categorical clipped objective derivative against finite differences,
    # including the same exploration mixture used for actual sampling.
    def objective(logits,adv,old):
        p=policy(logits);ratio=p[1]/old
        return -min(ratio*adv,np.clip(ratio,.8,1.2)*adv)
    z=np.array([.2,-.4,.7,-.1]);p=policy(z);old=p[1]
    s=(p-.025)/.9
    derivative=(.9/CONFIG['temperature'])*s[1]*(np.eye(4)[1]-s)
    for advantage in [-2.,1.3]:
        analytic=-advantage*derivative/old
        numeric=[]
        for k in range(4):
            d=np.eye(4)[k]*1e-5
            numeric.append((objective(z+d,advantage,old)-objective(z-d,advantage,old))/2e-5)
        assert np.max(abs(analytic-numeric))<1e-8
    # Same observation/action/acceptance can yield distinct body rewards; the
    # social-only control is identical. These are actual transitions at two h.
    paired=[]
    for h in [300.,3.]:
        after,_=settle(h,2,True,[0.]*5)
        full,parts=reward(h,after,2,True,True);common,_=reward(h,after,2,True,False)
        paired.append(dict(before=h,after=after,body_reward=full,social_reward=common,components=parts))
    assert paired[0]['body_reward']>0>paired[1]['body_reward']
    assert paired[0]['social_reward']==paired[1]['social_reward']
    # Unique, idempotent settlement; death and the observation cap are distinct.
    w=World(61151,0,'validation');identifier=w.encounter_id();record=w.step(identifier,2)
    assert w.step(identifier,2)==record and w.turn==1
    try:w.step(identifier,1)
    except ValueError:pass
    else:raise AssertionError('conflicting replay accepted')
    before=w.health;w.health=1.;w.step(w.encounter_id(),2);assert w.done and w.health==0
    assert before>0
    assert len(jobs('pilot'))==32 and len(jobs('confirm'))==16
    for seed in CONFIG['seeds']:
        items=schedule(seed)
        assert len(items)==72 and len({x['turn'] for x in items})==72
        assert {tuple(x['order']) for x in items[:24]}==set(__import__('itertools').permutations(range(4)))
        for start in range(0,72,8):assert sum(x['frame'] for x in items[start:start+8])==4
    assert len(probes())==216 and len(comprehension())==20
    # Cross-check against the earlier immutable physical mechanism where present.
    previous=Path(__file__).resolve().parents[1]/'social_closed_loop/core.py'
    physical_comparison=None
    if previous.exists():
        spec=spec_from_file_location('previous_body',previous);module=module_from_spec(spec);spec.loader.exec_module(module)
        physical_comparison=0.
        for t in range(1,65):
            h=float(generator.uniform(.001,350));a=int(generator.integers(4));y=bool(generator.integers(2)) if a else None
            eta=noises(61151,t)
            left=settle(h,a,y,eta)[0];right=module.settle(h,a,y,eta)[0]
            physical_comparison=max(physical_comparison,abs(left-right))
        assert physical_comparison<1e-12
    return dict(passed=True,decimal_events=events,max_decimal_error=maximum,
        ppo_finite_difference_passed=True,return_checks_passed=True,
        live_rewards_positive=True,paired_body_rewards=paired,
        global_nonterminal_reward_lower_bounds=live_lower_bounds,
        source_body_comparison_max=physical_comparison,pilot_jobs=32,confirm_jobs=16,
        probe_rows=len(probes()),comprehension_rows=len(comprehension()),
        config_sha256=__import__('hashlib').sha256(canonical(CONFIG)).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=validate();write(args.output,result);print(json.dumps(result))
