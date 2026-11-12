"""Fit all matched coverage and credit controls on realized action/response pairs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.policy import feedback_scores
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import InternalPolicy,fit
from experiments.lethal_boundary.design import ANCHOR_Q,ARMS,CONFIRM,CROSS,PILOT,SENSES,levels,prehistory,profiles
from experiments.lethal_boundary.physics import action_responses,experience,focal_audit
from experiments.lethal_boundary.states import ALL_H,WITNESS_H,WITNESS_K,Bank,at,probes
from experiments.social_closed_loop.core import BodyValue


def acquisition(seed,bank,body,d):
    action,mu,u,noise=draw_tape(seed,bank.training[None][2])
    slots=profiles(seed,bank.train)
    checks,checked=prehistory(seed)
    np.savez_compressed(d/"tape.npz",action=action,mu=mu,uniform=u,noise=noise,profile=slots,
                        prior_uniform=checks,prior_outcome=checked)
    targets=[]
    for mapping in (0,1):
        folder=d/f"m{mapping}";folder.mkdir(exist_ok=True)
        response=action_responses(bank.train,mapping,ANCHOR_Q[slots],action,u)
        raw={"response":response};credit={};audits={}
        for arm in ARMS:
            hs=levels(arm,seed)
            after,reward=experience(hs,action,response,noise,body)
            a= focal_audit(hs,action,response,after,reward,body)
            audits[arm]=a
            for row in a[:16]:
                if row["action"]>=2:
                    assert row["count"]>row["maintenance"]>0
                    if row["health"]<2.44:
                        assert row["deaths"]==row["count"]
                    else:
                        assert row["maintained_deaths"]==0 and row["deaths"]+row["maintenance"]==row["count"]
            scale=np.minimum(1.,[body.rul(h) for h in hs])
            credit[arm]=feedback_scores(action,reward,mu,scale)
            if arm=="M":
                for j,h in enumerate(hs):
                    source="D" if h==levels("D",seed)[j] else "R"
                    np.testing.assert_array_equal(raw[source+"_after"][j],after[j])
                    np.testing.assert_array_equal(raw[source+"_reward"][j],reward[j])
            else:
                raw[arm+"_after"]=after;raw[arm+"_reward"]=reward
        np.savez_compressed(folder/"outcomes.npz",**raw)
        write(folder/"training-audit.json",audits)
        targets.append(credit)
    return targets,slots


def seed_run(seed,bank,body,output):
    d=output/f"s{seed}"
    d.mkdir(exist_ok=True)
    targets,slots=acquisition(seed,bank,body,d)
    inputs={"plain":bank.plain_train,"record":bank.record_train(slots)}
    for mapping,credit in enumerate(targets):
        folder=d/f"m{mapping}";history={}
        for sense in SENSES:
            assignments={name:(name,name) for name in ARMS}
            if sense=="plain":assignments.update(CROSS)
            for name,(sensor,source) in assignments.items():
                model=InternalPolicy(bank,"linear")
                name=sense+"-"+name
                history[name]=fit(model,at(inputs[sense],levels(sensor,seed)),credit[source])
                model.save(folder/(name+".npz"))
                np.savez_compressed(folder/(name+"-probe.npz"),**probes(model,bank,sense))
        write(folder/"steps.json",history)
    write(d/"complete.json",{"seed":seed,"sampled_choices":49152,"mappings":2,"optimizer_steps_per_arm":75,
                              "training_succeeded":True,"arms_per_mapping":10})


def main():
    p=argparse.ArgumentParser()
    for k in ("cache","arch","output"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--wave",choices=("pilot","confirm"),required=True)
    p.add_argument("--seeds",type=int,nargs="+")
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=True)
    bank=Bank(a.cache,a.arch);body=BodyValue()
    if a.wave=="pilot":
        for sense in SENSES:
            np.savez_compressed(a.output/("frozen-"+sense+".npz"),**probes(None,bank,sense))
        write(a.output/"axes.json",{"health":ALL_H.tolist(),"witness_health":WITNESS_H.tolist(),"witness_counts":WITNESS_K.tolist(),
                                    "group":["P17/polite","P17/exchange","P42/polite","P42/exchange"],"axis":["health","history_count","group","semantic_action"]})
    else:
        assert all((a.output/f"s{s}/complete.json").exists() for s in PILOT)
    wave=PILOT if a.wave=="pilot" else CONFIRM
    seeds=a.seeds or wave
    assert set(seeds)<=set(wave) and len(set(seeds))==len(seeds)
    for s in seeds:
        if (a.output/f"s{s}/complete.json").exists():continue
        start=time.monotonic();seed_run(s,bank,body,a.output)
        print(json.dumps({"seed":s,"seconds":round(time.monotonic()-start,2)}),flush=True)


if __name__=="__main__":main()
