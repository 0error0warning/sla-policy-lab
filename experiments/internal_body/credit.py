"""Matched credit rules on the same policy/tape, followed by independently sampled lives."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from experiments.consequence_policy.experiment import evaluation_reference
from experiments.consequence_policy.study import EXPLORATION,TEMP
from experiments.direct_policy.protocol import rng,write
from experiments.internal_body.design import SPLIT_H
from experiments.internal_body.lives import WORLDS,life_tape,post,run_one
from experiments.internal_body.personal import LifeActor,LifeInputs
from experiments.social_closed_loop.core import BodyValue

PILOT=tuple(range(64601,64605));CONFIRM=tuple(range(64701,64721))
RULES=("fixed","scalar","scalar3","predict","predict1","fisher","predict_tied")


class Learner(LifeActor):
    def __init__(self,data,folder,seed,rule):
        super().__init__(data,folder,seed,"fixed" if rule=="fixed" else ("tied" if rule=="predict_tied" else "free"))
        self.rule=rule
        self.origin=LifeActor(data,folder,seed,"fixed") if rule.startswith("predict") else None

    def policy(self,indices,h):
        if np.ndim(indices)==0:self.current_h=h
        return super().policy(indices,h)

    def update(self,index,action,outcome,details):
        if self.rule in ("fixed","scalar"):return super().update(index,action,outcome,details)
        if self.rule=="scalar3":return super().update(index,action,3*(float(outcome)+.9)-.9,details)
        pi,s,r,_,den,pg=details
        pi,s,r,den=pi[0],s[0],r[0],float(den[0,0]);adv=float(outcome)+.9
        z=self.data
        if self.rule.startswith("predict"):
            old=self.origin.policy(index,self.current_h)[1][0]
            movement=np.log(s)-np.log(old)
            estimate=float(movement[action]-movement.mean())
            # Chosen centered, temperature-scaled logit in genuine residual coordinates.
            v=(z.t[:,action]-z.t.mean(axis=1))/TEMP
            grad=v/den-r*(v@r)/(len(r)*den**3)
            energy=float(grad@grad)
            step=.55*((1. if self.rule=="predict1" else 3.)*adv-estimate)/(.05+energy)
        else:
            jac=(np.eye(4,dtype=np.float32)-s)*(1-EXPLORATION)*s[:,None]/pi[:,None]/TEMP
            v=jac@z.t.T
            G=(v/den-(v@r)[:,None]*r/(len(r)*den**3)).astype(np.float64)
            Q=np.sqrt(pi)[:,None]*G
            fisher=float((Q*Q).sum())
            ridge=max(1e-7,.03*fisher)
            inner=Q@Q.T+np.eye(4)*ridge
            D=(G-(G@Q.T)@np.linalg.solve(inner,Q))/ridge
            ordinary=G/(.05+fisher)
            def local_size(matrix):return float(np.sum(pi[:,None]*(matrix@Q.T)**2))
            D*=np.sqrt(local_size(ordinary)/max(local_size(D),1e-24))
            grad=D[action].astype(np.float32)
            energy=fisher;step=.55*adv
        slot=0 if self.arm=="tied" else z.slot[index]
        self.W[slot]+=np.einsum("b,k,d->bkd",pg,self.keys[index],grad)*step
        return float(np.linalg.norm(grad)),energy


def generic_fit(cache,folder,seed,body,bank):
    from experiments.internal_body.population import acquire
    from experiments.internal_body.suffix import InternalPolicy,fit
    indices=np.arange(768)
    src=acquire(seed,bank,indices,body,folder)
    policy=InternalPolicy(bank,"generic")
    history=fit(policy,bank.grid(indices,SPLIT_H),(src[0]["real"]+src[1]["real"])/2)
    policy.save(folder/f"generic-{seed}.npz")
    write(folder/f"generic-{seed}-steps.json",history)


def forced(data,out,seed,rule,trace,order):
    actor=Learner(data,out,seed,rule)
    for row,j in zip(trace,order):
        i,h,a=int(row[0]),float(row[1]),int(row[2])
        actor.update(i,a,float(trace[j,6]),actor.policy(i,h))
    return actor


def shuffled_order(data,seed,world,rev,trace):
    p=data.slot[trace[:,0].astype(int)]
    band=np.digitize(trace[:,1],[6.,15.]);inds=np.arange(len(trace))
    gen=rng(seed,"prediction-credit",list(world),int(rev))
    for j in (0,1):
        for k in range(3):
            q=np.flatnonzero((p==j)&(band==k));inds[q]=gen.permutation(q)
    return inds


def summary(out,seeds):
    clusters=[]
    for seed in seeds:
        runs=json.loads((out/f"credit-{seed}.json").read_bytes())
        cluster={}
        for rev in (0,1):
            sample=[r for r in runs if r["reversal"]==rev]
            for arm in RULES:
                for k in ("style","healthy","danger"):
                    cluster[f"{rev}_{arm}_{k}"]=float(np.mean([r["arms"][arm]["post"][k] for r in sample]))
                for k in ("turns","tasks"):
                    cluster[f"{rev}_{arm}_{k}"]=float(np.mean([r["arms"][arm][k] for r in sample]))
                for t in (32,64):
                    cluster[f"{rev}_{arm}_p0_t{t}"]=float(np.mean([r["arms"][arm]["time_probes"][str(t)]["preferred_in_final_world"][0] for r in sample]))
            for other in ("fixed","scalar","scalar3","fisher","predict1","predict_tied"):
                cluster[f"{rev}_vs_{other}"]=cluster[f"{rev}_predict_style"]-cluster[f"{rev}_{other}_style"]
            cluster[f"{rev}_vs_shuffled"]=cluster[f"{rev}_predict_style"]-float(np.mean([r["shuffle"]["style"] for r in sample]))
            for arm in ("scalar","predict","predict1","scalar3","fisher"):
                cluster[f"{rev}_sametape_{arm}"]=float(np.mean([r["fixed_tape"][arm]["style"] for r in sample]))
        clusters.append(cluster)
    boot=rng(64807,"online-credit-bootstrap").integers(len(seeds),size=(12000,len(seeds)))
    stats={}
    for k in clusters[0]:
        x=np.array([r[k] for r in clusters])
        stats[k]=dict(mean=float(x.mean()),ci95=np.quantile(x[boot].mean(axis=1),[.025,.975]).tolist(),positive=int((x>0).sum()))
    gate=[stats[k] for k in ("0_vs_fixed","0_vs_shuffled")]
    return dict(seeds=list(seeds),cluster_means=clusters,stats=stats,passed=all(v["mean"]>=.1 and v["ci95"][0]>0 and v["positive"]>=(16 if len(seeds)==20 else 3) for v in gate))


def main():
    import torch
    from experiments.internal_body.suffix import StateBank
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wave",choices=("pilot","confirm"),required=True);p.add_argument("--threads",type=int,default=3)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    body=BodyValue();data=LifeInputs(a.cache);bank=StateBank(a.cache)
    seeds=PILOT if a.wave=="pilot" else CONFIRM
    if a.wave=="confirm":assert json.loads((a.output/"pilot.json").read_bytes())["passed"]
    for seed in seeds:
        tic=time.monotonic();generic_fit(a.cache,a.output,seed,body,bank)
        reports=[];raw={};parameters={}
        for j,world in enumerate(WORLDS):
            tape=life_tape(seed,world,data.rows)
            raw[f"w{j}-context"],raw[f"w{j}-uniform"],raw[f"w{j}-noise"]=tape
            for rev in (0,1):
                target=(1-world[0],world[1]) if rev else world
                row=dict(world=list(world),reversal=rev,arms={},fixed_tape={})
                prefix=f"w{j}-v{rev}"
                for arm in RULES:
                    actor=Learner(data,a.output,seed,arm)
                    trace,obs=run_one(actor,seed,world,rev,tape,body)
                    row["arms"][arm]=obs;raw[prefix+"-"+arm]=trace
                    if arm!="fixed":parameters[prefix+"-"+arm]=actor.W
                    if arm=="predict":
                        ind=shuffled_order(data,seed,world,rev,trace)
                        alt=forced(data,a.output,seed,arm,trace,ind)
                        row["shuffle"]=post(alt,target);raw[prefix+"-permutation"]=ind
                        parameters[prefix+"-shuffled"]=alt.W
                scalar=raw[prefix+"-scalar"]
                for rule in ("scalar","predict","predict1","scalar3","fisher"):
                    alt=forced(data,a.output,seed,rule,scalar,np.arange(len(scalar)))
                    if rule=="scalar":assert np.array_equal(alt.W,parameters[prefix+"-scalar"])
                    row["fixed_tape"][rule]=post(alt,target)
                    if rule!="scalar":parameters[prefix+"-same-"+rule]=alt.W
                reports.append(row)
        np.savez_compressed(a.output/f"credit-{seed}-raw.npz",**raw)
        np.savez_compressed(a.output/f"credit-{seed}-parameters.npz",**parameters)
        write(a.output/f"credit-{seed}.json",reports)
        print("credit seed",seed,round(time.monotonic()-tic,2),flush=True)
    result=summary(a.output,seeds);write(a.output/(a.wave+".json"),result)
    print(json.dumps({k:result["stats"][k] for k in ("0_vs_fixed","0_vs_shuffled","0_vs_scalar3","0_vs_scalar")}),flush=True)
if __name__=="__main__":main()
