"""Complete lives, partner reversal, matched credit replay and parameter transplant."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from experiments.consequence_policy.study import PARTNERS
from experiments.direct_policy.protocol import probabilities, rng, settle, write
from experiments.internal_body.design import CONFIRM, PILOT
from experiments.internal_body.personal import ARMS, LifeActor, LifeInputs
from experiments.social_closed_loop.core import BodyValue

WORLDS=((0,0),(0,1),(1,0),(1,1))
COLUMNS=("context","before","action","response","after","reward","scaled_reward","p0","p1","p2","p3","grad_norm","fisher")


def life_tape(seed,world,rows):
    gen=rng(seed,"unfamiliar-identity-life-v1",list(world))
    lookup={(r["partner"]%2,r["frame"],r["layout"]):i for i,r in enumerate(rows)
            if r["split"]=="new_partner" and r["task"]==4 and r["wording"]==3}
    layouts=np.concatenate([gen.permutation(24) for _ in range(11)])
    pairs=[(p,f) for p in range(2) for f in range(2)]*2
    ix=[]
    for t in range(32):
        for j,k in enumerate(gen.permutation(8)):
            p,f=pairs[k];ix.append(lookup[p,f,layouts[8*t+j]])
    u=gen.random((256,2))
    from scipy.special import ndtr,ndtri
    noise=.1*ndtri(ndtr(-3.)+(ndtr(3.)-ndtr(-3.))*gen.random((256,5)))
    return np.array(ix),u,noise


def post(actor,world,seen=False,only_healthy=False):
    data=actor.data
    key=(seen,tuple(world))
    if not hasattr(data,"assess"):data.assess={}
    if key not in data.assess:
        task,wording=(4,3) if seen else (5,4)
        ix=np.array([i for i,r in enumerate(data.rows) if r["split"]=="new_partner" and r["task"]==task and r["wording"]==wording])
        fav=np.array([2 if probabilities(PARTNERS[data.slot[i]],data.rows[i]["frame"],world[data.slot[i]])[1]>.5 else 3 for i in ix])
        data.assess[key]=ix,fav
    ix,fav=data.assess[key]
    hh=(90.,) if only_healthy else (2.,90.,300.)
    ans=[];by=[]
    for h in hh:
        pi=actor.policy(ix,h)[0]
        a=pi[:,2:].sum(-1)
        style=pi[np.arange(len(ix)),fav]
        ans.append(float(a.mean()));by.append([float(style[data.slot[ix]==j].mean()) for j in (0,1)])
    if only_healthy:return by[0]
    return dict(danger=ans[0],healthy=float(np.mean(ans[1:])),style=float(np.mean(by[1:])),by_partner=np.mean(by[1:],axis=0).tolist())


def replay(data,folder,seed,world,reversal,trace,truth,arm="free"):
    p=data.slot[trace[:,0].astype(int)];band=np.digitize(trace[:,1],[6.,15.])
    gen=rng(seed,"within-life-credit-control",list(world),int(reversal))
    shuffled=np.arange(len(trace));lag=np.arange(len(trace))
    for j in (0,1):
        group=np.flatnonzero(p==j)
        lag[group]=np.roll(group,1)
        for b in range(3):
            part=np.flatnonzero((p==j)&(band==b));shuffled[part]=gen.permutation(part)
    out={}
    target=(1-world[0],world[1]) if reversal else world
    for label,order in (("true",np.arange(len(trace))),("shuffle",shuffled),("lag",lag)):
        actor=LifeActor(data,folder,seed,arm)
        for k,r in enumerate(trace):
            i,h,a=int(r[0]),r[1],int(r[2])
            details=actor.policy(i,h)
            actor.update(i,a,trace[order[k],6],details)
        if label=="true":
            assert np.array_equal(actor.W,truth)
        else:
            out[label]=(post(actor,target),actor.W,order)
    return out


def run_one(actor,seed,world,reversal,tape,body,direct=None):
    indices,u,noise=tape;data=actor.data
    target=(1-world[0],world[1]) if reversal else world
    h=350.;trace=[];check={}
    for k,i in enumerate(indices):
        p=data.slot[i];row=data.rows[i]
        active=target if reversal and k>=32 else world
        details=actor.policy(i,h) if direct is None else direct.policy(i,h,actor)
        pi=details[0][0]
        action=int((u[k,0]>np.cumsum(pi)[:3]).sum())
        pr=probabilities(PARTNERS[p],row["frame"],active[p])
        yes=bool(u[k,1]<pr[action-1]) if action else None
        after,_=settle(h,action,yes,noise[k])
        reward=body.rul(after)-body.rul(h);scaled=reward/min(1.,body.rul(h))
        if direct is None:grad,fisher=actor.update(i,action,scaled,details)
        else:grad,fisher=direct.update(i,action,scaled,actor,details)
        trace.append([i,h,action,-1 if yes is None else int(yes),after,reward,scaled,*pi,grad,fisher])
        h=after
        if k+1 in (32,48,64):
            check[str(k+1)]=dict(alive=h>0,preferred_in_final_world=post(actor,target,only_healthy=True))
        if h<=0:break
    tr=np.array(trace,dtype=np.float64)
    final=post(actor,target)
    for t in (32,48,64):
        if str(t) not in check:
            check[str(t)]=dict(alive=False,preferred_in_final_world=post(actor,target,only_healthy=True))
    own=None
    if actor.arm=="free":
        saved=actor.W
        actor.W=saved[::-1].copy()
        own=post(actor,target)
        actor.W=saved
    return tr,dict(turns=len(tr),tasks=int((tr[:,2]>=2).sum()),capped=bool(h>0),
        post=final,seen=post(actor,target,True),time_probes=check,exchanged=own,
        risky_encounters=int((tr[:,1]<=3).sum()),risky_answers=int(((tr[:,1]<=3)&(tr[:,2]>=2)).sum()))


def summarize(output,seeds):
    rows=[]
    for seed in seeds:
        info=json.loads((output/f"life-{seed}.json").read_bytes())
        means={}
        for rev in (0,1):
            runs=[r for r in info if r["reversal"]==rev]
            for arm in ARMS:
                obs=[r["arms"][arm] for r in runs]
                for metric,v in (("tasks",[a["tasks"] for a in obs]),("turns",[a["turns"] for a in obs]),
                     ("style",[a["post"]["style"] for a in obs]),("seen",[a["seen"]["style"] for a in obs]),
                     ("danger",[a["post"]["danger"] for a in obs]),("healthy",[a["post"]["healthy"] for a in obs])):
                    means[f"{rev}_{arm}_{metric}"]=float(np.mean(v))
                for t in (32,48,64):
                    for partner in (0,1):
                        means[f"{rev}_{arm}_t{t}_p{partner}"]=float(np.mean([a["time_probes"][str(t)]["preferred_in_final_world"][partner] for a in obs]))
            means[f"{rev}_free_over_fixed"]=means[f"{rev}_free_style"]-means[f"{rev}_fixed_style"]
            for control in ("shuffle","lag"):
                means[f"{rev}_free_over_{control}"]=means[f"{rev}_free_style"]-float(np.mean([r["arms"]["free"][control]["style"] for r in runs]))
            means[f"{rev}_free_over_tied"]=means[f"{rev}_free_style"]-means[f"{rev}_tied_style"]
        rows.append(means)
    boot=rng(64802,"personal-bootstrap").integers(len(seeds),size=(12000,len(seeds)))
    stats={}
    for k in rows[0]:
        x=np.array([r[k] for r in rows])
        stats[k]=dict(mean=float(x.mean()),ci95=np.quantile(x[boot].mean(axis=1),[.025,.975]).tolist(),positive=int((x>0).sum()))
    gate=[stats["0_free_over_"+k] for k in ("fixed","shuffle")]
    return dict(seeds=list(seeds),cluster_means=rows,stats=stats,passed=all(v["mean"]>=.1 and v["ci95"][0]>0 and v["positive"]>=(16 if len(seeds)==20 else 3) for v in gate))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wave",choices=["pilot","confirm"],required=True);p.add_argument("--seeds",type=int,nargs="+")
    a=p.parse_args();data=LifeInputs(a.cache);body=BodyValue()
    all_seeds=PILOT if a.wave=="pilot" else CONFIRM
    seeds=a.seeds or all_seeds
    assert set(seeds)<=set(all_seeds)
    for seed in seeds:
        clock=time.monotonic();report=[];raw={};weights={}
        for wi,world in enumerate(WORLDS):
            tape=life_tape(seed,world,data.rows)
            raw[f"w{wi}-context"],raw[f"w{wi}-uniform"],raw[f"w{wi}-noise"]=tape
            for rev in (0,1):
                arms={}
                for name in ARMS:
                    actor=LifeActor(data,a.output,seed,name)
                    trace,measures=run_one(actor,seed,world,rev,tape,body)
                    key=f"w{wi}-v{rev}-{name}"
                    raw[key]=trace
                    if name!="fixed":weights[key]=actor.parameters()
                    if name=="free":
                        for control,(metric,checkpoint,order) in replay(data,a.output,seed,world,rev,trace,actor.W).items():
                            measures[control]=metric;weights[key+"-"+control]=checkpoint;raw[key+"-"+control]=order
                    arms[name]=measures
                report.append(dict(world=list(world),reversal=rev,arms=arms))
        np.savez_compressed(a.output/f"life-{seed}-raw.npz",**raw)
        np.savez_compressed(a.output/f"life-{seed}-parameters.npz",**weights)
        write(a.output/f"life-{seed}.json",report)
        print("finished personal seed",seed,round(time.monotonic()-clock,2),flush=True)
    if all((a.output/f"life-{s}.json").exists() for s in all_seeds):
        result=summarize(a.output,all_seeds);write(a.output/f"life-{a.wave}.json",result)
        print(json.dumps({k:result["stats"][k] for k in ("0_free_over_fixed","0_free_over_shuffle","0_free_over_tied","0_free_style","1_free_t32_p0","1_free_t64_p0")}),flush=True)
if __name__=="__main__":main()
