"""A fixed action-neutral body channel with updates only to original RMSNorm coordinates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.policy import deployed
from experiments.consequence_policy.study import TEST_H,TEMP,basis
from experiments.direct_policy.protocol import rng,write
from experiments.internal_body.design import SPLIT_H
from experiments.internal_body.population import acquire
from experiments.internal_body.suffix import InternalPolicy,StateBank,arr,distribution,fit
from experiments.social_closed_loop.core import BodyValue

PILOT=tuple(range(64401,64405));CONFIRM=tuple(range(64501,64521))


def carrier_basis(residual,weighted,seed):
    q,_=np.linalg.qr(weighted.astype(np.float64))
    mean=residual.astype(np.float64).mean(axis=0)
    u=mean-q@(q.T@mean);u/=np.linalg.norm(u)
    random=rng(seed,"fixed-unlabeled-orthogonal-carrier").normal(size=(len(u),12))
    sub=np.column_stack((q,u))
    v,_=np.linalg.qr(random-sub@(sub.T@random))
    assert np.abs(sub.T@v).max()<1e-10
    return u.astype(np.float32),v.astype(np.float32)


def rotate(r,g,u,v):
    # r may have one shared health index or distinct text vectors for each health.
    axis=g@v.T
    axis=axis/torch.linalg.vector_norm(axis,dim=-1,keepdim=True)
    along_u=r@u
    along_v=(r*axis[:,None]).sum(-1)
    co,si=.5,np.sqrt(.75)
    return r+((co-1)*along_u-si*along_v)[:,:,None]*u+ (si*along_u+(co-1)*along_v)[:,:,None]*axis[:,None]


def rotated(grid,bank,u,v):
    r=rotate(grid["r"],grid["g"],arr(u,bank.device),arr(v,bank.device))
    den=(r.square().mean(-1,keepdim=True)+bank.cfg["epsilon"]).sqrt()
    # The nullspace construction should preserve the four logits at birth even
    # before anchoring numerical roundoff to the genuine frozen Qwen output.
    old=grid["dot"]/grid["div"];new=(r@bank.t)/den
    max_logit=float((new-old).abs().max())
    norm_error=float((den-grid["div"]).abs().max())
    assert max_logit<.002 and norm_error<.002,(max_logit,norm_error)
    return {**grid,"r":r,"div":den},(max_logit,norm_error)


def stats(out,seeds):
    paired=[]
    for seed in seeds:
        values=[]
        for m in (0,1):
            a=json.loads((out/f"s{seed}-m{m}/native-metrics.json").read_bytes())
            z=a["sensor"]
            values.append(dict(critical=a["frozen"]["danger_answer"]-z["danger_answer"],
                critical_shuffle=a["shuffle"]["danger_answer"]-z["danger_answer"],
                healthy=z["healthy_answer"]-a["frozen"]["healthy_answer"],
                gap=z["gap"]-a["hidden"]["gap"]))
        paired.append({k:float(np.mean([v[k] for v in values])) for k in values[0]})
    boot=rng(64806,"original-only-bootstrap").integers(len(seeds),size=(12000,len(seeds)))
    result={}
    for k,t in dict(critical=.1,critical_shuffle=.1,healthy=-.05,gap=.1).items():
        x=np.array([r[k] for r in paired]);ci=np.quantile(x[boot].mean(axis=1),[.025,.975])
        passed=x.mean()>=t
        if k=="critical" and len(seeds)==4:passed=passed and (x>0).sum()>=3
        if k!="healthy" and len(seeds)==20:passed=passed and (x>0).sum()>=16 and ci[0]>0
        result[k]=dict(mean=float(x.mean()),positive=int((x>0).sum()),ci95=ci.tolist(),passed=bool(passed))
    return dict(seeds=list(seeds),paired=paired,stats=result,passed=all(v["passed"] for v in result.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--wave",choices=("pilot","confirm"),required=True)
    p.add_argument("--threads",type=int,default=3)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    seeds=PILOT if a.wave=="pilot" else CONFIRM
    if a.wave=="confirm":assert json.loads((a.output/"pilot.json").read_bytes())["passed"]
    bank=StateBank(a.cache);body=BodyValue()
    train=np.arange(768);test=np.arange(768,960)
    ref=[evaluation_reference(body,[bank.rows[i] for i in test],m) for m in (0,1)]
    for seed in seeds:
        observed=acquire(seed,bank,train,body,a.output)
        u,v=carrier_basis(bank.r[train],bank.t.cpu().numpy(),seed)
        u2,v2=carrier_basis(bank.r[train],bank.t.cpu().numpy(),seed+10000)
        normalized,err1=rotated(bank.grid(train,SPLIT_H),bank,u,v)
        evaluated,err2=rotated(bank.grid(test,TEST_H),bank,u,v)
        exchanged,_=rotated(bank.grid(test,TEST_H),bank,u2,v2)
        np.savez_compressed(a.output/f"basis-{seed}.npz",u=u,v=v,swapped_v=v2)
        for m,score in enumerate(observed):
            folder=a.output/f"s{seed}-m{m}"
            frozen=np.broadcast_to(deployed(bank.native[test]/TEMP),(len(TEST_H),len(test),4)).copy()
            probes={"frozen":frozen};logs={}
            for arm in ("sensor","shuffle","hidden","text"):
                model=InternalPolicy(bank,"native_prompt" if arm=="text" else "native_hidden")
                tr=normalized if arm in ("sensor","shuffle") else bank.grid(train,SPLIT_H,arm=="text")
                te=evaluated if arm in ("sensor","shuffle") else bank.grid(test,TEST_H,arm=="text")
                logs[arm]=fit(model,tr,score["shuffle" if arm=="shuffle" else "real"])
                model.save(folder/f"native-{arm}.npz")
                with torch.no_grad():
                    probes[arm]=distribution(model.scores(te)).cpu().numpy()
                    if arm=="sensor":
                        probes["carrier_exchanged"]=distribution(model.scores(exchanged)).cpu().numpy()
                        probes["carrier_removed"]=distribution(model.scores(bank.grid(test,TEST_H))).cpu().numpy()
            np.savez_compressed(folder/"native-probes.npz",**probes)
            write(folder/"native-metrics.json",{k:metrics(vv,ref[m]) for k,vv in probes.items()})
            write(folder/"native-steps.json",dict(optimization=logs,neutrality_train=err1,neutrality_test=err2))
        print("native carrier seed",seed,flush=True)
    result=stats(a.output,seeds);write(a.output/f"{a.wave}.json",result)
    print(json.dumps(result["stats"]),flush=True)
if __name__=="__main__":main()
