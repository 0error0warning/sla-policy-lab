"""Same-budget paired intervention on repeated body anchors and real continuous experiences."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import rng,write
from experiments.internal_body.suffix import InternalPolicy,arr,fit
from experiments.lethal_boundary.design import ANCHOR_Q,HEALTH,levels,profiles,prehistory
from experiments.lethal_boundary.physics import action_responses
from experiments.lethal_boundary.states import Bank,at,probes
from experiments.social_closed_loop.core import BodyValue,body_event

SEEDS=tuple(range(67201,67213))
CELLS={"fixed":("fixed","fixed"),"jitter":("jitter","jitter"),"jitter_gets_fixed":("jitter","fixed"),"fixed_gets_jitter":("fixed","jitter")}


def health_pairs(seed,rows,slots):
    anchors=levels("M",seed)
    fixed=np.broadcast_to(anchors[:,None],(8,len(rows))).copy()
    moving=fixed.copy()
    groups={}
    for i,(r,slot) in enumerate(zip(rows,slots,strict=True)):
        groups.setdefault((r["partner"],r["frame"],r["task"],int(slot)),[]).append(i)
    gen=rng(seed,"continuous-body-antithetic")
    for j,h in enumerate(anchors[:4]):
        radius=min(.50,h-.72,2.40-h) if h<2.44 else min(.50,h-4.54,6.49-h)
        assert radius>0
        for group in groups.values():
            assert len(group)%2==0
            u=gen.uniform(0,radius,size=len(group)//2)
            moving[j,group]+=gen.permutation(np.r_[-u,u])
            assert abs(moving[j,group].mean()-h)<1e-12
        assert np.unique(moving[j]).size==len(rows)
        assert (np.all(moving[j]<2.441) if h<2.44 else np.all((moving[j]>4.521)&(moving[j]<7.994)))
    assert not np.intersect1d(moving[:4],HEALTH).size
    return {"fixed":fixed,"jitter":moving}


class LocalHealthPolicy(InternalPolicy):
    def coefficients(self,x,g,p,masked=False,pooled=False):
        if g.ndim==2:return super().coefficients(x,g,p,masked,pooled)
        assert x.shape[0]==1 and not masked and not pooled and self.kind=="linear"
        per_basis=torch.einsum("nd,nbdr->nbr",x[0],self.A[p])
        return torch.einsum("hnb,nbr->hnr",g,per_basis)


def settle_cells(h,a,response,noise,body):
    before=h[:,:,None]
    after=body_event(before,np.asarray([0.,.25,1.75,1.75])[a],noise[...,0])
    for j in range(1,5):after=body_event(after,(response!=1).astype(float),noise[...,j])
    return after,body.interpolate(after,body.reference)-body.interpolate(before,body.reference)


def signals(a,outcome,mu,h,body):
    scale=np.minimum(1.,body.interpolate(h,body.reference))
    n=a.shape[-1];adv=(outcome-outcome.mean(-1,keepdims=True))*n/(n-1)/scale[:,:,None]
    return np.stack([np.mean(adv*(a==j),axis=-1)/mu[:,j] for j in range(4)],axis=-1).astype(np.float32)


def algebra_check(bank,grid,hs,seed):
    orig=InternalPolicy(bank,"linear");alt=LocalHealthPolicy(bank,"linear")
    with torch.no_grad():
        small=np.random.default_rng(seed).normal(0,.012,orig.B.shape).astype(np.float32)
        orig.B.copy_(torch.from_numpy(small));alt.B.copy_(orig.B)
    args=[at(grid,hs[:,0]),{**grid,"g":arr(basis(hs),"cpu")}]
    models=(orig,alt);outputs=[]
    for m,g in zip(models,args,strict=True):
        v=m.scores(g);(v.square().mean()).backward();outputs.append(v.detach().numpy())
    errors={"scores":float(np.abs(outputs[0]-outputs[1]).max())}
    for k in ("A","B"):
        errors["grad_"+k]=float((getattr(orig,k).grad-getattr(alt,k).grad).abs().max())
    assert errors["scores"]<2e-5 and max(errors["grad_A"],errors["grad_B"])<2e-5
    return errors


def run(root,bank,seed,body):
    d=root/f"s{seed}";d.mkdir(parents=True,exist_ok=True)
    slots=profiles(seed,bank.train);hs=health_pairs(seed,bank.train,slots)
    grid=bank.record_train(slots)
    a,mu,u,noise=draw_tape(seed,bank.training[None][2])
    pu,pr=prehistory(seed)
    np.savez_compressed(d/"tape.npz",action=a,mu=mu,uniform=u,noise=noise,profile=slots,prior_uniform=pu,prior_outcome=pr,**hs)
    checks=algebra_check(bank,grid,hs["fixed"],seed)
    for m in (0,1):
        folder=d/f"m{m}";folder.mkdir(exist_ok=True)
        response=action_responses(bank.train,m,ANCHOR_Q[slots],a,u);raw={"response":response}
        targets={};history={}
        for name,x in hs.items():
            after,reward=settle_cells(x,a,response,noise,body)
            targets[name]=signals(a,reward,mu,x,body)
            raw[name+"_after"]=after;raw[name+"_reward"]=reward
            if name=="fixed":focal=(x[:4,:,None]<2.44)
            else:np.testing.assert_array_equal(focal,x[:4,:,None]<2.44)
            chosen=(a[:4]>=2)&(response[:4]==1)
            assert np.all((after[:4]==0)[chosen]==np.broadcast_to(focal,chosen.shape)[chosen])
        np.savez_compressed(folder/"outcomes.npz",**raw)
        for name,(sensor,credit) in CELLS.items():
            model=LocalHealthPolicy(bank,"linear")
            history[name]=fit(model,{**grid,"g":arr(basis(hs[sensor]),"cpu")},targets[credit])
            model.save(folder/(name+".npz"))
            p=probes(model,bank,"record")
            np.savez_compressed(folder/(name+"-probe.npz"),**p)
            if seed==SEEDS[0] and m==0 and name=="fixed":
                other=InternalPolicy(bank,"linear")
                fit(other,at(grid,hs["fixed"][:,0]),targets["fixed"])
                op=probes(other,bank,"record")
                checks["fixed_fit_max_probability_difference"]=float(np.abs(op["panel"]-p["panel"]).max())
                checks["fixed_fit_mean_curve_max_difference"]=float(np.abs(op["mean"]-p["mean"]).max())
                assert checks["fixed_fit_max_probability_difference"]<.002
        write(folder/"steps.json",history)
    write(d/"complete.json",{"seed":seed,"checks":checks,"unique_focal_healths":int(np.unique(hs["jitter"][:4]).size),"examples_per_fit":a.size})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","arch","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    b=Bank(a.cache,a.arch);body=BodyValue()
    for s in SEEDS:
        if (a.output/f"s{s}/complete.json").exists():continue
        run(a.output,b,s,body);print("completed",s,flush=True)
