"""Independent replay of the acquired body responses and the two mortality calculations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import COSTS,settle,write,rng
from experiments.lethal_boundary.design import ARMS,ANCHOR_Q,CONFIRM,DOOM,HEALTH,PILOT,RESCUE,levels
from experiments.lethal_boundary.physics import cutoff,integration
from experiments.social_closed_loop.core import BodyValue


def inverse_after(h,choice,response,noise):
    c=np.asarray(COSTS)[choice]
    power=(1+c*(1+noise[...,0])/350)*np.prod(1+(response!=1)[...,None]*(1+noise[...,1:])/350,axis=-1)
    return np.maximum(0,700-(700-np.asarray(h)[:,None,None])*power)


def physical_check(body):
    risk,_=integration(HEALTH,body);other,_=integration(HEALTH,body,67091)
    h=np.array([.8125,1.9375,2.4375,2.9375,3.4375,3.9375,4.4375,4.8125,5.6875,8.9375,10.8125,12.8125,90.,300.])
    ix=np.searchsorted(HEALTH,h);np.testing.assert_array_equal(HEALTH[ix],h)
    u=rng(67911,"sequential-independent-risk").random((131072,5))
    eps=.1*ndtri(ndtr(-3.)+(ndtr(3.)-ndtr(-3.))*u)
    err=0.
    for i,x in enumerate(h):
        for a,c in enumerate(COSTS):
            for maintained in (0,1):
                y=np.maximum(0,x-c*(2-x/350)*(1+eps[:,0]))
                for n in range(1,5):
                    y=np.maximum(0,y-(1-maintained)*(2-y/350)*(1+eps[:,n]))
                val=float((y==0).mean())
                err=max(err,abs(val-risk[ix[i],a,maintained]))
    assert err<.007 and np.abs(risk-other).max()<.007
    for c in (.25,1.75):
        for e in (-.3,0,.3):
            edge=cutoff(c,e);a=1 if c==.25 else 2
            assert settle(edge-1e-8,a,True,[e]*5)[0]==0 and settle(edge+1e-8,a,True,[e]*5)[0]>0
    for x in np.r_[DOOM,DOOM+.55]:
        assert settle(x,2,True,[-.3]*5)[0]==0
    for x in np.r_[RESCUE,RESCUE+.55]:
        assert settle(x,2,True,[.3]*5)[0]>0 and settle(x,2,False,[-.3]*5)[0]==0
    return {"protected_answer_support":cutoff(1.75,np.array([-.3,0,.3])).tolist(),
        "unprotected_answer_support":[700*(1-1/((1+1.75*z/350)*(1+z/350)**4)) for z in (.7,1,1.3)],
        "independent_sequential_risk_max_difference":err,"sobol_between_scrambles_max_difference":float(np.abs(risk-other).max())}


def run(root,cache,out,seeds):
    body=BodyValue();rows=contexts()[:768]
    assert all(not np.intersect1d(HEALTH,levels(arm,s)).size for arm in ARMS for s in seeds)
    with np.load(cache/"plain.npz") as z:native=z["n"][:768]
    qa=physical_check(body)
    signals={};unique=settlements=scalar_reads=steps=0;max_error=0.;model_rows=[]
    for seed in seeds:
        d=root/f"s{seed}";raw=draw_tape(seed,native)
        with np.load(d/"tape.npz") as tape:
            for name,value in zip(("action","mu","uniform","noise"),raw,strict=True):np.testing.assert_array_equal(value,tape[name])
            action,mu,u,noise=raw;slots=tape["profile"];q=ANCHOR_Q[slots]
            np.testing.assert_array_equal(np.bincount(slots),[384,384])
            np.testing.assert_array_equal(tape["prior_uniform"]<ANCHOR_Q[:,None],tape["prior_outcome"])
            np.testing.assert_array_equal(tape["prior_outcome"].sum(1),[5,17])
        unique+=action.size
        for m in (0,1):
            folder=d/f"m{m}"
            fav=np.array([2 if (r["partner"]^r["frame"]^m)==0 else 3 for r in rows])
            probs=np.where(action==0,0.,np.where(action==1,.15,np.where(action==fav[None,:,None],q[None,:,None],.1)))
            response=np.where(action==0,-1,(u<probs).astype(np.int8))
            audit=json.loads((folder/"training-audit.json").read_bytes())
            with np.load(folder/"outcomes.npz") as z:
                np.testing.assert_array_equal(z["response"],response)
                for arm in ARMS:
                    hs=levels(arm,seed); y=inverse_after(hs,action,response,noise)
                    if arm=="M":
                        dy=np.array([z[("D" if h==levels("D",seed)[j] else "R")+"_after"][j] for j,h in enumerate(hs)])
                        rw=np.array([z[("D" if h==levels("D",seed)[j] else "R")+"_reward"][j] for j,h in enumerate(hs)])
                    else:dy=z[arm+"_after"];rw=z[arm+"_reward"]
                    err=float(np.max(abs(y-dy)));max_error=max(max_error,err)
                    np.testing.assert_allclose(y,dy,rtol=0,atol=2e-12)
                    reward=body.interpolate(y,body.reference)-body.interpolate(hs[:,None,None],body.reference)
                    np.testing.assert_allclose(reward,rw,atol=1e-10,rtol=0)
                    np.testing.assert_array_equal(y==0,dy==0);settlements+=action.size
                    for j in (0,3,7):
                        for i in (0,129,391,767):
                            a=int(action[j,i,0]);r=int(response[j,i,0])
                            v=settle(hs[j],a,None if a==0 else bool(r),noise[j,i,0])[0]
                            assert abs(v-dy[j,i,0])<1e-12
                            scalar_reads+=1
                    sig=[]
                    for a in range(4):
                        rr=[v for v in audit[arm] if v["slot"]<4 and v["action"]==a]
                        n=sum(v["count"] for v in rr)
                        sig.append({"counts":n,"maintained":sum(v["maintenance"] for v in rr),
                                    "maintained_death":sum(v["maintained_deaths"] for v in rr),"deaths":sum(v["deaths"] for v in rr),
                                    "mean_advantage":sum(v["count"]*v["mean_advantage"] for v in rr)/n})
                    signals[f"{seed}-{m}-{arm}"]=sig
                for a in range(4):
                    group=[signals[f"{seed}-{m}-{arm}"][a] for arm in ARMS]
                    assert len({(v["counts"],v["maintained"]) for v in group})==1
            log=json.loads((folder/"steps.json").read_bytes())
            assert len(log)==10
            for condition,hist in log.items():
                assert len(hist)==75 and all(np.isfinite(v).all() for v in hist)
                with np.load(folder/(condition+".npz")) as model:
                    assert model["A"].shape==(2,12,33,4) and model["B"].shape==(4,2560)
                    assert (bn:=np.linalg.norm(model["B"]))>0
                model_rows.append({"seed":seed,"mirror":m,"condition":condition,"B_norm":float(bn),"first_loss":hist[0][1],"last_loss":hist[-1][1]})
                steps+=75
    out.mkdir(parents=True,exist_ok=True);write(out/"training-signals.json",signals);write(out/"param-audit.json",model_rows)
    qa.update({"seed_ids":seeds,"unique_sampled_choices":unique,"paired_body_settlements":settlements,"direct_scalar_checks":scalar_reads,
              "inverse_sequential_after_max_error":max_error,"fully_checked_optimizer_steps":steps,"checkpoints_checked":len(model_rows),"per_fit_sampled_choices":49152})
    write(out/"audit.json",qa);print(json.dumps(qa))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","cache","output"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    a=p.parse_args();run(a.root,a.cache,a.output,list(PILOT if a.pilot else (*PILOT,*CONFIRM)))
