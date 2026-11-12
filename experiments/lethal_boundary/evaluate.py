"""Pointwise physical and behavioral evaluation on novel healths and partner probabilities."""
import argparse
import csv
import gzip
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from experiments.direct_policy.protocol import rng,write
from experiments.lethal_boundary.design import ARMS,CONFIRM,CROSS,HEALTH,INTERIOR,PILOT,QS,favorite,levels
from experiments.lethal_boundary.physics import integration,maintenance_table
from experiments.social_closed_loop.core import BodyValue

METRICS=("answer","intrinsic_fatal_answer","answer_and_die","total_death","safe_task","preferred_answer","reference_regret","expected_rul_change")
MI={v:i for i,v in enumerate(METRICS)}
BANDS={"doomed":HEALTH<=2.44,"transition":(HEALTH>2.44)&(HEALTH<4.521),"rescue":(HEALTH>=4.521)&(HEALTH<=6.5),"healthy":np.isin(HEALTH,[90,300])}
GROUPS=[{"partner":p,"frame":f} for p in range(2) for f in range(2)]
CORE=("plain-frozen","record-frozen",*("plain-"+k for k in (*ARMS,*CROSS)),*("record-"+k for k in ARMS))


def index(axis,h):
    loc=np.searchsorted(axis,h)
    np.testing.assert_array_equal(axis[loc],h)
    return loc


def policies(means,config):
    if means.shape[1]==1:
        return np.broadcast_to(means[:,0],(len(QS),*means[:,0].shape))
    if config=="fixed":return np.broadcast_to(means[:,11],(len(QS),*means[:,11].shape))
    if config=="clean":return means[:,np.rint(20*QS).astype(int)].transpose(1,0,2,3)
    w=binom.pmf(np.arange(21)[None,:],20,QS[:,None])
    if config=="reversed":w=w[:,::-1]
    return np.einsum("qk,hkga->qhga",w,means,optimize=True)


def physics_matrix(death,value,mapping):
    p=np.stack([maintenance_table(GROUPS,mapping,q) for q in QS])[:,None]
    d=(1-p)*death[None,:,:,0][:,:,None]+p*death[None,:,:,1][:,:,None]
    v=(1-p)*value[None,:,:,0][:,:,None]+p*value[None,:,:,1][:,:,None]
    return d,v,np.tile(favorite(GROUPS,mapping),(len(QS),1))


def scores(prob,physics,intrinsic):
    d,v,preferred=physics
    a=prob[...,2:].sum(-1)
    joint=(prob[...,2:]*d[...,2:]).sum(-1)
    exp=(prob*v).sum(-1)
    # Score the actual expression with higher maintenance. At q=.15 all requested probabilities can tie with the refusal; only answering styles enter this measure.
    fav=np.take_along_axis(prob,preferred[:,None,:,None],axis=-1)[...,0]
    out=np.stack((a,a*intrinsic[None,:,None],joint,(prob*d).sum(-1),a-joint,fav,v.max(-1)-exp,exp),axis=-1)
    return out.mean(axis=2)


def interpolation(pi,axis,anchors,mode):
    anc=np.sort(anchors)
    if mode=="nearest":
        near=np.argmin(np.abs(np.log1p(HEALTH[:,None])-np.log1p(anc[None])),axis=1)
        return pi[:,index(axis,anc)[near]]
    x=np.log1p(anc); h=np.log1p(HEALTH)
    hi=np.clip(np.searchsorted(x,h),1,len(x)-1);lo=hi-1
    t=np.clip((h-x[lo])/(x[hi]-x[lo]),0.,1.)[None,:,None,None]
    return (1-t)*pi[:,index(axis,anc)[lo]]+t*pi[:,index(axis,anc)[hi]]


def summaries(surface):
    out={}
    for band,idx in BANDS.items():
        v=surface[INTERIOR][:,idx].mean(axis=(0,1))
        for key in METRICS:out[band+"_"+key]=float(v[MI[key]])
        for j in (2,6):
            means=surface[j,idx].mean(axis=0)
            for key in ("answer","safe_task","total_death"):
                out[f"{band}_{key}_q{round(100*QS[j])}"]=float(means[MI[key]])
    out["q_rescue_minus_doom_interaction"]=(out["rescue_answer_q75"]-out["rescue_answer_q35"])-(out["doomed_answer_q75"]-out["doomed_answer_q35"])
    return out


def crossings(surface):
    ix=HEALTH<7.995;h=HEALTH[ix]
    result=[]
    for j,q in enumerate(QS):
        x=surface[j,ix,MI["answer"]]
        changes=np.flatnonzero((x[:-1]<.5)!=(x[1:]<.5))
        up=[float(h[i]+(h[i+1]-h[i])*(.5-x[i])/(x[i+1]-x[i])) for i in changes if x[i+1]>=.5]
        down=[float(h[i]+(h[i+1]-h[i])*(.5-x[i])/(x[i+1]-x[i])) for i in changes if x[i+1]<.5]
        result.append({"q":q,"starts_above":bool(x[0]>=.5),"ends_above":bool(x[-1]>=.5),"up":up,"down":down})
    return result


def seed_eval(seed,root,axis,phys,idx):
    mirrors=[];records=[];calibration=[];cross=[];surrogates=[]
    intrinsic=phys[2]
    for m in (0,1):
        curves={};one={};integ={};sharing={}
        folder=root/f"s{seed}/m{m}"
        for nm in CORE:
            sense,arm=nm.split("-",1)
            file=root/(f"frozen-{sense}.npz") if arm=="frozen" else folder/(nm+"-probe.npz")
            with np.load(file) as z:obs=z["mean"];masked=z["masked"]
            pi=policies(obs,"iid")
            pr=pi[:,idx]
            curve=scores(pr,phys[m],intrinsic)
            curves[nm]=curve;one[nm]=summaries(curve)
            cross.append({"mapping":m,"arm":nm,"points":crossings(curve)})
            if sense=="record" and arm in ("D","R","M","Mshift","frozen"):
                for mode in ("fixed","clean","reversed"):
                    one[nm+"-"+mode]=summaries(scores(policies(obs,mode)[:,idx],phys[m],intrinsic))
                for mode in ("masked",):
                    mp=policies(masked,"iid")
                    one[nm+"-"+mode]=summaries(scores(np.broadcast_to(mp,pr.shape),phys[m],intrinsic))
            if arm in ("M","Mshift"):
                intermediate=BANDS["transition"]
                anchors=levels(arm,seed)
                for how in ("nearest","loglinear"):
                    pred=interpolation(pi,axis,anchors,how)
                    alt=scores(pred,phys[m],intrinsic)
                    diff=alt[INTERIOR][:,intermediate]-curve[INTERIOR][:,intermediate]
                    integ[nm+"-"+how]={
                        "absolute_answer_difference":float(np.abs(diff[...,MI["answer"]]).mean()),
                        "surrogate_minus_real_regret":float(diff[...,MI["reference_regret"]].mean()),
                        "surrogate_minus_real_death":float(diff[...,MI["total_death"]].mean()),
                        "surrogate_minus_real_safe_task":float(diff[...,MI["safe_task"]].mean()),
                        "absolute_probability_difference":float(np.abs(pred[INTERIOR][:,intermediate]-pr[INTERIOR][:,intermediate]).mean())}
            if sense=="plain":
                assert np.array_equal(pr[0],pr[-1])
            sharing[nm]=float(np.max(np.abs(pr[0]-pr[-1])))
        mirrors.append(curves);records.append(one);calibration.append(integ);surrogates.append(sharing)
    def avg(items):return {k:{j:float(np.mean([x[k][j] for x in items])) for j in items[0][k]} for k in items[0]}
    return ({k:(mirrors[0][k]+mirrors[1][k])/2 for k in CORE},avg(records),avg(calibration),cross,surrogates)


@lru_cache
def bootstrap(n):
    return rng(67871,"boundary-cluster-bootstrap",n).integers(n,size=(12000,n))


def summarize(values):
    a=np.asarray(values,dtype=float)
    bs=bootstrap(len(a))
    return {"mean":float(a.mean()),"ci95":np.quantile(a[bs].mean(1),[.025,.975]).tolist(),
            "positive":int((a>0).sum()),"n":len(a),"range":[float(a.min()),float(a.max())]}


def run(root,out,seeds):
    out.mkdir(parents=True,exist_ok=True)
    axis=np.array(json.loads((root/"axes.json").read_bytes())["health"])
    idx=index(axis,HEALTH);body=BodyValue()
    d,v=integration(HEALTH,body);rep,_=integration(HEALTH,body,seed=67091)
    phys=[physics_matrix(d,v,m) for m in (0,1)]
    phys.append(d[:,2,1])
    np.savez_compressed(out/"physical.npz",health=HEALTH,qs=QS,conditional_death=d,second_integration=rep,conditional_rul_change=v)
    records=[];integ=[];raw=[];allcross={};span={}
    for s in seeds:
        curves,r,i,c,z=seed_eval(s,root,axis,phys,idx)
        records.append(r);integ.append(i);raw.append(curves);allcross[str(s)]=c;span[str(s)]=z
    write(out/"per-seed.json",{"seeds":seeds,"metrics":records,"anchor_interpolation":integ})
    write(out/"crossings.json",allcross)
    absolute={k:{j:summarize([v[k][j] for v in records]) for j in records[0][k]} for k in records[0]}
    pairs=[("record-M","record-R"),("record-D","record-R"),("record-M","record-D"),("record-Mshift","record-M"),("record-M","record-frozen"),
           ("plain-M","plain-R"),("plain-D","plain-R"),("plain-D_gets_R","plain-D"),("plain-R_gets_D","plain-R"),
           ("plain-D_gets_R","plain-R"),("plain-D","plain-R_gets_D"),("record-M","record-M-fixed"),("record-M","record-M-reversed"),
           ("record-M","plain-M"),("record-M","record-M-masked")]
    contrasts={f"{left}_minus_{right}":{j:summarize([v[left][j]-v[right][j] for v in records]) for j in records[0][left]} for left,right in pairs}
    interp={k:{j:summarize([v[k][j] for v in integ]) for j in integ[0][k]} for k in integ[0]}
    body_pair=contrasts["record-M_minus_record-R"]
    fat=summarize([v["record-R"]["doomed_intrinsic_fatal_answer"]-v["record-M"]["doomed_intrinsic_fatal_answer"] for v in records])
    death=summarize([v["record-R"]["doomed_total_death"]-v["record-M"]["doomed_total_death"] for v in records])
    social=contrasts["record-M_minus_record-frozen"]["q_rescue_minus_doom_interaction"]
    body_pass=(fat["mean"]>=.1 and fat["ci95"][0]>0 and fat["positive"]>=np.ceil(.8*len(seeds)) and death["ci95"][0]>0
               and min(body_pair["rescue_safe_task"]["mean"],body_pair["healthy_safe_task"]["mean"])>=-.05)
    q_pass=social["mean"]>=.05 and social["ci95"][0]>0
    surf={k:np.stack([r[k] for r in raw]) for k in CORE}
    np.savez_compressed(out/"seed-surfaces.npz",health=HEALTH,q=QS,seed=np.array(seeds),metric=np.array(METRICS),**surf)
    with gzip.open(out/"pointwise.csv.gz","wt",encoding="utf8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["arm","q_true","health",*METRICS])
        for name,z in surf.items():
            mean=z.mean(0)
            for qi,q in enumerate(QS):
                for hi,h in enumerate(HEALTH):writer.writerow([name,q,h,*mean[qi,hi]])
    report={"seeds":seeds,"absolute":absolute,"paired":contrasts,"interpolation":interp,
            "gates":{"bodily":bool(body_pass),"novel_partner_probability":bool(q_pass)},
            "fatal_reduction":fat,"total_death_reduction":death,"q_interaction_over_frozen":social,
            "max_second_quadrature_disagreement":float(np.abs(rep-d).max()),
            "plain_max_behavior_change_from_unobserved_q":max(float(w[k]) for s in span.values() for w in s for k in CORE if k.startswith("plain-"))}
    write(out/"summary.json",report)
    print(json.dumps({"seeds":len(seeds),"gates":report["gates"],"fatal":fat,"death":death,"social":social}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wave",choices=("pilot","confirm"),required=True)
    a=p.parse_args();run(a.root,a.output,list(PILOT if a.wave=="pilot" else CONFIRM))
