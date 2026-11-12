"""Train exactly one decision-token intervention inside the real Qwen forward graph."""
from collections import defaultdict
import json
import time

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.environment import draw_tape, settled_tape
from experiments.consequence_policy.experiment import evaluation_reference
from experiments.consequence_policy.policy import deployed, feedback_scores
from experiments.consequence_policy.study import KL, TEMP, TEST_H, basis
from experiments.direct_policy.protocol import rng, write
from experiments.internal_body.capture import encode, grouped
from experiments.internal_body.design import LR_A, LR_B, RANK, SPLIT_H, all_rows
from experiments.internal_body.features import linear_projection
from experiments.internal_body.suffix import distribution
from experiments.social_closed_loop.core import BodyValue

SITES={"site3":(3,3),"site15":(3,15),"site31":(3,31),"local31":(31,31)}
LEVELS=(2.,6.,90.,300.)
NSTEP=32


def panel(rows):
    return np.array([i for i,r in enumerate(rows) if r["split"]=="test" and
       r["frame"]==(r["partner"]+r["layout"])%2 and r["task"]==4+(r["layout"]//2)%2])


def schedule(seed,prefixes):
    gen=rng(seed,"actual-depth-batches")
    groups=defaultdict(list)
    for i in range(768):groups[len(prefixes[i])].append(i)
    batches=[]
    for _ in range(NSTEP):
        width=len(prefixes[int(gen.integers(768))])
        batches.append((gen.choice(groups[width],8,replace=False),gen.permutation(8)))
    return batches


def forward(actor,prefixes,indices):
    ids=torch.as_tensor([prefixes[i] for i in indices],device="cuda")
    out=actor.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
    return out.logits[:,-1,actor.ids].float()


def calibrate(actor,indices,prefixes,out):
    lm=actor.model.model.language_model
    captured={}
    def capture(site):
        def pre(module,args):captured[site]=args[0][:,-1].detach().float().cpu().numpy()
        return pre
    hooks=[lm.layers[s].mlp.down_proj.register_forward_pre_hook(capture(s)) for s in (3,31)]
    raw={};native=np.zeros((len(prefixes),4),np.float32)
    try:
        with torch.inference_mode():
            for batch in grouped(indices,prefixes,4):
                native[batch]=forward(actor,prefixes,batch).cpu().numpy()
                for s,v in captured.items():
                    if s not in raw:raw[s]=np.zeros((768,v.shape[-1]),np.float32)
                    for j,i in enumerate(batch):
                        if i<768:raw[s][i]=v[j]
    finally:
        for h in hooks:h.remove()
    proj={s:linear_projection(raw[s],np.ones(768,dtype=bool)) for s in raw}
    np.savez_compressed(out/"calibration.npz",native=native,layer3=proj[3],layer31=proj[31])
    return native,proj


class Site(nn.Module):
    def __init__(self,actor,site,projection):
        super().__init__()
        source,dest=SITES[site]
        lm=actor.model.model.language_model
        self.source,self.dest=source,dest
        self.register_buffer("projection",torch.tensor(projection,device="cuda"))
        self.A=nn.Parameter(torch.tensor(np.random.default_rng(64800).normal(0,.06,(2,12,33,RANK)),dtype=torch.float32,device="cuda"))
        self.B=nn.Parameter(torch.zeros((RANK,lm.layers[dest].mlp.down_proj.out_features),device="cuda"))
        self.enabled=True
        def read(module,args):self.x=args[0][:,-1].detach().float()@self.projection
        def inject(module,args,original):
            if not self.enabled:return original
            t=torch.einsum("nb,nbdr->ndr",self.g,self.A[self.p])
            c=torch.einsum("nd,ndr->nr",self.x,t)
            d=c@self.B
            self.rms=float(d.detach().square().mean().sqrt())
            return torch.cat((original[:,:-1],(original[:,-1].float()+d).to(original.dtype)[:,None]),dim=1)
        down=lm.layers[dest].mlp.down_proj
        down.add_module("body_residual",self)
        self.handles=(lm.layers[source].mlp.down_proj.register_forward_pre_hook(read),down.register_forward_hook(inject))

    def condition(self,partners,health):
        self.g=torch.as_tensor(basis(health),dtype=torch.float32,device="cuda")
        self.p=torch.as_tensor(partners,dtype=torch.long,device="cuda")

    def close(self,actor):
        for h in self.handles:h.remove()
        delattr(actor.model.model.language_model.layers[self.dest].mlp.down_proj,"body_residual")


def evaluate(actor,adapter,prefixes,rows,indices):
    pi=np.zeros((len(LEVELS),len(indices),4),np.float32)
    key={i:j for j,i in enumerate(indices)}
    with torch.no_grad():
        for j,h in enumerate(LEVELS):
            for batch in grouped(indices,prefixes,4):
                adapter.condition([rows[i]["partner"] for i in batch],[h]*len(batch))
                pi[j,[key[i] for i in batch]]=distribution(forward(actor,prefixes,batch)).cpu().numpy()
    return pi


def summary(pi,ref):
    risk,_,style=ref
    ans=pi[:,:,2:].sum(-1)
    preferred=pi[:,np.arange(len(style)),style]
    death=(pi*risk[[TEST_H.index(h) for h in LEVELS]]).sum(-1)
    return dict(danger=float(ans[0].mean()),healthy=float(ans[2:].mean()),style=float(preferred[2:].mean()),
                death=death.mean(axis=1).tolist(),actions=pi.mean(axis=1).tolist())


def train(actor,adapter,prefixes,rows,native,scores,batches):
    optim=torch.optim.Adam([dict(params=[adapter.A],lr=LR_A),dict(params=[adapter.B],lr=LR_B)])
    hist=[]
    for step,(contexts,hs) in enumerate(batches):
        optim.zero_grad(set_to_none=True)
        chunks=[]
        for off in (0,4):
            ix=contexts[off:off+4];levels=hs[off:off+4]
            adapter.condition([rows[i]["partner"] for i in ix],[SPLIT_H[h] for h in levels])
            pi=distribution(forward(actor,prefixes,ix))
            frozen=torch.tensor(deployed(native[ix]/TEMP),device="cuda")
            feedback=torch.tensor(scores[levels,ix],device="cuda")
            gain=(pi*feedback).sum(-1).mean()
            kl=(pi*(pi/frozen).log()).sum(-1).mean()
            loss=-gain+KL*kl
            (loss/2).backward()
            chunks.append([float(loss.detach()),float(gain.detach()),float(kl.detach()),adapter.rms])
        norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
        optim.step()
        hist.append([step+1,*np.mean(chunks,axis=0).tolist(),float(norm)])
    return hist


def run(actor,root,out):
    job=json.loads((root/"depth-job.json").read_bytes())
    rows=all_rows()[:960];ids=[encode(actor,r) for r in rows]
    tested=panel(rows)
    assert len(tested)==48
    native,proj=calibrate(actor,list(range(768))+tested.tolist(),ids,out)
    write(out/"panel.json",dict(indices=tested.tolist(),levels=LEVELS,input_ids=ids,sites=SITES))
    body=BodyValue()
    refs=[evaluation_reference(body,[rows[i] for i in tested],m) for m in (0,1)]
    frozen=np.broadcast_to(deployed(native[tested]/TEMP),(len(LEVELS),len(tested),4)).copy()
    receipt=[]
    for seed in job["seeds"]:
        choice,mu,u,noise=draw_tape(seed,native[:768])
        np.savez_compressed(out/f"tape-{seed}.npz",choice=choice,mu=mu,response_uniform=u,noise=noise)
        batches=schedule(seed,ids)
        write(out/f"schedule-{seed}.json",[[x.tolist(),h.tolist()] for x,h in batches])
        for m in (0,1):
            folder=out/f"s{seed}-m{m}";folder.mkdir(exist_ok=True)
            after,reward,response=settled_tape(choice,u,noise,rows[:768],m,body,SPLIT_H)
            scores=feedback_scores(choice,reward,mu,np.minimum(1.,[body.rul(h) for h in SPLIT_H]))
            np.savez_compressed(folder/"observed.npz",after=after,reward=reward,response=response)
            report={"frozen":summary(frozen,refs[m])}
            for name in job.get("sites",list(SITES)):
                write(out/"status.json",dict(stage="direct_training",seed=seed,mapping=m,site=name))
                site=Site(actor,name,proj[SITES[name][0]])
                try:
                    tic=time.monotonic()
                    initA=site.A.detach().cpu().numpy().copy()
                    hist=train(actor,site,ids,rows,native,scores,batches)
                    pi=evaluate(actor,site,ids,rows,tested)
                    a,b=(t.detach().cpu().numpy() for t in (site.A,site.B))
                    # Disabled check uses identical four-row call shape and frozen native action tokens.
                    site.enabled=False
                    check_batch=next(grouped(tested,ids,4))
                    with torch.no_grad():restored=forward(actor,ids,check_batch).cpu().numpy()
                    restore_err=float(np.abs(restored-native[check_batch]).max())
                    np.savez_compressed(folder/(name+".npz"),A=a,B=b,probes=pi)
                    report[name]=dict(**summary(pi,refs[m]),seconds=time.monotonic()-tic,
                        state_A_change_l2=float(np.linalg.norm(a-initA)),state_B_l2=float(np.linalg.norm(b)),
                        parameter_names=[n for n,p in actor.model.named_parameters() if p.requires_grad],
                        disabled_frozen_max_logit_gap=restore_err,steps=hist)
                finally:
                    site.close(actor)
                write(folder/"metrics.json",report)
            receipt.append([seed,m])
    return dict(completed=receipt,sites=job.get("sites",list(SITES)),microbatch=4,steps=NSTEP)
