"""Differentiable native Qwen final MLP/residual/RMSNorm suffix for a fixed prefix.

The base score anchors the BF16 forward. Changes are computed in FP32 and checked
against actual whole-model installations. No linearization of RMSNorm is used.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.study import EXPLORATION, KL, TEMP, basis
from experiments.internal_body.design import LR_A, LR_B, LR_NATIVE, RANK, SPLIT_H, STEPS, prompt_levels

KINDS = ("linear","nonlinear","postnorm","blind","no_route","shuffle","zero","maintain","high",
         "prompt_lora","native_prompt","native_hidden","generic")


def arr(x,device):
    return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device=device)


class StateBank:
    def __init__(self,folder,device="cpu"):
        self.device=device
        folder=Path(folder)
        with np.load(folder/"native-state.npz") as z:
            self.r=z["r"];self.z=z["z"];self.x=z["features"];self.native=z["native"]
        with np.load(folder/"architecture.npz") as z:
            self.norm_np=z["norm"];self.head_np=z["head"]
        self.cfg=json.loads((folder/"architecture.json").read_bytes())
        self.rows=json.loads((folder/"rows.json").read_bytes())
        self.partners=np.array([r["partner"] for r in self.rows])
        self.norm=arr(self.norm_np,device)
        self.head=arr(self.head_np,device)
        self.t=(self.norm+self.cfg["norm_offset"])[:,None]*self.head.T
        self.prompts={}
        for j,h in enumerate(prompt_levels()):
            with np.load(folder/f"prompt-{j:02}.npz") as z:
                self.prompts[h]=(z["index"],z["r"],z["features"],z["native"])
        self.grids={}

    def grid(self,indices,levels,text=False):
        indices=np.asarray(indices,dtype=int)
        key=(tuple(indices.tolist()),tuple(levels),text)
        if key in self.grids:return self.grids[key]
        if not text:
            r,x,n=(a[indices][None] for a in (self.r,self.x,self.native))
        else:
            states=[]
            for h in levels:
                idx,r0,x0,n0=self.prompts[h]
                j=np.searchsorted(idx,indices)
                assert np.array_equal(idx[j],indices)
                states.append((r0[j],x0[j],n0[j]))
            r,x,n=[np.stack(s) for s in zip(*states)]
        r,x,n=[arr(a,self.device) for a in (r,x,n)]
        square=r.square().mean(-1,keepdim=True)
        div=(square+self.cfg["epsilon"]).sqrt()
        result={"r":r,"x":x,"n":n,"sq":square,"div":div,"dot":r@self.t,
                "g":arr(basis(levels),self.device),"partner":torch.as_tensor(self.partners[indices],device=self.device)}
        self.grids[key]=result
        return result


class InternalPolicy(nn.Module):
    def __init__(self,bank,kind):
        super().__init__()
        if kind not in KINDS:raise ValueError(kind)
        self.kind=kind
        self.register_buffer("head",bank.head)
        self.register_buffer("weighted",bank.t)
        self.register_buffer("norm0",bank.norm)
        self.register_buffer("neutral",arr(basis(SPLIT_H).mean(axis=0),bank.device))
        self.eps=bank.cfg["epsilon"]
        if self.is_native:
            self.delta=nn.Parameter(torch.zeros_like(bank.norm))
        else:
            initial=np.random.default_rng(64800).normal(0,.06,size=(2,12,bank.x.shape[1],RANK))
            self.A=nn.Parameter(arr(initial,bank.device))
            self.B=nn.Parameter(torch.zeros((RANK,len(bank.norm)),device=bank.device))

    @property
    def is_native(self):
        return self.kind.startswith("native_")

    @property
    def uses_text(self):
        return self.kind in ("prompt_lora","native_prompt")

    def coefficients(self,x,g,p,masked=False,pooled=False):
        if masked or self.kind in ("blind","prompt_lora"):
            g=self.neutral.expand(len(g),-1)
        matrices=self.A
        if pooled or self.kind=="no_route":
            matrices=matrices.mean(dim=0,keepdim=True).expand(2,-1,-1,-1)
        table=torch.einsum("hb,pbdr->phdr",g,matrices)
        # Partners have the same feature and health calculations; no ecology enters the actor.
        a0=torch.einsum("hnd,hdr->hnr",x.expand(len(g),-1,-1),table[0])
        a1=torch.einsum("hnd,hdr->hnr",x.expand(len(g),-1,-1),table[1])
        c=torch.where((p%2)[None,:,None].bool(),a1,a0)
        return c.tanh() if self.kind=="nonlinear" else c

    def scores(self,grid,masked=False,pooled=False):
        g,n=grid["g"],grid["n"]
        if self.is_native:
            change=(self.norm0+self.delta).to(torch.bfloat16).float()-self.norm0
            return (n+((grid["r"]/grid["div"])*change)@self.head.T).expand(len(g),-1,-1)
        c=self.coefficients(grid["x"],g,grid["partner"],masked,pooled)
        delta_dot=c@(self.B@self.weighted)
        if self.kind=="postnorm":
            change=delta_dot/grid["div"]
        else:
            cross=(c*(grid["r"]@self.B.T)).sum(-1,keepdim=True)
            delta2=(c*(c@(self.B@self.B.T))).sum(-1,keepdim=True)
            square=grid["sq"]+(2*cross+delta2)/self.B.shape[1]
            div=(square.clamp_min(0)+self.eps).sqrt()
            change=(grid["dot"]+delta_dot)/div-grid["dot"]/grid["div"]
        return n+change

    def save(self,path):
        np.savez_compressed(path,kind=np.array(self.kind),**{name:p.detach().cpu().numpy() for name,p in self.named_parameters()})

    @classmethod
    def load(cls,bank,path):
        with np.load(path) as data:
            model=cls(bank,str(data["kind"]))
            with torch.no_grad():
                for name,p in model.named_parameters():p.copy_(arr(data[name],bank.device))
        return model


def distribution(scores):
    return (1-EXPLORATION)*torch.softmax(scores/TEMP,dim=-1)+EXPLORATION/4


def fit(model,grid,consequences,steps=STEPS):
    if model.is_native:
        groups=[dict(params=[model.delta],lr=LR_NATIVE)]
    else:
        groups=[dict(params=[model.A],lr=LR_A),dict(params=[model.B],lr=LR_B)]
    optimizer=torch.optim.Adam(groups)
    baseline=distribution(grid["n"]).detach()
    observed=arr(consequences,baseline.device)
    history=[]
    for t in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pi=distribution(model.scores(grid))
        gain=(pi*observed).sum(-1).mean()
        kl=(pi*(pi/baseline).log()).sum(-1).mean()
        loss=-gain+KL*kl
        loss.backward()
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
        assert torch.isfinite(norm)
        optimizer.step()
        history.append([t+1,float(loss.detach()),float(gain.detach()),float(kl.detach()),float(norm)])
    return history
