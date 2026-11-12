"""Private terminal LoRA and two genuine Qwen execution paths."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.study import basis
from experiments.individual_life.design import CAP,NEUTRAL,RATE,RIDGE,rows,starting
from experiments.internal_body.suffix import distribution
from experiments.lethal_boundary.states import expand_bf16


class Individual(nn.Module):
    def __init__(self,source,birth,device):
        super().__init__()
        A,B=starting(source,birth)
        self.A=nn.Parameter(torch.tensor(A,device=device));self.B=nn.Parameter(torch.tensor(B,device=device))

    def live(self):
        return self.A,self.B

    def snapshot(self):
        return self.A.detach().clone(),self.B.detach().clone()

    def bits(self):
        return np.concatenate([v.detach().cpu().numpy().ravel() for v in self.live()]).astype("<f4",copy=False).view("<u4").copy()


def correction(x,gate,slot,weights):
    a=torch.stack([pair[0][slot] for pair in weights])
    b=torch.stack([pair[1] for pair in weights])
    table=torch.einsum("h,ihdr->idr",gate,a)
    coef=torch.einsum("id,idr->ir",x,table)
    return torch.einsum("ir,ird->id",coef,b)


class Native:
    def __init__(self,cache,arch,device="cpu"):
        self.device=device
        records=json.loads((Path(cache)/"requests.json").read_bytes());self.rows=rows()
        assert len(records)==len(self.rows) and all(all(r[k]==s[k] for k in s) for r,s in zip(records,self.rows))
        self.ids=[r["ids"] for r in records]; self.slot=np.array([r["partner"]-2 for r in records])
        self.tensor=lambda x:torch.as_tensor(x,device=device)
        c=json.loads((Path(arch)/"architecture.json").read_bytes())
        self.eps=c["epsilon"]
        with np.load(Path(arch)/"architecture.npz") as weights:
            self.head=self.tensor(weights["head"]);self.scale=self.tensor(weights["norm"]+c["norm_offset"])
        with np.load(Path(cache)/"prefix.npz") as z:
            self.x=self.tensor(z["x"]);self.n=self.tensor(z["n"])
            self.down=self.tensor(expand_bf16(z["down"]))
            self.skip=self.tensor(expand_bf16(z["skip"])).bfloat16()
            self.z=self.tensor(expand_bf16(z["z"]))
            assert torch.equal((self.skip+self.down.bfloat16()).float(),self.tensor(expand_bf16(z["r"])))
        self.base_words=self.words(self.z)
        self.probe_ix=np.array([i for i,r in enumerate(self.rows) if (r["task"],r["wording"])==(5,4)])
        assert len(self.probe_ix)==96
        self.source_birth=Path(arch)/"birth-source.npz"

    def new(self,birth):
        return Individual(self.source_birth,birth,self.device)

    def gate(self,health,visible):
        return self.tensor(np.asarray(basis([health])[0] if visible else NEUTRAL,dtype=np.float32))

    def words(self,z):
        return (z@self.head.T).bfloat16().float()

    def decide(self,i,delta):
        r=(self.skip[i]+(self.down[i]+delta).bfloat16()).float()
        z=(r*torch.rsqrt(r.square().mean(-1,keepdim=True)+self.eps)*self.scale).bfloat16().float()
        return distribution(self.n[i]+(self.words(z)-self.base_words[i]))

    def panel(self,i,health,visible,weights):
        x=self.x[i].expand(len(weights),-1)
        return self.decide(i,correction(x,self.gate(health,visible),int(self.slot[i]),weights))

    def probe(self,model,visible,health):
        i=self.probe_ix
        g=self.gate(health,visible)
        T=torch.einsum("b,pbdr->pdr",g,model.A)
        c=torch.einsum("nd,ndr->nr",self.x[i],T[self.tensor(self.slot[i])])
        with torch.no_grad():return self.decide(i,c@model.B).detach().cpu().numpy()

    def bind(self,model,shadow):
        pass

    def close(self):
        pass


class Complete:
    """Install the private A/B on the original final down projection on every living request."""
    def __init__(self,actor,native,arch):
        self.actor=actor;self.native=native;self.rows=native.rows;self.device="cuda"
        self.proj=torch.tensor(np.load(Path(arch)/"architecture.npz")["projection"],device="cuda")
        self.layer=actor.model.model.language_model.layers[-1].mlp.down_proj
        self.handle=None;self.probe_ix=native.probe_ix

    def new(self,birth):
        return self.native.new(birth)

    def bind(self,model,shadow):
        assert self.handle is None
        self.layer.add_module("individual_life_lora",model)
        if shadow is not None:self.layer.add_module("same_tape_control_lora",shadow)
        def inject(_,args,value):
            x=args[0][:,-1].detach().float()@self.proj
            delta=correction(x,self.g,self.partner,self.weights)
            return torch.cat([value[:,:-1],(value[:,-1].float()+delta).to(value.dtype)[:,None]],dim=1)
        self.handle=self.layer.register_forward_hook(inject)

    def close(self):
        self.handle.remove();self.handle=None;self.weights=None
        delattr(self.layer,"individual_life_lora")
        if hasattr(self.layer,"same_tape_control_lora"):delattr(self.layer,"same_tape_control_lora")

    def panel(self,i,health,visible,weights):
        self.g=self.native.gate(health,visible);self.partner=int(self.native.slot[i]);self.weights=weights
        token=torch.tensor([self.native.ids[i]]*len(weights),device="cuda")
        value=self.actor.model(input_ids=token,attention_mask=torch.ones_like(token),use_cache=False,logits_to_keep=1)
        return distribution(value.logits[:,-1,self.actor.ids].float())

    def probe(self,model,visible,health):
        return self.native.probe(model,visible,health)


def derivatives(panel,requests):
    all_grads=[]
    for i,(row,action,model) in enumerate(requests):
        gg=torch.autograd.grad(panel[row,action].log(),model.live(),retain_graph=i+1<len(requests))
        all_grads.append(tuple(g.detach() for g in gg))
    return all_grads


def update(model,grads,signal,behavior_prob,current_prob):
    ratio=float(current_prob/behavior_prob);used=float(np.clip(ratio,.5,2.))
    norms=[float(torch.linalg.vector_norm(g).item()) for g in grads]
    energy=sum(n*n for n in norms)
    scale=RATE*float(signal)*used/(RIDGE+energy)
    size=abs(scale)*np.sqrt(energy)
    if size>CAP:scale*=CAP/size
    with torch.no_grad():
        for p,g in zip(model.live(),grads,strict=True):p.add_(g,alpha=float(scale))
    return {"behavior_probability":float(behavior_prob),"training_probability":float(current_prob),"importance_ratio":ratio,
            "applied_ratio":used,"A_gradient_norm":norms[0],"B_gradient_norm":norms[1],"parameter_multiplier":float(scale),
            "nominal_step_norm":float(abs(scale)*np.sqrt(energy))}
