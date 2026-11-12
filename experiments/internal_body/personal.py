"""Within-life gradient updates in the full native hidden space, without ecology labels."""
from pathlib import Path
import numpy as np

from experiments.consequence_policy.policy import deployed, softmax
from experiments.consequence_policy.study import EXPLORATION, TEMP, basis
from experiments.internal_body.design import SPLIT_H

ARMS=("fixed","free","tied","decision_key","flat_personal","experience_only","scratch","supplied")
FREE=set(ARMS)-{"fixed","supplied"}


def normalized(a):
    return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-10)


def language_key(all_values,train):
    x=all_values[train].astype(np.float64)
    mean=x.mean(0);xc=x-mean
    eig,u=np.linalg.eigh(xc@xc.T/len(xc))
    j=np.argsort(eig)[-4:][::-1]
    assert eig[j].min()>0
    v=xc.T@u[:,j]/(np.sqrt(len(xc))*eig[j])
    coords=(all_values.astype(np.float64)-mean)@v
    return normalized(np.column_stack((np.ones(len(coords)),coords))).astype(np.float32)


def personal_gate(health):
    # Smooth radial basis only, with no prescribed action association.
    d=np.log1p(np.asarray(health))[...,None]-np.log1p([2.,20.,200.])
    return normalized(np.exp(-.5*(d/1.35)**2)).astype(np.float32)


class LifeInputs:
    def __init__(self,cache):
        import json
        self.rows=json.loads((cache/"rows.json").read_bytes())
        with np.load(cache/"native-state.npz") as z:
            self.r=z["r"].astype(np.float32);self.n=z["native"];self.x=z["features"]
            self.decision=language_key(z["z"],np.arange(768))
        with np.load(cache/"request-only.npz") as z:
            idx={str(t):j for j,t in enumerate(z["text"])}
            lookup=np.array([idx[r["text"]] for r in self.rows])
            latent=language_key(z["hidden"],np.unique(lookup[:768]))
            self.semantic=latent[lookup]
        with np.load(cache/"architecture.npz") as z:
            cfg=json.loads((cache/"architecture.json").read_bytes());self.eps=cfg["epsilon"]
            self.t=(z["norm"]+cfg["norm_offset"])[:,None]*z["head"].T
        self.scale0=np.sqrt(np.mean(self.r**2,axis=1,keepdims=True)+self.eps)
        self.base_dots=(self.r/self.scale0)@self.t
        self.slot=np.array([r["partner"]%2 for r in self.rows])
        self.neutral12=basis(SPLIT_H).mean(axis=0).astype(np.float32)
        self.neutral3=normalized(personal_gate(SPLIT_H).mean(axis=0)).astype(np.float32)


def read_internal(path):
    with np.load(path) as z:return z["A"],z["B"]


class LifeActor:
    def __init__(self,data,checkpoints,seed,arm):
        self.data,self.arm=data,arm
        a,self.B=read_internal(checkpoints/f"generic-{seed}.npz")
        self.A=a.mean(0)
        self.W=np.zeros((1 if arm=="tied" else 2,3,5,self.B.shape[1]),dtype=np.float32) if arm in FREE else None
        self.u=np.zeros(2,np.float64)
        self.direction=None
        if arm=="supplied":
            self.direction=[read_internal(checkpoints/f"s{seed}-m{m}/linear.npz") for m in (0,1)]
        self.keys=data.decision if arm=="decision_key" else data.semantic

    def gates(self,h):
        bg=self.data.neutral12 if self.arm=="experience_only" else basis([h])[0].astype(np.float32)
        pg=self.data.neutral3 if self.arm in ("experience_only","flat_personal") else personal_gate(h)
        return bg,pg

    def components(self,indices,h):
        z=self.data;i=np.atleast_1d(indices);p=z.slot[i]
        bg,pg=self.gates(h)
        feat=z.x[i]
        if self.arm=="scratch":delta=np.zeros_like(z.r[i])
        else:delta=(feat@np.einsum("b,bdr->dr",bg,self.A))@self.B
        direction=None
        if self.direction is not None:
            parts=[]
            for A,B in self.direction:
                T=np.einsum("b,pbdr->pdr",bg,A)
                parts.append(np.einsum("nd,ndr->nr",feat,T[p])@B)
            direction=(parts[0]-parts[1])/2
            delta=delta+self.u[p,None].astype(np.float32)*direction
        if self.W is not None:
            # One persisted slot for tied weights; otherwise a new partner gets its own zero matrix.
            blocks=np.einsum("b,pbkd->pkd",pg,self.W)
            if len(i)==1:
                delta=delta+self.keys[i]@blocks[0 if self.arm=="tied" else p[0]]
            else:
                for q in (0,1):
                    mask=p==q
                    if mask.any():delta[mask]+=self.keys[i[mask]]@blocks[0 if self.arm=="tied" else q]
        return z.r[i]+delta,direction,pg

    def policy(self,indices,h):
        z=self.data;i=np.atleast_1d(indices)
        r,d,pg=self.components(i,h)
        den=np.sqrt(np.mean(r*r,axis=-1,keepdims=True)+z.eps)
        scores=z.n[i]+(r/den)@z.t-z.base_dots[i]
        return deployed(scores/TEMP),softmax(scores/TEMP),r,d,den,pg

    def update(self,index,action,normalized_reward,details):
        if self.arm=="fixed":return 0.,0.
        z=self.data;p=z.slot[index];pi,s,r,d,den,pg=details
        pi,s,r,den=pi[0],s[0],r[0],float(den[0,0])
        jac=(np.eye(4,dtype=np.float32)-s)*(1-EXPLORATION)*s[:,None]/pi[:,None]/TEMP
        v=jac@z.t.T
        gradient=v/den-(v@r)[:,None]*r/(len(r)*den**3)
        # Logged NumPy scalars and online Python floats must take the same FP32 update path.
        advantage=float(normalized_reward)+.9
        if self.arm=="supplied":
            scalar=gradient@d[0]
            fisher=float(pi@(scalar*scalar))
            self.u[p]=np.clip(self.u[p]+.5*advantage*scalar[action]/(1.+fisher),-1.,1.)
            return float(abs(scalar[action])),fisher
        fisher=float(pi@np.sum(gradient**2,axis=1))
        step=.55*advantage/(.05+fisher)
        slot=0 if self.arm=="tied" else p
        self.W[slot]+=np.einsum("b,k,d->bkd",pg,self.keys[index],gradient[action])*step
        return float(np.linalg.norm(gradient[action])),fisher

    def parameters(self):
        return self.W if self.W is not None else self.u
