"""Frozen Qwen acquisitions and independent installations of actual learned parameters."""
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.consequence_policy.study import WORDS, contexts
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import DecisionState, grouped
from experiments.lethal_boundary.design import prompt


def encode(actor,row,count):
    text=actor.tok.apply_chat_template(prompt(row,count),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=actor.tok.encode(text,add_special_tokens=False)
    assert len(ids)<512 and all(actor.tok.encode(text+w,add_special_tokens=False)==ids+[k] for w,k in zip(WORDS,actor.ids))
    return ids


def pack_bf16(x):
    bits=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    assert not (bits & 65535).any()
    return (bits>>16).astype(np.uint16)


def capture(actor,root,out):
    rows=contexts()
    projection=np.load(root/"architecture.npz")["projection"]
    state=DecisionState(actor)
    lm=actor.model.model.language_model
    parts={}
    def skip(_,args):
        parts["skip"]=args[0][:,-1].detach()
    def down(_,args,val):
        parts["down"]=val[:,-1].detach()
    hs=lm.layers[-1].post_attention_layernorm.register_forward_pre_hook(skip)
    hd=lm.layers[-1].mlp.down_proj.register_forward_hook(down)
    tokens={}; metadata=[]
    try:
        for count in (None,*range(21)):
            include_train=count in (None,5,17)
            ids=range(960) if include_train else range(768,960)
            ix=np.asarray(list(ids))
            prefixes={i:encode(actor,rows[i],count) for i in ids}
            name="plain" if count is None else f"k{count:02d}"
            tokens[name]=[prefixes[i] for i in ix]
            rr=np.empty((len(ix),2560),np.uint16)
            native=np.empty((len(ix),4),np.float32)
            x=np.empty((len(ix),projection.shape[1]),np.float32)
            detail={v:np.empty((192,2560),np.uint16) for v in ("skip","down","z")}
            for batch in grouped(ids,prefixes,8):
                vv=state.batch([prefixes[i] for i in batch])
                ii=np.asarray(batch);loc=np.searchsorted(ix,ii)
                rr[loc]=pack_bf16(vv["r"])
                x[loc]=vv["v"]@projection
                native[loc]=vv["native"]
                held=ii>=768
                if held.any():
                    for k in detail:
                        d=vv["z"] if k=="z" else parts[k].float().cpu().numpy()
                        detail[k][ii[held]-768]=pack_bf16(d[held])
            np.savez_compressed(out/(name+".npz"),idx=ix,r=rr,x=x,n=native,**detail)
            metadata.append({"name":name,"contexts":len(ix),"min_tokens":min(map(len,prefixes.values())),"max_tokens":max(map(len,prefixes.values()))})
            write(out/"status.json",{"stage":"capture","completed":len(metadata),"last":name})
    finally:
        hs.remove();hd.remove();state.close()
    write(out/"tokens.json",tokens)
    write(out/"contexts.json",rows)
    return {"panels":metadata,"record_past_only":True,"frozen_prefixes":sum(v["contexts"] for v in metadata)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",required=True)
    a=p.parse_args()
    out=a.root/"output";out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    actor=FrozenQwen(a.root/"expected-weights.json",out)
    if a.mode=="boundarycapture":
        result=capture(actor,a.root,out)
    elif a.mode=="boundaryaudit":
        from experiments.lethal_boundary.hardware import run
        result=run(actor,a.root,out)
    else:raise ValueError(a.mode)
    actor.verify_unchanged()
    write(out/"receipt.json",{"mode":a.mode,"result":result,"elapsed_seconds":time.monotonic()-started,
        "peak_bytes":torch.cuda.max_memory_allocated(),"backbone_restored":True,"model_shards_verified":True,
        "native_ids":actor.ids,"device":torch.cuda.get_device_name(),"versions":{k:version(k) for k in ("torch","transformers","numpy","scipy")}})
    write(out/"status.json",{"stage":"complete"})


if __name__=="__main__":main()
