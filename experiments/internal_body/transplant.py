"""Independent full-Qwen deployment and physical same-parameter relocation."""
import json
import numpy as np
import torch
from torch import nn
from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import encode,grouped
from experiments.internal_body.depth import LEVELS,forward,panel
from experiments.internal_body.design import all_rows
from experiments.internal_body.jacobians import LAYERS
from experiments.internal_body.placement import CAP,SEEDS
from experiments.internal_body.suffix import distribution
from hashlib import sha256


def deploy(actor,root,out):
    rows=all_rows()[:960]
    ids=[encode(actor,r) for r in rows]
    indices=panel(rows)
    batches=list(grouped(indices,ids,4))
    with np.load(root/"tangent-common.npz") as z:
        projection=torch.tensor(z["projection"],device="cuda")
        norms=z["residual_norm"]
    lm=actor.model.model.language_model
    frozen=np.zeros((len(indices),4),np.float32)
    lookup={i:j for j,i in enumerate(indices)}
    with torch.no_grad():
        for batch in batches:
            frozen[[lookup[i] for i in batch]]=distribution(forward(actor,ids,batch)).cpu().numpy()
    np.savez_compressed(out/"frozen.npz",indices=indices,prob=frozen)
    plan=json.loads((root/"transplant-job.json").read_bytes())
    receipt=[]
    for seed in plan["seeds"]:
        for m in (0,1):
            folder=out/f"s{seed}-m{m}";folder.mkdir(exist_ok=True)
            records={};probes={};known={}
            for donor in LAYERS:
                signals=("real","shuffle") if seed in SEEDS[:4] else ("real",)
                for signal in signals:
                    with np.load(root/"weights"/f"s{seed}-m{m}"/f"l{donor}-{signal}.npz") as z:
                        block=nn.Module()
                        block.register_parameter("A",nn.Parameter(torch.tensor(z["A"],device="cuda"),requires_grad=False))
                        block.register_parameter("B",nn.Parameter(torch.tensor(z["B"],device="cuda"),requires_grad=False))
                    settings=[(donor,1.,LEVELS)]
                    if seed in SEEDS[:4] and signal=="real":
                        settings+=[(l,1.,(2.,300.)) for l in LAYERS if l!=donor]
                        settings.append((donor,.5,(2.,300.)))
                    for recv,strength,levels in settings:
                        name=f"d{donor}-r{recv}-{signal}-x{strength:g}"
                        down=lm.layers[recv].mlp.down_proj
                        down.add_module("conditional_residual",block)
                        context={};hashes={}
                        def read(_,args):
                            context["feature"]=args[0][:,-1].float()@projection
                        def inject(_,args,original):
                            f=context["feature"];p=context["p"];g=context["g"]
                            c=torch.einsum("nd,ndr->nr",f,torch.einsum("b,nbdr->ndr",g,block.A[p]))
                            d=c@block.B
                            length=torch.linalg.vector_norm(d,dim=-1).clamp_min(1e-8)
                            factor=(CAP*context["donor_norm"]/length).clamp(max=1.)
                            actual=d*(strength*factor[:,None])
                            context["applied"]=actual
                            return torch.cat((original[:,:-1],(original[:,-1].float()+actual).to(original.dtype)[:,None]),dim=1)
                        h1=lm.layers[3].mlp.down_proj.register_forward_pre_hook(read)
                        h2=down.register_forward_hook(inject)
                        result=np.empty((len(levels),len(indices),4),np.float32)
                        try:
                            with torch.no_grad():
                                for hi,h in enumerate(levels):
                                    context["g"]=torch.tensor(basis([h])[0],dtype=torch.float32,device="cuda")
                                    digest=sha256()
                                    for batch in batches:
                                        context["p"]=torch.as_tensor([rows[i]["partner"] for i in batch],device="cuda")
                                        context["donor_norm"]=torch.as_tensor(norms[batch,LAYERS.index(donor)],device="cuda")
                                        out_score=forward(actor,ids,batch)
                                        result[hi,[lookup[i] for i in batch]]=distribution(out_score).cpu().numpy()
                                        digest.update(context["applied"].contiguous().cpu().numpy().tobytes())
                                    hashes[str(h)]=digest.hexdigest()
                        finally:
                            h1.remove();h2.remove();delattr(down,"conditional_residual")
                        if strength==1.:
                            for h,checksum in hashes.items():
                                key=(donor,signal,h)
                                if key in known:assert known[key]==checksum,("relocated correction changed",key)
                                known[key]=checksum
                        probes[name]=result;records[name]=dict(donor=donor,receiver=recv,strength=strength,signal=signal,
                                                               levels=list(levels),applied_sha256=hashes)
            np.savez_compressed(folder/"full-probes.npz",**probes);write(folder/"records.json",records)
            receipt.append([seed,m]);write(out/"status.json",dict(stage="actual_transplants",finished=receipt[-1],complete_worlds=len(receipt)))
    return dict(completed=receipt,contexts=len(indices),native_full_forward=True,matched_delta_hash_verified=True)
