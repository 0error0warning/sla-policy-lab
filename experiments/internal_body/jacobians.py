"""Record actual Qwen local logit derivatives at three zero-intervention sites."""
import numpy as np
import torch
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import encode,grouped
from experiments.internal_body.depth import forward
from experiments.internal_body.design import all_rows
from experiments.internal_body.features import linear_projection,project

LAYERS=(3,15,31)


def capture(actor,root,out):
    lm=actor.model.model.language_model
    rows=all_rows()[:960];prefixes=[encode(actor,r) for r in rows]
    hooks=[];offsets={};rms={};current={}
    d=lm.layers[-1].mlp.down_proj.out_features
    raw=np.empty((len(rows),lm.layers[3].mlp.down_proj.in_features),np.float32)
    native=np.zeros((len(rows),4),np.float32)
    norm=np.zeros((len(rows),3),np.float32)
    derivatives={l:np.empty((len(rows),4,d),np.float32) for l in LAYERS}
    def early(_,args):current["v"]=args[0][:,-1].detach().float().cpu().numpy()
    hooks.append(lm.layers[3].mlp.down_proj.register_forward_pre_hook(early))
    def add(site):
        def hook(_,args,result):
            delta=torch.zeros((len(result),d),device=result.device,requires_grad=True)
            offsets[site]=delta
            return torch.cat((result[:,:-1],(result[:,-1].float()+delta).to(result.dtype)[:,None]),dim=1)
        return hook
    def observe(site):
        def hook(_,args,result):
            v=result[0] if isinstance(result,tuple) else result
            rms[site]=v[:,-1].detach().float().square().sum(dim=1).sqrt().cpu().numpy()
        return hook
    for l in LAYERS:
        hooks.append(lm.layers[l].mlp.down_proj.register_forward_hook(add(l)))
        hooks.append(lm.layers[l].register_forward_hook(observe(l)))
    count=0
    try:
        for ix in grouped(range(len(rows)),prefixes,4):
            logits=forward(actor,prefixes,ix)
            native[ix]=logits.detach().cpu().numpy();raw[ix]=current["v"]
            for j,l in enumerate(LAYERS):norm[ix,j]=rms[l]
            for a in range(4):
                grads=torch.autograd.grad(logits[:,a].sum(),[offsets[l] for l in LAYERS],retain_graph=a<3)
                for l,v in zip(LAYERS,grads):derivatives[l][ix,a]=v.detach().float().cpu().numpy()
            count+=len(ix)
            if count%48<4:write(out/"status.json",dict(stage="real_jacobians",finished=count,total=len(rows)))
    finally:
        for h in hooks:h.remove()
    projection=linear_projection(raw,np.arange(len(raw))<768)
    np.savez_compressed(out/"common.npz",native=native,residual_norm=norm,projection=projection,features=project(raw,projection))
    for l,grad in derivatives.items():
        assert np.isfinite(grad).all() and np.all(np.linalg.norm(grad,axis=-1)>0)
        np.savez_compressed(out/f"layer-{l}.npz",gradient=grad)
    write(out/"rows.json",[dict(**r,ids=p) for r,p in zip(rows,prefixes)])
    return dict(contexts=len(rows),derivative_shape=[len(rows),4,d],layers=list(LAYERS),
                calibration_zero_delta=True,trainable_original_parameters=0)
