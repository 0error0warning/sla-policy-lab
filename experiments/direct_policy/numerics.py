"""Post hoc finite-scale directional derivative diagnostic on real Qwen.

This never changes the main experiment's model, rewards, or trial selection.
It checks whether a small BF16 forward can resolve a computed LoRA gradient.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from protocol import prompt,request,write
from run_gpu import Actor,file_hash,tensor_hash


SCALES=[1e-6,1e-5,1e-4,1e-3,1e-2,.05,.1]


def run(root):
    import torch
    from safetensors.torch import save_file
    output=root/'output';output.mkdir(exist_ok=False);start=time.monotonic()
    status=dict(stage='initializing',post_hoc_diagnostic=True,optimizer_steps=0)
    def progress(add_steps=0,**kw):
        status.update(kw);status['optimizer_steps']+=add_steps
        status['elapsed_seconds']=time.monotonic()-start;write(output/'status.json',status)
    actor=Actor(root,output,progress);actor.reset('early')
    initial=actor.snapshot();message=prompt('P17',request(0,0,0),300.,[0,1,2,3],False)
    prefix=actor.encode(message)
    actor.model.zero_grad(set_to_none=True)
    selected=actor.distribution([prefix])[0,2];logp=selected.log();logp.backward()
    gradients={n:p.grad.detach().cpu().float().clone() for n,p in actor.parameters['early'].items()}
    save_file(gradients,str(output/'gradient.safetensors'))
    norm=float(torch.sqrt(sum((g.double()**2).sum() for g in gradients.values())))
    assert norm>0 and np.isfinite(norm)
    direction={n:g/norm for n,g in gradients.items()}
    def observation():
        with torch.no_grad():
            z=actor.logits([prefix])[0]
            p=actor.distribution([prefix])[0]
            return dict(logits=z.cpu().tolist(),probabilities=p.cpu().tolist(),logp=float(p[2].log()))
    actor.install(initial);base=observation();repeat=observation();assert base==repeat
    entries=[]
    for scale in SCALES:
        points={}
        for sign in (1,-1):
            state={n:v+sign*scale*direction[n] for n,v in initial.items()}
            actor.install(state);points[str(sign)]=observation()
        actor.install(initial);restored=observation();assert restored==base
        finite=(points['1']['logp']-points['-1']['logp'])/(2*scale)
        entries.append(dict(parameter_step_l2=scale,plus=points['1'],minus=points['-1'],
            central_difference=finite,autograd_directional_derivative=norm,
            difference_over_autograd=finite/norm,restored_exact=True))
        progress(stage='directional-check',completed_scales=len(entries))
    # Reverse-order repeat detects state/call-order drift independently of sign.
    repeats=[]
    for scale in (.0001,.1):
        points={}
        for sign in (-1,1):
            actor.install({n:v+sign*scale*direction[n] for n,v in initial.items()});points[str(sign)]=observation()
        original=next(r for r in entries if r['parameter_step_l2']==scale)
        assert points['1']==original['plus'] and points['-1']==original['minus']
        repeats.append(dict(parameter_step_l2=scale,reverse_order_exact=True))
    actor.install(initial);actor.verify_base();assert tensor_hash(actor.snapshot().items())==tensor_hash(initial.items())
    value=dict(scope='Post hoc BF16 real-model finite-scale diagnostic; not an optimizer comparison or new life',
        source_sha256=file_hash(Path(__file__)),input_messages=message,input_token_ids=prefix,
        gradient_sha256=file_hash(output/'gradient.safetensors'),torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        action=2,initial=base,gradient_norm=norm,scales=entries,reverse_repeats=repeats,
        initial_adapter_sha256=tensor_hash(initial.items()),all_parameters_restored=True,
        base_unchanged=True,base_compute_dtype='bfloat16',lora_dtype='float32',
        elapsed_seconds=time.monotonic()-start)
    write(output/'numerics.json',value);progress(stage='complete',base_unchanged=True)
    print(json.dumps({k:v for k,v in value.items() if k in ('gradient_norm','scales','reverse_repeats','elapsed_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.root)
