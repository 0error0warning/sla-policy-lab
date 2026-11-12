"""Launch finite lives. A seed replicates an entire life, never an optimizer training batch."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.direct_policy.protocol import write
from experiments.individual_life.design import BIRTHS,CONFIRM,HIDDEN,PILOT,VISIBLE,rows,tape
from experiments.individual_life.life import run_one
from experiments.individual_life.model import Complete,Native
from experiments.social_closed_loop.core import BodyValue


def directory(out,visible,seed,birth,rule):
    return out/("visible" if visible else "hidden")/f"s{seed}"/birth/rule


def execute(native,backend,out,seeds,births,physical,wave):
    out.mkdir(parents=True,exist_ok=True)
    body=BodyValue();complete=[]
    for visible in ((True,False) if wave=="pilot" else (True,)):
        if not visible:
            assert complete and all(v["next_living_decisions"]==v["turns"]-1 for v in complete)
        for seed in seeds:
            source=tape(seed,rows())
            for birth in births:
                if wave=="confirm":rules=HIDDEN
                elif not physical:rules=VISIBLE if visible else HIDDEN
                elif visible:rules=VISIBLE if seed==PILOT[0] else HIDDEN
                else:rules=HIDDEN if seed==PILOT[0] else ()
                for rule in rules:
                    d=directory(out,visible,seed,birth,rule)
                    if (d/"life.json").exists():
                        record=json.loads((d/"life.json").read_bytes())
                        assert record["seed"]==seed and record["birth"]==birth and record["rule"]==rule and record["visible"]==visible
                    else:record=run_one(d,backend,body,source,seed,birth,visible,rule)
                    assert record["main_steps"]==(0 if rule=="frozen" else record["turns"])
                    complete.append(record)
                    status={"stage":"living_decisions","completed":len(complete),"seed":seed,"birth":birth,"rule":rule,"visible":visible}
                    write(out/"status.json",status)
                    if physical:write(out.parent/"status.json",status)
            print("finished",wave,("visible" if visible else "hidden"),seed,flush=True)
    results={"wave":wave,"engine":"complete_qwen" if physical else "native_cached_prefix",
             "lives":len(complete),"encounters":sum(v["turns"] for v in complete),"complete_seeds":list(seeds),
             "next_living":sum(v["next_living_decisions"] for v in complete),"terminal":sum(v["physical_death"] for v in complete)}
    write(out/"cohort.json",results);return results


def hardware(actor,root,out):
    params=json.loads((root/"study-plan.json").read_bytes())
    native=Native(root,root,"cuda")
    assert actor.ids==json.loads((root/"capture-ids.json").read_bytes())
    from experiments.internal_body.capture import encode
    for i in range(len(native.rows)):assert encode(actor,native.rows[i])==native.ids[i]
    original_parameters=sum(p.numel() for p in actor.model.parameters())
    backend=Complete(actor,native,root)
    res=execute(native,backend,out/"lives",tuple(params["seeds"]),tuple(params["births"]),True,params["wave"])
    from experiments.individual_life.replay import compare_timing
    if params["wave"]=="pilot":
        audits=[]
        for birth in BIRTHS:
            audits.append(compare_timing(directory(out/"lives",True,PILOT[0],birth,"true1"),backend,birth))
        write(out/"same-tape-timing.json",audits)
    res["native_parameter_count_before_and_after"]=original_parameters
    assert original_parameters==sum(p.numel() for p in actor.model.parameters())
    return res


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("cache","arch","output"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--wave",choices=("pilot","confirm"),required=True)
    p.add_argument("--births",nargs="+",choices=BIRTHS)
    a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    backend=Native(a.cache,a.arch)
    execute(backend,backend,a.output,PILOT if a.wave=="pilot" else CONFIRM,tuple(a.births or BIRTHS),False,a.wave)
