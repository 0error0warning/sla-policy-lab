"""Entry point for pinned-model acquisitions and direct GPU checks."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import time

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--mode",required=True)
    args=parser.parse_args()
    out=args.root/"output"
    out.mkdir(exist_ok=True,parents=True)
    start=time.monotonic()
    actor=FrozenQwen(args.root/"expected-weights.json",out)
    if args.mode=="capture":
        from experiments.internal_body.capture import capture
        result=capture(actor,args.root,out)
    elif args.mode.startswith("depth"):
        from experiments.internal_body.depth import run
        result=run(actor,args.root,out)
    else:
        from experiments.internal_body.live import run
        result=run(actor,args.root,out)
    actor.verify_unchanged()
    torch=actor.torch
    write(out/"receipt.json",dict(mode=args.mode,result=result,elapsed_seconds=time.monotonic()-start,
          model_shards_verified=True,backbone_restored=True,native_token_ids=actor.ids,
          device=torch.cuda.get_device_name(),cuda=torch.version.cuda,peak_bytes=torch.cuda.max_memory_allocated(),
          versions={k:version(k) for k in ("torch","transformers","numpy","scipy","safetensors")}))
    write(out/"status.json",dict(stage="complete"))

if __name__=="__main__":
    main()
