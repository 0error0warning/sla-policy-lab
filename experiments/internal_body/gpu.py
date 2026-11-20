"""Entry point for pinned-model acquisitions and direct GPU checks."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    out = args.root / "output"
    out.mkdir(exist_ok=True, parents=True)
    start = time.monotonic()
    actor = FrozenQwen(args.root / "expected-weights.json", out)
    if args.mode.startswith("capture"):
        from experiments.internal_body.capture import capture

        result = capture(actor, args.root, out)
    elif args.mode.startswith("depth"):
        from experiments.internal_body.depth import run

        result = run(actor, args.root, out)
        if json.loads((args.root / "depth-job.json").read_bytes()).get(
            "also_jacobians"
        ):
            from experiments.internal_body.jacobians import capture

            dest = out / "jacobians"
            dest.mkdir(exist_ok=True)
            write(
                out / "status.json",
                {"stage": "jacobian_acquisition", "depth_training_completed": True},
            )
            result["jacobians"] = capture(actor, args.root, dest)
    elif args.mode.startswith("transplant"):
        from experiments.internal_body.transplant import deploy

        result = deploy(actor, args.root, out)
        if json.loads((args.root / "transplant-job.json").read_bytes()).get(
            "fresh_tail"
        ):
            from experiments.internal_body.fresh import capture

            write(out / "status.json", {"stage": "fresh_singleton_tail"})
            result["fresh_tail"] = capture(actor, args.root, out / "fresh-tail")
    elif args.mode.startswith("seen"):
        from experiments.internal_body.seen import run

        result = run(actor, args.root, out)
    elif args.mode.startswith("jacobian"):
        from experiments.internal_body.jacobians import capture

        result = capture(actor, args.root, out)
    else:
        from experiments.internal_body.live import run

        result = run(actor, args.root, out)
    actor.verify_unchanged()
    torch = actor.torch
    write(
        out / "receipt.json",
        {
            "mode": args.mode,
            "result": result,
            "elapsed_seconds": time.monotonic() - start,
            "model_shards_verified": True,
            "backbone_restored": True,
            "native_token_ids": actor.ids,
            "device": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "versions": {
                k: version(k)
                for k in ("torch", "transformers", "numpy", "scipy", "safetensors")
            },
        },
    )
    write(out / "status.json", {"stage": "complete"})


if __name__ == "__main__":
    main()
