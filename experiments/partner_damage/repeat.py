"""Independent device check: original and paired interfaces at fixed pre-owned parameters."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import torch
import numpy as np

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.design import LEVELS
from experiments.loreft_life.adapters import signature, vector
from experiments.loreft_omission.gpu import Batch, update_roles
from experiments.social_closed_loop.core import BodyValue
from .design import GRID
from .gpu import train_once, continue_fork, mix_panel, model_from


def run(root):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    events = [
        e
        for e in json.loads((root / "events.json").read_bytes())
        if e["focus"] or e["key"] == "s77101t1"
    ]
    tapes = json.loads((root / "tapes.json").read_bytes())
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    batch = Batch(actor, root)
    body = BodyValue()
    results = []
    try:
        write(out / "tokens.json", dict(ids=batch.q.inputs.ids, action_ids=actor.ids))
        with np.load(root / "anchors.npz") as z:
            for case in events:
                key = case["key"]
                e = case["factual"]
                folder = out / key
                folder.mkdir()
                write(
                    out / "status.json", dict(stage="same_device_interfaces", case=key)
                )
                m = model_from(batch, z[key + "_pre"])
                m0 = model_from(batch, z[key + "_pre"])
                a = [None, None, None, m, None, None]
                b = [m0, None, None, None, None, None]
                with torch.no_grad():
                    crop = batch.panel(a, True)[3]
                    alt = batch.panel(b, True)[0]
                    native = np.stack(
                        [
                            [
                                batch.ordinary(int(i), h, a)[3].cpu().numpy()
                                for i in GRID
                            ]
                            for h in LEVELS
                        ]
                    )
                p = batch.ordinary(e["row"], e["health"], a)
                prev = p[3].detach().cpu().numpy()
                u = update_roles(a, p, e, [3])[3]
                q = batch.ordinary(e["row"], e["health"], b)
                altp = q[0].detach().cpu().numpy()
                uu = update_roles(b, q, e, [0])[0]
                old, ms, meta = train_once(batch, case, z, folder, body)
                idx = 0 if e["maintenance"] else 3
                meta["patch"] = mix_panel(batch, old, ms[idx], case, folder)
                if case["focus"]:
                    meta["continuation"] = continue_fork(
                        batch, case, tapes[str(case["seed"])], ms, folder, z
                    )
                check = dict(
                    key=key,
                    native_vs_cache=float(abs(crop - native).max()),
                    slot_pre_error=float(abs(crop - alt).max()),
                    slot_source_error=float(abs(prev - altp).max()),
                    slot_post_hash_equal=uu["after"] == u["after"],
                    archive_slot_hash_equal=u["after"] == signature(z[key + "_post"]),
                    pre_anchor_error=float(abs(crop - z[key + "_prepanel"]).max()),
                    original_vs_factorial=signature(vector(m))
                    == meta["source_update_hash"],
                )
                np.savez_compressed(
                    folder / "legacy.npz",
                    before=crop,
                    native_before=native,
                    slot0_before=alt,
                    source=prev,
                    source0=altp,
                    post_weights=vector(m),
                    slot0_post_weights=vector(m0),
                )
                write(folder / "receipt.json", {"anchor": check, **meta})
                results.append(check)
    finally:
        batch.close()
    actor.verify_unchanged()
    write(
        out / "receipt.json",
        dict(
            checks=results,
            device=torch.cuda.get_device_name(),
            seconds=time.monotonic() - start,
            frozen_backbone=True,
            device_memory=torch.cuda.get_device_properties(0).total_memory,
            compute_capability=list(torch.cuda.get_device_capability()),
            versions={
                k: version(k)
                for k in ("torch", "numpy", "transformers", "scipy", "triton")
            },
        ),
    )
    write(out / "status.json", dict(stage="complete", cases=len(results)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["socialcheck"], required=True)
    run(p.parse_args().root)
