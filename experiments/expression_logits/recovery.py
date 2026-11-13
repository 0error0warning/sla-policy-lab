"""Re-run the archived parent entry, listen to its focal scores, then install frozen source states."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import write
from experiments.forecast_stages.design import PARENT_RUNS, VIEWS
from experiments.forecast_stages.gpu import Probe, one_history
from experiments.internal_body.suffix import distribution
from experiments.lifetime_cooperation.design import PILOT
from experiments.loreft_life.adapters import signature, vector
from experiments.loreft_omission.gpu import load_vector
from experiments.partner_prediction.model import Actor

from .design import CASE_ID, CASE_SEED, TWIN_ID


class ParentProbe(Probe):
    def __init__(self, actor):
        super().__init__(actor)
        self.stage = None
        self.focal = {}

    def panel(self, model, seed, t, head, x, full):
        self.stage = (seed, t)
        return super().panel(model, seed, t, head, x, full)

    def read(self, model, i, h, vectors, check):
        if self.stage != (CASE_SEED, 31) or i not in (CASE_ID, TWIN_ID) or h != 6.0:
            return super().read(model, i, h, vectors, check)
        saved = []
        frozen = self.actor.b.engine.actor

        def capture(_, args, output):
            saved.append(
                output[:, -1, frozen.ids].detach().float().cpu().numpy().copy()
            )

        handle = frozen.model.lm_head.register_forward_hook(capture)
        try:
            probs = super().read(model, i, h, vectors, check)
        finally:
            handle.remove()
        np.testing.assert_array_equal(saved[-1][:3], saved[-1][3:][::-1])
        self.focal[str(i)] = {
            "scores": saved[-1][:3].tolist(),
            "probs": probs.tolist(),
            "q": vectors.tolist(),
            "sha": signature(vector(model)),
            "check": bool(check),
            "native_only": self.native_only,
        }
        return probs


@torch.no_grad()
def frozen_rescore(actor, frozen, root, out):
    last = []
    output_dtypes = set()

    def hook(_, args, output):
        if last:
            raise RuntimeError("multiple vocabulary outputs")
        output_dtypes.add((str(output.dtype), str(args[0].dtype)))
        last.append((output[:, -1, frozen.ids].float(), args[0][:, -1].contiguous()))

    handle = frozen.model.lm_head.register_forward_hook(hook)
    model = actor.new("true")
    model.requires_grad_(False)
    hd = frozen.model.lm_head
    b = (
        np.zeros(4, dtype=np.float32)
        if hd.bias is None
        else hd.bias[frozen.ids].float().cpu().numpy()
    )
    np.savez_compressed(
        out / "readout.npz",
        weight=hd.weight[frozen.ids].float().cpu().numpy(),
        bias=b,
        token_ids=frozen.ids,
    )
    max_cache = max_hidden = max_mirror = max_reconstructed = 0.0
    calls = 0
    drift = {}
    try:
        with np.load(root / "frozen-snapshots.npz") as data:
            keys, slots, weights, hashes, qs, old = (
                data[k]
                for k in ("keys", "slots", "policy", "policy_sha", "q", "previous_prob")
            )
            for pos, seed in enumerate(PILOT):
                write(
                    out / "status.json",
                    {"stage": "strict_snapshot_scores", "seed": seed, "completed": pos},
                )
                rr = np.flatnonzero(keys[:, 0] == seed)
                zs, ps, xs = [], [], []
                active = None
                for n in rr:
                    slot = int(slots[n])
                    if slot != active:
                        if active is not None:
                            assert signature(vector(model)) == hashes[active].decode()
                        load_vector(model, weights[slot])
                        active = slot
                        assert signature(vector(model)) == hashes[slot].decode()
                    i = int(keys[n, 4])
                    q = qs[n]
                    roles = actor.lanes([(model, v) for v in [*q, *reversed(q)]])
                    full = actor.ordinary(i, 6.0, roles)
                    z, x = last.pop()
                    raw = actor.inject(actor.common[i, 6.0].expand(6, -1), roles)
                    cache = actor.b.engine._read(i, raw)
                    cz, cx = last.pop()
                    max_cache = max(
                        max_cache,
                        float((full - cache).abs().max()),
                        float((z - cz).abs().max()),
                    )
                    max_hidden = max(
                        max_hidden, float((x.float() - cx.float()).abs().max())
                    )
                    max_mirror = max(
                        max_mirror,
                        *(
                            float((t[:3] - t[3:].flip(0)).abs().max())
                            for t in (z, full, x.float())
                        ),
                    )
                    max_reconstructed = max(
                        max_reconstructed, float((distribution(z) - full).abs().max())
                    )
                    zs.append(z[:3].cpu().numpy().copy())
                    ps.append(full[:3].cpu().numpy().copy())
                    xs.append(
                        x[:3].contiguous().view(torch.uint16).cpu().numpy().copy()
                    )
                    calls += 1
                assert signature(vector(model)) == hashes[active].decode()
                folder = out / "lives" / f"s{seed}"
                folder.mkdir(parents=True)
                np.savez_compressed(
                    folder / "scores.npz",
                    rows=rr,
                    keys=keys[rr],
                    scores=zs,
                    prob=ps,
                    hidden_bits=xs,
                )
                d = abs(np.asarray(ps, dtype=float) - old[rr])
                drift[seed] = {
                    "max_pp": float(100 * d.max()),
                    "same_distributions": int((d == 0).all(-1).sum()),
                    "distributions": 3 * len(rr),
                }
    finally:
        handle.remove()
    assert max_mirror == max_reconstructed == 0 and len(output_dtypes) == 1
    native_dtype, input_dtype = output_dtypes.pop()
    return {
        "model_forward_calls": calls,
        "cache_comparisons": calls,
        "cache_max": max_cache,
        "cache_hidden_max": max_hidden,
        "mirror_error": max_mirror,
        "score_policy_error": max_reconstructed,
        "archived_probability_check": drift,
        "exported_score_dtype": "torch.float32",
        "native_logit_dtype": native_dtype,
        "head_input_dtype": input_dtype,
        "weight_dtype": str(hd.weight.dtype),
    }


def run(root, mode):
    assert mode == "logitrecover"
    out = root / "output"
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    tapes = json.loads((root / "tapes.json").read_bytes())
    anchors = {tag: np.load(root / (tag + "-anchors.npz")) for tag in PARENT_RUNS}
    frozen = FrozenQwen(root / "expected-weights.json", out)
    actor = Actor(frozen, root, out)
    probe = ParentProbe(actor)
    alignment = {}
    try:
        with np.load(root / "frozen-snapshots.npz") as saved:
            for k, seed in enumerate(PILOT):
                write(
                    out / "status.json",
                    {"stage": "parent_chronology", "seed": seed, "completed": k},
                )
                folder = root / "parent-replayed" / f"s{seed}"
                folder.mkdir(parents=True)
                one_history(actor, probe, folder, seed, tapes[str(seed)], anchors)
                rows = np.flatnonzero(saved["checkpoints"][:, 0] == seed)
                times = saved["checkpoints"][rows, 1]
                with np.load(folder / "states.npz") as s:
                    equal = s["sha"][times] == saved["policy_sha"][rows]
                    alignment[seed] = {"exact": int(equal.sum()), "checked": len(equal)}
        write(
            out / "parent-focus.json",
            {
                "readouts": probe.focal,
                "parameter_matches": alignment,
                "parent_probe_checks": probe.comparisons,
                "parent_probe_max_error": probe.max_error,
            },
        )
        measurements = frozen_rescore(actor, frozen, root, out)
    finally:
        actor.b.close()
        for w in anchors.values():
            w.close()
    frozen.verify_unchanged()
    write(
        out / "receipt.json",
        dict(
            mode=mode,
            views=VIEWS,
            words=WORDS,
            seeds=PILOT,
            token_ids=frozen.ids,
            training_updates=0,
            parent_preparatory_bodily_writes=511,
            device=torch.cuda.get_device_name(),
            seconds=time.monotonic() - started,
            peak_bytes=torch.cuda.max_memory_allocated(),
            frozen_backbone=True,
            versions={
                k: version(k) for k in ("torch", "transformers", "numpy", "scipy")
            },
            **measurements,
        ),
    )
    write(
        out / "status.json",
        {"stage": "complete", "histories": len(PILOT), "recovered_readout": True},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["logitrecover"], required=True)
    a = p.parse_args()
    run(a.root, a.mode)
