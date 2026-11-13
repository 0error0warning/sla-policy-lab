"""Read native Qwen output scores with the complete frozen model; no learning or cached reconstruction."""

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import write
from experiments.forecast_stages.design import VIEWS
from experiments.internal_body.suffix import distribution
from experiments.lifetime_cooperation.design import PILOT
from experiments.loreft_life.adapters import frame, signature, vector
from experiments.loreft_life.engine import Engine, People
from experiments.loreft_omission.gpu import load_vector
from experiments.partner_prediction.model import Actor, ReadPolicy


class NativeScores:
    def __init__(self, frozen, root, out):
        self.frozen = frozen
        self.engine = Engine(frozen, root)
        base = People(self.engine.q)
        self.engine.bind(base)
        self.weights = [base.birth] * 6
        self.model = ReadPolicy(frame("cuda"))
        self.model.requires_grad_(False)
        self.head = frozen.model.lm_head
        self.ids = frozen.ids
        self.last = None
        self.dtype = None
        self.input_dtype = None
        self.calls = 0
        self.max_lane_error = 0.0
        self.max_prob_error = 0.0
        self.handle = self.head.register_forward_hook(self.capture)
        w = self.head.weight[self.ids].detach().float().cpu().numpy()
        b = self.head.bias
        bias = (
            np.zeros(4, dtype=np.float32)
            if b is None
            else b[self.ids].detach().float().cpu().numpy()
        )
        np.savez_compressed(
            out / "readout.npz", weight=w, bias=bias, token_ids=self.ids
        )

    def close(self):
        self.handle.remove()
        self.engine.close()

    def capture(self, _module, args, out):
        if self.last is not None:
            raise RuntimeError("multiple output-head invocations in one forward")
        self.dtype = str(out.dtype)
        self.input_dtype = str(args[0].dtype)
        assert self.input_dtype == "torch.bfloat16"
        self.last = (out[:, -1, self.ids].float(), args[0][:, -1, :].contiguous())

    def read(self, i, q):
        forecast = torch.as_tensor(q, device="cuda", dtype=torch.float32)
        roles = [(self.model, forecast[j]) for j in (0, 1, 2, 2, 1, 0)]
        self.last = None
        self.engine.custom = lambda h: Actor.inject(h, roles)
        try:
            p = self.engine.q.ordinary(i, 6.0, self.weights)
        finally:
            self.engine.custom = None
        if self.last is None:
            raise RuntimeError("Qwen's native head was not called")
        scores, x = self.last
        self.last = None
        self.calls += 1
        error = float((p - distribution(scores)).abs().max())
        mirror = max(
            float((t[:3] - t[3:].flip(0)).abs().max()) for t in (p, scores, x.float())
        )
        self.max_prob_error = max(self.max_prob_error, error)
        self.max_lane_error = max(self.max_lane_error, mirror)
        assert error == mirror == 0
        # BF16 words are stored as exact unsigned bit patterns for an independent CPU projection.
        return (
            scores[:3].cpu().numpy().copy(),
            p[:3].cpu().numpy().copy(),
            x[:3].contiguous().view(torch.uint16).cpu().numpy().copy(),
        )


@torch.no_grad()
def run(root, mode):
    assert mode == "expressionlogits"
    out = root / "output"
    out.mkdir(exist_ok=True)
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    native = NativeScores(actor, root, out)
    original = signature(vector(native.model))
    drift = {}
    try:
        with np.load(root / "frozen-snapshots.npz") as data:
            keys, slots, weights, hashes, q, expected = (
                data[k]
                for k in ("keys", "slots", "policy", "policy_sha", "q", "previous_prob")
            )
            for j, seed in enumerate(PILOT):
                write(
                    out / "status.json",
                    {"stage": "native_logits", "seed": seed, "completed": j},
                )
                rows = np.flatnonzero(keys[:, 0] == seed)
                all_z, all_p, all_x = [], [], []
                active = None
                for n in rows:
                    slot = int(slots[n])
                    if slot != active:
                        if active is not None:
                            assert (
                                signature(vector(native.model))
                                == hashes[active].decode()
                            )
                        load_vector(native.model, weights[slot])
                        assert signature(vector(native.model)) == hashes[slot].decode()
                        active = slot
                    z, pp, hidden = native.read(int(keys[n, 4]), q[n])
                    all_z.append(z)
                    all_p.append(pp)
                    all_x.append(hidden)
                    if keys[n, 1] == 0:
                        assert np.array_equal(z, np.broadcast_to(z[:1], z.shape))
                assert signature(vector(native.model)) == hashes[active].decode()
                folder = out / "lives" / f"s{seed}"
                folder.mkdir(parents=True)
                np.savez_compressed(
                    folder / "scores.npz",
                    rows=rows,
                    keys=keys[rows],
                    scores=all_z,
                    prob=all_p,
                    hidden_bits=all_x,
                )
                delta = abs(np.array(all_p, dtype=float) - expected[rows])
                drift[seed] = {
                    "max_pp": float(100 * delta.max()),
                    "same_distributions": int(np.all(delta == 0, axis=-1).sum()),
                    "distributions": 3 * len(rows),
                }
            assert original == hashes[0].decode()
    finally:
        native.close()
    actor.verify_unchanged()
    write(
        out / "receipt.json",
        {
            "mode": mode,
            "seeds": PILOT,
            "views": VIEWS,
            "words": WORDS,
            "token_ids": native.ids,
            "native_logit_dtype": native.dtype,
            "head_input_dtype": native.input_dtype,
            "weight_dtype": str(native.head.weight.dtype),
            "model_forward_calls": native.calls,
            "mirror_error": native.max_lane_error,
            "score_policy_error": native.max_prob_error,
            "archived_probability_check": drift,
            "device": torch.cuda.get_device_name(),
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.monotonic() - start,
            "frozen_backbone": True,
            "training_updates": 0,
            "versions": {
                k: version(k) for k in ("torch", "transformers", "numpy", "scipy")
            },
        },
    )
    write(
        out / "status.json",
        {"stage": "complete", "histories": len(PILOT), "model_forwards": native.calls},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=["expressionlogits"])
    a = p.parse_args()
    run(a.root, a.mode)
