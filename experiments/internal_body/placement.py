"""Fit a first-order, norm-limited internal adapter from full observed experience tapes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.study import KL, TEST_H, basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.design import LR_A, LR_B, RANK, SPLIT_H, STEPS
from experiments.internal_body.jacobians import LAYERS
from experiments.internal_body.population import acquire
from experiments.internal_body.suffix import arr, distribution
from experiments.social_closed_loop.core import BodyValue

SEEDS = tuple(range(65001, 65013))
CAP = 0.12


class Tangent(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(
            torch.tensor(
                np.random.default_rng(64800).normal(0, 0.06, (2, 12, 33, RANK)),
                dtype=torch.float32,
            )
        )
        self.B = nn.Parameter(torch.zeros(RANK, d))

    def scores(self, x, native, norm, jac, p, g):
        input_factor = torch.einsum("nd,nbdr->nbr", x, self.A[p])
        c = torch.einsum("hb,nbr->hnr", g, input_factor)
        length = ((c @ (self.B @ self.B.T)) * c).sum(-1).clamp_min(1e-12).sqrt()
        scaler = (CAP * norm[None] / length).clamp(max=1)
        shift = torch.einsum("hnr,nar->hna", c, jac @ self.B.T)
        return native[None] + shift * scaler[:, :, None], scaler


def fit_tangent(args, test, scores):
    _x, n, _norm, j, _p, _g = args
    model = Tangent(j.shape[-1])
    optimizer = torch.optim.Adam(
        [{"params": [model.A], "lr": LR_A}, {"params": [model.B], "lr": LR_B}]
    )
    baseline = distribution(n).detach()
    val = torch.as_tensor(scores, dtype=torch.float32)
    history = []
    for t in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits, scale = model.scores(*args)
        pi = distribution(logits)
        gain = (pi * val).sum(-1).mean()
        kl = (pi * (pi / baseline).log()).sum(-1).mean()
        loss = -gain + KL * kl
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            [
                t + 1,
                float(loss.detach()),
                float(gain.detach()),
                float(kl.detach()),
                float(grad),
                float((scale < 1 - 1e-6).float().mean().detach()),
            ]
        )
    with torch.no_grad():
        logits, scaler = model.scores(*test)
        pi = distribution(logits).numpy()
    return model, pi, scaler.detach().numpy(), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    rows = json.loads((a.capture / "rows.json").read_bytes())
    with np.load(a.capture / "common.npz") as z:
        x, n, allnorm = [
            torch.as_tensor(z[k], dtype=torch.float32)
            for k in ("features", "native", "residual_norm")
        ]
    bank = SimpleNamespace(native=n.numpy(), rows=rows)
    gradients = {
        l: torch.as_tensor(
            np.load(a.capture / f"layer-{l}.npz")["gradient"], dtype=torch.float32
        )
        for l in LAYERS
    }
    tr = np.arange(768)
    te = np.arange(768, 960)
    partner = torch.tensor([r["partner"] for r in rows])
    body = BodyValue()
    ref = [evaluation_reference(body, [rows[i] for i in te], m) for m in (0, 1)]
    grids = {}
    for j, l in enumerate(LAYERS):
        for stage, ix, levels in (("train", tr, SPLIT_H), ("test", te, TEST_H)):
            grids[l, stage] = (
                x[ix],
                n[ix],
                allnorm[ix, j],
                gradients[l][ix],
                partner[ix],
                arr(basis(levels), "cpu"),
            )
    for seed in SEEDS:
        scores = acquire(seed, bank, tr, body, a.output)
        for m in (0, 1):
            folder = a.output / f"s{seed}-m{m}"
            hist = {}
            measures = {}
            probes = {}
            for l in LAYERS:
                for sig in ("real", "shuffle"):
                    name = f"l{l}-{sig}"
                    model, pi, _scale, history = fit_tangent(
                        grids[l, "train"], grids[l, "test"], scores[m][sig]
                    )
                    np.savez_compressed(
                        folder / (name + ".npz"),
                        A=model.A.detach().numpy(),
                        B=model.B.detach().numpy(),
                    )
                    hist[name] = history
                    probes[name] = pi
                    measures[name] = metrics(pi, ref[m])
            write(folder / "tangent-history.json", hist)
            write(folder / "predicted-metrics.json", measures)
            np.savez_compressed(folder / "predicted-probes.npz", **probes)
        print("tangent seed", seed, flush=True)


if __name__ == "__main__":
    main()
