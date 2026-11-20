"""Two zero-at-birth LoReFTs on the same frozen experienced actor; only their attachment differs."""

import numpy as np
import torch

from experiments.loreft_life.adapters import advance, vector
from .design import GRID, SMALL, STRENGTH
from experiments.lifetime_cooperation.design import LEVELS


def residual(raw, models):
    return torch.stack(
        [
            raw[j]
            if m is None
            else (raw[j].float() + m.delta(raw[j].float())).to(raw.dtype)
            for j, m in enumerate(models)
        ]
    )


class TwoSites:
    def __init__(self, batch, pre):
        self.batch = batch
        self.pre = pre
        self.early = [None] * 6
        self.late = [None] * 6
        self.handle = batch.engine.lm.layers[3].register_forward_hook(self.inject_early)

    def close(self):
        self.handle.remove()
        self.batch.engine.custom = None

    def inject_early(self, _, args, out):
        v = out[0] if isinstance(out, tuple) else out
        if not any(x is not None for x in self.early):
            return out
        modified = torch.cat(
            (v[:, :-1], residual(v[:, -1], self.early)[:, None]), dim=1
        )
        return (modified, *out[1:]) if isinstance(out, tuple) else modified

    def panel(self, i, h, early, late, checkpoint=False):
        self.early = early
        self.late = late

        def intervene(raw):
            base = residual(raw, [self.pre] * 6)
            return residual(base, self.late)

        self.batch.engine.custom = intervene
        try:
            p = self.batch.q.ordinary(i, h, self.batch.weights)
        finally:
            self.batch.engine.custom = None
        if checkpoint:
            self.last = p.detach().cpu().numpy()
        return p


def train_contrast(batch, pre, event, folder):
    e = event["factual"]
    i = e["row"]
    h = e["health"]
    a = e["action"]
    for t in pre.live():
        t.requires_grad_(False)
    sites = TwoSites(batch, pre)
    try:
        em, lm = batch.new(), batch.new()
        p = sites.panel(
            i, h, [None, em, None, None, None, None], [None, None, lm, None, None, None]
        )
        old = p.detach().cpu().numpy().copy()
        assert np.array_equal(old, np.broadcast_to(old[0], old.shape))
        eg = torch.autograd.grad(p[1, a].log(), em.live(), retain_graph=True)
        lg = torch.autograd.grad(p[2, a].log(), lm.live())
        candidates = []
        cal = []
        logs = []
        with torch.no_grad():
            for name, grads in (("early", eg), ("terminal", lg)):
                models = []
                steps = []
                for f in STRENGTH:
                    m = batch.new()
                    steps.append(
                        advance(
                            m, grads, e["signal"] * f, e["prob"][0][a], float(old[0, a])
                        )
                    )
                    models.append(m)
                mask = [None] + models
                q = (
                    sites.panel(
                        i,
                        h,
                        mask if name == "early" else [None] * 6,
                        mask if name == "terminal" else [None] * 6,
                    )
                    .cpu()
                    .numpy()
                )
                candidates.append(models)
                cal.append(q)
                logs.append(steps)
            cal = np.asarray(cal)
            tv = (
                0.5
                * abs(cal[:, 1:].astype(float) - cal[:, :1].astype(float)).sum(-1)
                * 100
            )
            # Match only by source query, with strongest common effect as the tiebreaker.
            feasible = [
                (j, k)
                for j in range(5)
                for k in range(5)
                if min(tv[0, j], tv[1, k]) >= 0.5
            ]
            pairs = feasible or [(j, k) for j in range(5) for k in range(5)]
            j, k = min(
                pairs,
                key=lambda x: (
                    abs(tv[0, x[0]] - tv[1, x[1]]),
                    -min(tv[0, x[0]], tv[1, x[1]]),
                    x,
                ),
            )
            valid = bool(feasible and abs(tv[0, j] - tv[1, k]) < 0.5)
            out = []
            for hi, idx in SMALL:
                q = sites.panel(
                    int(GRID[idx]),
                    LEVELS[hi],
                    [None, candidates[0][0], None, candidates[0][j], None, None],
                    [None, None, candidates[1][0], None, candidates[1][k], None],
                )
                out.append(q.detach().cpu().numpy()[:5])
            np.savez_compressed(
                folder / "sites.npz",
                panels=np.asarray(out),
                calibration=cal,
                gradients=np.array(
                    [
                        np.sqrt(sum(float(torch.linalg.vector_norm(v)) ** 2 for v in g))
                        for g in (eg, lg)
                    ]
                ),
                early=np.stack([vector(m) for m in candidates[0]]),
                terminal=np.stack([vector(m) for m in candidates[1]]),
            )
            return dict(
                early_index=j,
                terminal_index=k,
                matched=valid,
                source_tv=tv.tolist(),
                steps=logs,
                early_block=4,
                terminal_block=len(batch.engine.lm.layers),
            )
    finally:
        sites.close()
