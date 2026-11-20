"""Private activation adapters and the #17 normalized score-gradient learning rule."""

from hashlib import sha256
import numpy as np
import torch
from torch import nn
from experiments.individual_life.design import CAP, RATE, RIDGE

RANK = 2
DIM = 2560


def frame(device):
    gen = np.random.default_rng(78000)
    q, _ = np.linalg.qr(gen.normal(size=(DIM, RANK)))
    return torch.tensor(q.T.copy(), dtype=torch.float32, device=device)


def dot_rows(rows, h):
    return (rows * h.unsqueeze(0)).sum(dim=-1)


class LoReFT(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.R = nn.Parameter(initial.clone())
        self.W = nn.Parameter(initial.clone())
        self.bias = nn.Parameter(initial.new_zeros(initial.shape[0]))

    def delta(self, h):
        # Equivalent to h + R.T (W h + b - R h); zero exactly at birth.
        coef = dot_rows(self.W - self.R, h) + self.bias
        return (self.R * coef[:, None]).sum(0)

    def live(self):
        return self.R, self.W, self.bias

    def tangent(self, grads):
        r = self.R.detach()
        g = grads[0]
        cross = g @ r.T
        return (g - (0.5 * (cross + cross.T)) @ r, *grads[1:])

    @torch.no_grad()
    def retract(self):
        # Inverse square root of the tiny (rank-by-rank) Gram matrix.
        r = self.R.double()
        eig, vec = torch.linalg.eigh(r @ r.T)
        if float(eig.min()) <= 0:
            raise RuntimeError("LoReFT lost row rank")
        self.R.copy_(((vec * eig.rsqrt()) @ vec.T @ r).float())
        err = float(
            (self.R @ self.R.T - torch.eye(r.shape[0], device=r.device)).abs().max()
        )
        if err > 2e-5:
            raise RuntimeError(f"non-orthonormal LoReFT frame: {err}")
        return err


class ResidualLoRA(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.V = nn.Parameter(initial.clone())
        self.Ut = nn.Parameter(torch.zeros_like(initial))

    def delta(self, h):
        return (self.Ut * dot_rows(self.V, h)[:, None]).sum(0)

    def live(self):
        return self.V, self.Ut


def vector(model):
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.live()]
    ).astype("<f4", copy=False)


def signature(a):
    return sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


@torch.no_grad()
def advance(model, grads, signal, behavior, current):
    on_manifold = isinstance(model, LoReFT)
    g = model.tangent(grads) if on_manifold else grads
    energy = sum(float(torch.linalg.vector_norm(v)) ** 2 for v in g)
    ratio = float(np.clip(current / behavior, 0.5, 2.0))
    scale = RATE * signal * ratio / (RIDGE + energy)
    size = abs(scale) * np.sqrt(energy)
    if size > CAP:
        scale *= CAP / size
    before = vector(model)
    for p, grad in zip(model.live(), g, strict=True):
        p.add_(grad, alpha=float(scale))
    error = model.retract() if on_manifold and scale != 0.0 else 0.0
    after = vector(model)
    return dict(
        before=signature(before),
        after=signature(after),
        gradient_norm=float(np.sqrt(energy)),
        actual_l2=float(np.linalg.norm(after.astype(float) - before)),
        nominal_l2=float(abs(scale) * np.sqrt(energy)),
        orthogonality_error=error,
        importance=ratio,
        behavior=float(behavior),
        current=float(current),
    )
