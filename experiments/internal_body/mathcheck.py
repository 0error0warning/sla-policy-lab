"""Independent dense residual and derivative checks; no experiment scores are simulated."""

from types import SimpleNamespace

import numpy as np
import torch

from experiments.consequence_policy.study import basis
from experiments.internal_body.suffix import InternalPolicy, distribution


def main():
    torch.manual_seed(22)
    d, n = 53, 9
    norm = torch.randn(d) * 0.1
    head = torch.randn(4, d) * 0.13
    bank = SimpleNamespace(
        device="cpu",
        norm=norm,
        head=head,
        t=(1 + norm[:, None]) * head.T,
        cfg={"epsilon": 1e-6},
        x=np.zeros((1, 33)),
    )
    r = torch.randn(1, n, d)
    x = torch.randn(1, n, 33)
    square = r.square().mean(-1, keepdim=True)
    div = (square + 1e-6).sqrt()
    g = torch.as_tensor(basis([2.0, 6.0, 300.0]), dtype=torch.float32)
    p = torch.arange(n) % 2
    grid = {
        "r": r,
        "x": x,
        "g": g,
        "partner": p,
        "sq": square,
        "div": div,
        "dot": r @ bank.t,
        "n": r @ bank.t / div,
    }
    for kind in ("linear", "nonlinear", "postnorm", "blind", "no_route"):
        model = InternalPolicy(bank, kind)
        torch.testing.assert_close(
            model.scores(grid), grid["n"].expand(3, -1, -1), atol=1e-6, rtol=1e-6
        )
        with torch.no_grad():
            model.B.normal_(0, 0.2)
        actual = model.scores(grid)
        c = model.coefficients(x, g, p)
        delta = c @ model.B
        residual = r + delta
        dense = (residual @ bank.t) / (
            residual.square().mean(-1, keepdim=True) + 1e-6
        ).sqrt()
        if kind == "postnorm":
            dense = (r + delta) @ bank.t / div
        torch.testing.assert_close(actual, dense, atol=3e-6, rtol=1e-5)
        grad = torch.autograd.grad(
            actual.square().mean(), (model.A, model.B), retain_graph=True
        )
        other = torch.autograd.grad(dense.square().mean(), (model.A, model.B))
        for a, b in zip(grad, other):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)
        if kind == "linear":
            z = torch.randn(1, 33)
            t = torch.einsum("b,bdr->dr", g[0], model.A[0])
            torch.testing.assert_close(
                (z @ t) @ model.B, z @ (t @ model.B), atol=2e-6, rtol=1e-5
            )
    # Analytic gradient used in personal learning, compared with autograd on all actions.
    residual = torch.randn(d, requires_grad=True)
    den = (residual.square().mean() + 1e-6).sqrt()
    native = residual @ bank.t / den
    pi = distribution(native)
    s = torch.softmax(native / 2, dim=-1)
    jac = (torch.eye(4) - s) * 0.95 * s[:, None] / pi[:, None] / 2
    v = jac @ bank.t.T
    reference = v / den - (v @ residual)[:, None] * residual[None] / (d * den**3)
    emp = torch.stack(
        [
            torch.autograd.grad(pi[i].log(), residual, retain_graph=True)[0]
            for i in range(4)
        ]
    )
    torch.testing.assert_close(reference, emp, atol=1e-6, rtol=1e-5)
    from experiments.internal_body.carrier import carrier_basis, rotate

    u, axes = carrier_basis(r.numpy()[0], bank.t.numpy(), 7)
    body = rotate(r, g, torch.from_numpy(u), torch.from_numpy(axes))
    torch.testing.assert_close(
        body @ bank.t, (r @ bank.t).expand(3, -1, -1), atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        body.square().sum(-1), r.square().sum(-1).expand(3, -1), atol=1e-5, rtol=1e-5
    )
    print(
        "Dense residual scores, A/B gradients, low-rank folding, online gradients and carrier neutrality pass."
    )


if __name__ == "__main__":
    main()
