"""Refit selected published population, native and optional tangent checkpoints on CPU."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.policy import feedback_scores
from experiments.direct_policy.protocol import write
from experiments.internal_body.carrier import carrier_basis, rotated
from experiments.internal_body.design import SPLIT_H
from experiments.internal_body.suffix import InternalPolicy, StateBank, fit
from experiments.social_closed_loop.core import BodyValue


def signal(folder, seed, m, scale):
    with (
        np.load(folder / f"behavior-{seed}.npz") as b,
        np.load(folder / f"s{seed}-m{m}/outcomes.npz") as o,
    ):
        return feedback_scores(b["action"], o["reward"], b["mu"], scale)


def main():
    p = argparse.ArgumentParser()
    for k in ("cache", "population", "native", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--placement", type=Path)
    p.add_argument("--jacobians", type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    bank = StateBank(a.cache)
    body = BodyValue()
    tr = np.arange(768)
    scale = np.minimum(1.0, [body.rul(h) for h in SPLIT_H])
    reports = []
    for seed, m in ((64001, 0), (64120, 1)):
        scores = signal(a.population, seed, m, scale)
        for arm in ("linear", "nonlinear", "postnorm", "native_prompt"):
            model = InternalPolicy(bank, arm)
            fit(model, bank.grid(tr, SPLIT_H, model.uses_text), scores)
            with np.load(a.population / f"s{seed}-m{m}" / (arm + ".npz")) as z:
                for name, v in model.named_parameters():
                    np.testing.assert_array_equal(v.detach().numpy(), z[name])
            reports.append({"seed": seed, "mapping": m, "arm": arm, "exact": True})
    torch.set_num_threads(3)
    for seed, m in ((64401, 0), (64520, 1)):
        u, v = carrier_basis(bank.r[tr], bank.t.numpy(), seed)
        with np.load(a.native / f"basis-{seed}.npz") as z:
            np.testing.assert_array_equal(u, z["u"])
            np.testing.assert_array_equal(v, z["v"])
        grid, _ = rotated(bank.grid(tr, SPLIT_H), bank, u, v)
        model = InternalPolicy(bank, "native_hidden")
        fit(model, grid, signal(a.native, seed, m, scale))
        with np.load(a.native / f"s{seed}-m{m}/native-sensor.npz") as z:
            np.testing.assert_array_equal(model.delta.detach().numpy(), z["delta"])
        reports.append(
            {"seed": seed, "mapping": m, "arm": "native_sensor", "exact": True}
        )
    if a.placement:
        from experiments.consequence_policy.study import TEST_H, basis
        from experiments.internal_body.placement import fit_tangent

        torch.set_num_threads(4)
        common = np.load(a.jacobians / "common.npz")
        i, j = np.arange(768), np.arange(768, 960)
        part = torch.tensor([r["partner"] for r in bank.rows[:960]])

        def grid(idx, h, position, jac):
            return (
                torch.tensor(common["features"][idx]),
                torch.tensor(common["native"][idx]),
                torch.tensor(common["residual_norm"][idx, position]),
                jac[idx],
                part[idx],
                torch.tensor(basis(h), dtype=torch.float32),
            )

        for position, l in enumerate((3, 15, 31)):
            jac = torch.as_tensor(np.load(a.jacobians / f"layer-{l}.npz")["gradient"])
            model, _, _, _ = fit_tangent(
                grid(i, SPLIT_H, position, jac),
                grid(j, TEST_H, position, jac),
                signal(a.placement, 65001, 0, scale),
            )
            with np.load(a.placement / "s65001-m0" / f"l{l}-real.npz") as z:
                np.testing.assert_array_equal(model.A.detach().numpy(), z["A"])
                np.testing.assert_array_equal(model.B.detach().numpy(), z["B"])
            reports.append(
                {"seed": 65001, "mapping": 0, "arm": f"tangent{l}", "exact": True}
            )
    write(a.output, reports)
    print("Bitwise-reproduced fitted checkpoints:", len(reports))


if __name__ == "__main__":
    main()
