"""Replay the continuous tapes and independently refit selected actual checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.policy import feedback_scores
from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import InternalPolicy, arr, fit
from experiments.lethal_boundary.design import ANCHOR_Q, ARMS, CROSS, levels
from experiments.lethal_boundary.jitter import CELLS, SEEDS, LocalHealthPolicy, signals
from experiments.lethal_boundary.states import Bank, at, probes
from experiments.social_closed_loop.core import BodyValue


def independent_settle(before, action, response, noise):
    h = before[:, :, None]
    gain = 1 + np.take([0, 0.25, 1.75, 1.75], action) * (1 + noise[..., 0]) / 350
    for t in range(1, 5):
        gain *= 1 + (response != 1) * (1 + noise[..., t]) / 350
    return np.maximum(0.0, 700 - (700 - h) * gain)


def run(cache, arch, core, jitter, out):
    bank = Bank(cache, arch)
    body = BodyValue()
    replay = []
    total = 0
    observed = 0
    maxerr = 0.0
    steps = 0
    for s in SEEDS:
        folder = jitter / f"s{s}"
        with np.load(folder / "tape.npz") as z:
            a, mu, uniform, noise = draw_tape(s, bank.training[None][2])
            for k, x in zip(
                ("action", "mu", "uniform", "noise"),
                (a, mu, uniform, noise),
                strict=True,
            ):
                np.testing.assert_array_equal(z[k], x)
            slot = z["profile"]
            fx, jy = z["fixed"], z["jitter"]
            assert fx.shape == jy.shape == (8, 768)
            np.testing.assert_array_equal(jy[4:], fx[4:])
            np.testing.assert_array_equal(jy[:4] < 2.44, fx[:4] < 2.44)
            grp = np.array(
                [
                    r["partner"] * 16 + r["frame"] * 8 + r["task"] * 2 + slot[i]
                    for i, r in enumerate(bank.train)
                ]
            )
            for k in np.unique(grp):
                np.testing.assert_allclose(
                    jy[:, grp == k].mean(1), fx[:, 0], atol=1e-12, rtol=0
                )
            for j in range(4):
                assert len(np.unique(jy[j])) == 768
            assert len(np.unique(jy[:4])) == 3072
        observed += a.size
        for m in (0, 1):
            loc = folder / f"m{m}"
            q = ANCHOR_Q[slot]
            yes = np.array([2 + (r["partner"] ^ r["frame"] ^ m) for r in bank.train])
            p = np.where(
                a == 0,
                0.0,
                np.where(
                    a == 1,
                    0.15,
                    np.where(a == yes[None, :, None], q[None, :, None], 0.1),
                ),
            )
            resp = np.where(a == 0, -1, uniform < p)
            with np.load(loc / "outcomes.npz") as z:
                np.testing.assert_array_equal(z["response"], resp)
                for mode, h in (("fixed", fx), ("jitter", jy)):
                    y = independent_settle(h, a, resp, noise)
                    reward = body.interpolate(y, body.reference) - body.interpolate(
                        h[:, :, None], body.reference
                    )
                    maxerr = max(maxerr, float(abs(y - z[mode + "_after"]).max()))
                    np.testing.assert_allclose(
                        y, z[mode + "_after"], atol=2e-12, rtol=0
                    )
                    np.testing.assert_allclose(
                        reward, z[mode + "_reward"], atol=1e-10, rtol=0
                    )
                    total += a.size
                if (s, m) == (SEEDS[-1], 1):
                    input0 = bank.record_train(slot)
                    for label, (sensor, source) in CELLS.items():
                        h = fx if source == "fixed" else jy
                        target = signals(a, z[source + "_reward"], mu, h, body)
                        before = fx if sensor == "fixed" else jy
                        model = LocalHealthPolicy(bank, "linear")
                        fit(model, {**input0, "g": arr(basis(before), "cpu")}, target)
                        replay.append(compare(model, loc / label, bank, "record", s, m))
            log = json.loads((loc / "steps.json").read_bytes())
            for label in CELLS:
                assert len(log[label]) == 75 and np.isfinite(log[label]).all()
                with np.load(loc / (label + ".npz")) as model:
                    assert (
                        model["A"].shape == (2, 12, 33, 4)
                        and model["B"].shape == (4, 2560)
                        and np.linalg.norm(model["B"]) > 0
                    )
                steps += len(log[label])
    for s, m in ((67001, 0), (67120, 1)):
        loc = core / f"s{s}/m{m}"
        with np.load(loc.parent / "tape.npz") as t, np.load(loc / "outcomes.npz") as z:
            slot = t["profile"]
            actions = t["action"]
            mu = t["mu"]
            for sense in ("plain", "record"):
                g = bank.plain_train if sense == "plain" else bank.record_train(slot)
                for label in (*ARMS, *CROSS) if sense == "plain" else ARMS:
                    sensor, source = CROSS.get(label, (label, label))
                    hs = levels(source, s)
                    if source == "M":
                        reward = np.stack(
                            [
                                z[("D" if h == levels("D", s)[j] else "R") + "_reward"][
                                    j
                                ]
                                for j, h in enumerate(hs)
                            ]
                        )
                    else:
                        reward = z[source + "_reward"]
                    target = feedback_scores(
                        actions, reward, mu, np.minimum(1.0, [body.rul(h) for h in hs])
                    )
                    model = InternalPolicy(bank, "linear")
                    fit(model, at(g, levels(sensor, s)), target)
                    replay.append(
                        compare(model, loc / (sense + "-" + label), bank, sense, s, m)
                    )
    out.mkdir(exist_ok=True, parents=True)
    result = {
        "checked_jitter_seeds": list(SEEDS),
        "unique_continuous_focal_values_per_seed": 3072,
        "jitter_sampled_choices": observed,
        "jitter_physical_settlements": total,
        "jitter_optimizer_steps": steps,
        "max_inverse_body_error": maxerr,
        "refits": replay,
    }
    write(out / "replay.json", result)
    print(
        "Verified",
        len(replay),
        "training refits and",
        total,
        "continuous-exposure settlements.",
    )


def compare(model, stem, bank, sense, s, m):
    with np.load(stem.with_suffix(".npz")) as original:
        err = {
            k: float(np.max(np.abs(v.detach().numpy() - original[k])))
            for k, v in model.named_parameters()
        }
    with np.load(stem.with_name(stem.name + "-probe.npz")) as saved:
        pi = probes(model, bank, sense)
        err.update({k: float(np.abs(v - saved[k]).max()) for k, v in pi.items()})
    assert max(err.values()) < 2e-5, err
    return {
        "seed": s,
        "mapping": m,
        "label": stem.name,
        "max_errors": err,
        "exact": max(err.values()) == 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("cache", "arch", "core", "jitter", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(a.cache, a.arch, a.core, a.jitter, a.output)
