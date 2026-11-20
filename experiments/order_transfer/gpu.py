"""Equal-budget one-injury training on native Qwen gradients; paired healthy rollouts."""

import argparse
import json
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import settle, world
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.first_fatal.study import arrivals
from experiments.individual_life.design import action_at, offers
from experiments.individual_life.model import derivatives, update
from experiments.order_transfer.design import HEALTH, evaluation_indices, item_bank


def from_bits(backend, bits):
    model = backend.new("blank")
    n = model.A.numel()
    with torch.no_grad():
        for p, x in zip(model.live(), (bits[:n], bits[n:]), strict=True):
            p.copy_(torch.as_tensor(x.view("<f4"), device=p.device).view_as(p))
    np.testing.assert_array_equal(bits, model.bits())
    return model


def norm(grad):
    return float(np.sqrt(sum(torch.sum(g.double().square()).item() for g in grad)))


def four_gradients(backend, model, orders, h, choice):
    ref = model.snapshot()
    slots = [ref, ref, ref, model.live(), ref, ref]
    values = []
    probability = []
    for index in orders:
        q = backend.ordinary(index, h, slots)
        probability.append(float(q[3, choice].detach().item()))
        values.append(derivatives(q, [(3, choice, model)])[0])
    mean = tuple(
        torch.stack(part).double().mean(dim=0).float()
        for part in zip(*values, strict=True)
    )
    return mean, values, probability


def train(backend, c, indices, bits, archive):
    pre = from_bits(backend, bits)
    repeat = from_bits(backend, bits)
    rotate = from_bits(backend, bits)
    old = from_bits(backend, archive)
    original = c["original_context"]["layout"]
    h = c["checks"][0][1]
    signal = c["signal"]
    chosen = c["chosen"]
    backend.bind(repeat, None, None)
    try:
        g, gs, p = four_gradients(backend, repeat, [indices[original]] * 4, h, chosen)
        avg, gs_rot, v = four_gradients(
            backend, rotate, [indices[k] for k in c["train_layouts"]], h, chosen
        )
        assert (
            all(torch.equal(a, b) for x in gs for a, b in zip(g, x, strict=True))
            and p == [p[0]] * 4
        )
        assert p[0] == v[0] and all(
            torch.equal(a, b) for a, b in zip(g, gs_rot[0], strict=True)
        )
        step = update(repeat, g, signal, c["behavior_probability"], p[0])
        radius = step["nominal_step_norm"]
        nm = norm(avg)
        cosine = None
        if nm > 0:
            if norm(g) > 0:
                cosine = sum(
                    (a.double() * b.double()).sum().item()
                    for a, b in zip(g, avg, strict=True)
                ) / (norm(g) * nm)
            scale = float(np.sign(signal) * radius / nm)
            with torch.no_grad():
                for param, grad in zip(rotate.live(), avg, strict=True):
                    param.add_(grad, alpha=scale)
        else:
            scale = 0.0
    finally:
        backend.close()
    post_repeat = repeat.bits()
    post_rotate = rotate.bits()

    def displacement(x, y):
        return float(
            np.linalg.norm(x.view("<f4").astype(float) - y.view("<f4").astype(float))
        )

    diff = displacement(bits, post_repeat), displacement(bits, post_rotate)
    if nm > 0:
        assert abs(diff[0] - diff[1]) < 1e-6
    log = {
        "views_per_arm": 4,
        "updates_per_arm": 1,
        "original_propensity": p[0],
        "rotated_propensities": v,
        "fixed_signal": signal,
        "historical_propensity_delta": p[0] - c["historical_training_probability"],
        "common_importance_ratio": step["applied_ratio"],
        "quota": radius,
        "single_gradient_norm": norm(g),
        "mean_rotation_gradient_norm": nm,
        "rotation_gradient_norms": [norm(gg) for gg in gs_rot],
        "gradient_cosine": cosine,
        "rotate_multiplier": scale,
        "repeat_multiplier": step["parameter_multiplier"],
        "repeat_step": diff[0],
        "rotate_step": diff[1],
        "zero_mean_gradient": nm == 0.0,
        "repeat_matches_archived": bool(np.array_equal(post_repeat, archive)),
        "repeat_minus_archived_l2": displacement(post_repeat, archive),
        "repeat_hash": sha256(post_repeat.tobytes()).hexdigest(),
        "rotate_hash": sha256(post_rotate.tobytes()).hexdigest(),
    }
    return (pre, repeat, rotate, old), log


def healthy_rollout(c, items, evaluate):
    results = []
    for role in range(3):
        h = 350.0
        events = []
        for t, (i, uniform, noise, _) in enumerate(
            arrivals(c["evaluation_seed"], items), 1
        ):
            prob = evaluate(i, h, full=(t == 1 and role == 0))[role]
            act = action_at(prob, uniform[0])
            chances, _ = offers(items[i], world(c["seed"]))
            m = bool(uniform[1] < chances[act]) if act else None
            after, _, _ = settle(h, act, m, noise)
            assert 90 <= h and 0 < after < h
            events.append(
                {
                    "turn": t,
                    "index": i,
                    "health": h,
                    "after": after,
                    "action": act,
                    "maintained": m,
                    "uniform": float(uniform[0]),
                    "partner_uniform": float(uniform[1]),
                    "probabilities": prob.tolist(),
                    "completed": bool(act >= 2 and after > 0),
                }
            )
            h = after
            if h < 90:
                break
        results.append(events)
    return results


def evaluate_case(backend, c, indices, held, items, models, out, case_id):
    pre, repeat, rotate, old = models
    backend.bind(pre, None, None)
    weights = [m.live() for m in (pre, repeat, rotate, old, pre, rotate)]
    try:

        def score(i, h, *, full=False):
            if full:
                v = backend.ordinary(i, h, weights)
                if backend.fast:
                    cached = backend.tail(i, h, weights)
                    if not torch.equal(v, cached):
                        raise NumericalMismatch(
                            "readout differs on complete forward; repeat the evaluation phase"
                        )
                    backend.stats["paired_panels"] += 1
            else:
                v = backend.panel(i, h, weights)
            p = v.cpu().numpy()
            np.testing.assert_array_equal(p[0], p[4])
            np.testing.assert_array_equal(p[2], p[5])
            return p[:4].copy()

        with torch.no_grad():
            order = c["original_context"]["layout"]
            risk = np.array(
                [
                    [
                        score(i, h, full=(o in (order, (order + 11) % 24)))
                        for o, i in enumerate(indices)
                    ]
                    for _, h in c["checks"][:2]
                ]
            )
            i, h = c["checks"][2]
            next_req = score(i, h, full=True)
            good = np.array(
                [
                    [
                        score(i, level, full=(j == (case_id + 17 * k) % len(held)))
                        for j, i in enumerate(held)
                    ]
                    for k, level in enumerate(HEALTH)
                ]
            )
            lives = healthy_rollout(c, items, score)
        np.savez_compressed(
            out / "probabilities.npz",
            risk=risk,
            successor=next_req,
            healthy=good,
            healthy_indices=np.asarray(held),
            risk_indices=np.asarray(indices),
        )
        np.savez_compressed(
            out / "post-updates.npz", repeat=repeat.bits(), rotate=rotate.bits()
        )
        write(out / "healthy-lives.json", lives)
    finally:
        backend.close()


def run(root):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((root / "order-cases.json").read_bytes())
    assert len(cases) == 48
    for file, expected in (
        (
            "architecture.npz",
            "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        ),
        (
            "birth-source.npz",
            "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
        ),
    ):
        assert sha256((root / file).read_bytes()).hexdigest() == expected
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    items, lookup = item_bank()
    assert backend.rows == items
    write(out / "status.json", {"stage": "calibrating_native_readouts"})
    fallback = None
    try:
        backend.prepare()
    except NumericalMismatch as exc:
        backend.fast = False
        fallback = str(exc)
    write(
        out / "calibration.json",
        {"exact": backend.fast, "fallback": fallback, "counts": backend.stats.copy()},
    )
    params = sum(v.numel() for v in actor.model.parameters())
    logs = []
    bycase = []
    with (
        np.load(root / "order-weights.npz") as data,
        np.load(root / "order-anchors.npz") as z,
    ):
        anchors = z["probabilities"]
        for n, c in enumerate(cases):
            before, old = (data[c[k]] for k in ("pre_hash", "historical_post_hash"))
            assert all(
                sha256(x.tobytes()).hexdigest() == c[k]
                for x, k in ((before, "pre_hash"), (old, "historical_post_hash"))
            )
            indices, held = evaluation_indices(c, items, lookup)
            models, log = train(backend, c, indices, before, old)
            d = out / "cases" / (f"s{c['seed']}") / c["birth"]
            d.mkdir(parents=True, exist_ok=True)
            bycase.append((c, indices, held, models, d, n))
            logs.append(log)
            write(d / "update.json", log)
            write(
                out / "status.json",
                {"stage": "paired_gradient_updates", "completed": n + 1},
            )

        def evaluate_all():
            for c, indices, held, models, d, n in bycase:
                evaluate_case(backend, c, indices, held, items, models, d, n)
                with np.load(d / "probabilities.npz") as zq:
                    orig = c["original_context"]["layout"]
                    a = np.stack(
                        (zq["risk"][0, orig], zq["risk"][1, orig], zq["successor"])
                    )
                err = np.max(abs(a[:, [0, 3]] - anchors[n]), axis=2)
                write(
                    d / "anchors.json",
                    {
                        "pre_max": float(err[:, 0].max()),
                        "archived_post_max": float(err[:, 1].max()),
                        "both_exact_contexts": int(np.all(err == 0, axis=1).sum()),
                        "repeat_minus_archived_max": float(
                            np.max(abs(a[:, 1] - a[:, 3]))
                        ),
                    },
                )
                write(
                    out / "status.json",
                    {"stage": "held_orders_and_healthy_lives", "completed": n + 1},
                )

        try:
            evaluate_all()
        except NumericalMismatch as exc:
            if not backend.fast:
                raise
            fallback = str(exc)
            backend.fast = False
            write(
                out / "status.json",
                {
                    "stage": "repeating_all_scores_with_complete_model",
                    "reason": fallback,
                },
            )
            evaluate_all()
    actor.verify_unchanged()
    assert params == sum(v.numel() for v in actor.model.parameters())
    write(
        out / "receipt.json",
        {
            "mode": "ordmix",
            "cases": len(cases),
            "views_per_arm": 4,
            "steps_per_arm_per_case": 1,
            "total_optimizer_steps": 96,
            "native_gradient_forwards": 8 * len(cases),
            "frozen_backbone": True,
            "params": params,
            "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "readout_fallback": fallback,
            "counts": backend.stats,
            "max_budget_error": max(
                abs(x["repeat_step"] - x["rotate_step"]) for x in logs
            ),
            "historically_exact_updates": sum(
                x["repeat_matches_archived"] for x in logs
            ),
            "zero_mean_gradients": sum(x["zero_mean_gradient"] for x in logs),
            "versions": {
                k: version(k) for k in ("torch", "transformers", "numpy", "scipy")
            },
        },
    )
    write(out / "status.json", {"stage": "complete", "completed": 48})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["ordmix"], required=True)
    run(parser.parse_args().root)
