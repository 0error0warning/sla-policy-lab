"""One matched true-injury step with and without within-frame wording augmentation."""

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
from experiments.first_fatal.engine import NumericalMismatch, Qwen
from experiments.internal_body.capture import encode
from experiments.order_transfer.design import HEALTH
from experiments.order_transfer.gpu import (
    four_gradients,
    from_bits,
    healthy_rollout,
    norm,
)
from experiments.paraphrase_transfer.design import UNSEEN, bank, grid, training_views


def register_texts(backend, items):
    base = len(backend.rows)
    assert items[:base] == backend.rows and backend.full.native is backend.inputs
    extra = [encode(backend.actor, r) for r in items[base:]]
    backend.rows.extend(items[base:])
    backend.inputs.ids.extend(extra)
    backend.inputs.slot = np.array([r["partner"] - 2 for r in items])
    assert len(backend.inputs.ids) == len(backend.rows) == len(backend.inputs.slot)
    assert len({tuple(v) for v in extra}) == len(extra)
    return base


def displacement(model, bits):
    return float(
        np.linalg.norm(
            model.bits().view("<f4").astype(float) - bits.view("<f4").astype(float)
        )
    )


def train(backend, c, lookup, bits, oldbits):
    pre = from_bits(backend, bits)
    old = from_bits(backend, oldbits)
    arms = []
    results = []
    first = None
    directions = []
    orig = c["original_context"]
    h = c["checks"][0][1]
    for augmented in (False, True):
        model = from_bits(backend, bits)
        backend.bind(model, None, None)
        indices = training_views(orig, lookup, c["train_layouts"], augmented)
        try:
            direction, individual, prob = four_gradients(
                backend, model, indices, h, c["chosen"]
            )
            gnorm = norm(direction)
            assert len(individual) == 16 and len(prob) == 16
            if not augmented:
                first = prob[:4]
                assert prob == first * 4 and all(
                    torch.equal(a, b)
                    for j in range(4, 16)
                    for a, b in zip(individual[j], individual[j % 4], strict=True)
                )
            else:
                assert first == prob[:4]
            scale = (
                float(np.sign(c["signal"]) * c["radius"] / gnorm) if gnorm > 0 else 0.0
            )
            with torch.no_grad():
                for par, grad in zip(model.live(), direction, strict=True):
                    par.add_(grad, alpha=scale)
        finally:
            backend.close()
        actual = displacement(model, bits)
        if gnorm:
            assert abs(actual - c["radius"]) < 1e-6
        results.append(
            {
                "indices": indices,
                "view_propensities": prob,
                "gradient_norm": gnorm,
                "scale": scale,
                "actual_l2": actual,
                "post_hash": sha256(model.bits().tobytes()).hexdigest(),
                "zero_gradient": gnorm == 0,
            }
        )
        directions.append(direction)
        arms.append(model)
    norms = [r["gradient_norm"] for r in results]
    cos = (
        sum(
            (a.double() * b.double()).sum().item()
            for a, b in zip(*directions, strict=True)
        )
        / np.prod(norms)
        if all(norms)
        else None
    )
    log = {
        "original": results[0],
        "augmented": results[1],
        "signed_feedback": c["signal"],
        "factual_importance": c["importance_ratio"],
        "quota": c["radius"],
        "cosine": None if cos is None else float(cos),
        "original_matches_pr15": np.array_equal(arms[0].bits(), oldbits),
        "pr15_l2_error": displacement(arms[0], oldbits),
        "updates_per_arm": 1,
        "views_per_arm": 16,
    }
    return (pre, *arms, old), log


def evaluate(backend, c, items, lookup, models, d, n):
    pre, original, aug, old = models
    backend.bind(pre, None, None)
    weights = [m.live() for m in (pre, original, aug, old, pre, aug)]
    try:

        def score(i, h, *, full=False):
            if full:
                prob = backend.ordinary(i, h, weights)
                if backend.fast:
                    if not torch.equal(prob, backend.tail(i, h, weights)):
                        raise NumericalMismatch(
                            "Mismatch on checked wording. Recompute evaluation using complete forwards."
                        )
                    backend.stats["paired_panels"] += 1
            else:
                prob = backend.panel(i, h, weights)
            v = prob.cpu().numpy()
            np.testing.assert_array_equal(v[0], v[4])
            np.testing.assert_array_equal(v[2], v[5])
            return v[:4].copy()

        o = c["original_context"]
        old_h, new_h = (c["checks"][j][1] for j in (0, 1))
        matrix = grid(o, lookup, o["partner"])
        other = grid(o, lookup, 5 - o["partner"])[4:]
        held = backend.inputs.probe_ix
        assert len(held) == 96 and len(other) == len(UNSEEN)
        with torch.no_grad():
            originals = score(c["checks"][0][0], old_h, full=True)
            own = np.array(
                [
                    [
                        score(
                            i,
                            new_h,
                            full=(
                                k == (n + 7 * j) % 24 or (j == 0 and k == o["layout"])
                            ),
                        )
                        for k, i in enumerate(row)
                    ]
                    for j, row in enumerate(matrix)
                ]
            )
            swapped = np.array(
                [
                    [
                        score(i, new_h, full=(k == (n + 5 * j) % 24))
                        for k, i in enumerate(row)
                    ]
                    for j, row in enumerate(other)
                ]
            )
            nxt = score(*c["checks"][2], full=True)
            good = np.array(
                [
                    [
                        score(i, h, full=(j == (n + 19 * k) % 96))
                        for j, i in enumerate(held)
                    ]
                    for k, h in enumerate(HEALTH)
                ]
            )
            lives = healthy_rollout(c, items, score)
        np.savez_compressed(
            d / "probabilities.npz",
            original=originals,
            own=own,
            other=swapped,
            successor=nxt,
            healthy=good,
            own_ix=np.array(matrix),
            other_ix=np.array(other),
            healthy_ix=np.array(held),
        )
        np.savez_compressed(
            d / "post-updates.npz", original=original.bits(), augmented=aug.bits()
        )
        write(d / "healthy-lives.json", lives)
        return np.stack((originals, own[0, o["layout"]], nxt))
    finally:
        backend.close()


def run(root, mode="phrmix"):
    out = root / "output"
    out.mkdir(exist_ok=True, parents=True)
    cases = json.loads((root / "phrase-cases.json").read_bytes())
    assert len(cases) == 48
    for name, h in (
        (
            "architecture.npz",
            "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        ),
        (
            "birth-source.npz",
            "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
        ),
    ):
        assert sha256((root / name).read_bytes()).hexdigest() == h
    start = time.monotonic()
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    diagnostics = {}
    if mode == "phrcheck":
        from experiments.paraphrase_transfer.numerical_repeat import (
            environment,
            original_interface,
            previous_slot_check,
        )

        score_history, token_checks = original_interface(backend, cases, root)
        write(out / "runtime.json", environment())
        write(out / "legacy-token-checks.json", token_checks)
        write(out / "status.json", {"stage": "checking_unmodified_parent_interface"})
        diagnostics["before_text_extension"] = score_history()
    items, lookup = bank(root / "wording.json")
    base = register_texts(backend, items)
    if mode == "phrcheck":
        diagnostics["after_text_extension"] = score_history()
    ids = [
        i
        for i, r in enumerate(items)
        if i >= base or (r["task"], r["wording"]) in ((4, 3), (5, 4))
    ]
    write(
        out / "status.json",
        {"stage": "validating_tokenized_wordings", "contexts": len(ids)},
    )
    fallback = None
    try:
        backend.prepare(ids)
    except NumericalMismatch as exc:
        backend.fast = False
        fallback = str(exc)
    if mode == "phrcheck":
        diagnostics["after_calibration"] = score_history()
        np.savez_compressed(out / "legacy-interface.npz", **diagnostics)
    write(
        out / "calibration.json",
        {
            "exact": backend.fast,
            "fallback": fallback,
            "counts": backend.stats.copy(),
            "contexts": len(ids),
        },
    )
    params = sum(v.numel() for v in actor.model.parameters())
    completed = []
    logs = []
    with (
        np.load(root / "phrase-weights.npz") as saved,
        np.load(root / "phrase-anchors.npz") as z,
    ):
        anchors = z["probabilities"]
        for n, c in enumerate(cases):
            bits, old = (saved[c[key]] for key in ("pre_hash", "pr15_hash"))
            assert all(
                sha256(bits_.tobytes()).hexdigest() == c[k]
                for bits_, k in ((bits, "pre_hash"), (old, "pr15_hash"))
            )
            models, log = train(backend, c, lookup, bits, old)
            d = out / "cases" / f"s{c['seed']}" / c["birth"]
            d.mkdir(parents=True, exist_ok=True)
            write(d / "update.json", log)
            logs.append(log)
            completed.append((c, models, d, n))
            write(
                out / "status.json",
                {"stage": "paired_wording_updates", "completed": n + 1},
            )

        def evaluate_all():
            for c, models, d, n in completed:
                v = evaluate(backend, c, items, lookup, models, d, n)
                if mode == "phrcheck":
                    slots = previous_slot_check(backend, c, models)
                    np.savez_compressed(
                        d / "slot-check.npz",
                        alternate=slots,
                        current=v[:, [1, 3]].transpose(1, 0, 2),
                    )
                err = np.max(abs(v[:, [0, 3]] - anchors[n]), axis=2)
                write(
                    d / "anchors.json",
                    {
                        "identical_contexts": int(np.all(err == 0, axis=1).sum()),
                        "max_pre": float(err[:, 0].max()),
                        "max_pr15": float(err[:, 1].max()),
                        "original_vs_pr15_max": float(np.max(abs(v[:, 1] - v[:, 3]))),
                    },
                )
                write(
                    out / "status.json",
                    {"stage": "held_wordings_successors_and_tasks", "completed": n + 1},
                )

        try:
            evaluate_all()
        except NumericalMismatch as exc:
            if not backend.fast:
                raise
            backend.fast = False
            fallback = str(exc)
            write(
                out / "status.json",
                {
                    "stage": "repeating_evaluation_with_complete_Qwen",
                    "reason": fallback,
                },
            )
            evaluate_all()
    actor.verify_unchanged()
    assert params == sum(v.numel() for v in actor.model.parameters())
    write(
        out / "receipt.json",
        {
            "mode": mode,
            "cases": len(cases),
            "parameters": params,
            "frozen_backbone": True,
            "views_per_arm": 16,
            "steps_per_case_per_arm": 1,
            "total_steps": 96,
            "native_gradient_forwards": len(cases) * 32,
            "parent_exact_updates": sum(x["original_matches_pr15"] for x in logs),
            "total_cached_contexts": len(backend.cache),
            "max_budget_difference": max(
                abs(x["original"]["actual_l2"] - x["augmented"]["actual_l2"])
                for x in logs
            ),
            "zero_gradients": sum(
                x[r]["zero_gradient"] for x in logs for r in ("original", "augmented")
            ),
            "counts": backend.stats,
            "fallback": fallback,
            "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "versions": {
                k: version(k) for k in ("numpy", "torch", "transformers", "scipy")
            },
        },
    )
    write(out / "status.json", {"stage": "complete", "completed": 48})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["phrmix", "phrcheck"], required=True)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.mode)
