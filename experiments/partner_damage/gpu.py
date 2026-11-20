"""Controlled single-update social/body decomposition and partner-path interventions."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.design import ITEMS, LEVELS
from experiments.loreft_life.adapters import advance, signature, vector
from experiments.loreft_omission.gpu import Batch, load_vector, update_roles
from experiments.social_closed_loop.core import BodyValue
from .design import CELLS, GRID, PAIRS, HYBRIDS, factorial, swapped
from .representation import capture
from .sites import train_contrast


def model_from(batch, values):
    m = batch.new()
    load_vector(m, values)
    return m


def differentiated(batch, model, row, h, a):
    p = batch.ordinary(row, h, [model, None, None, None, None, None])
    grad = torch.autograd.grad(p[0, a].log(), model.live())
    return tuple(g.detach() for g in grad), p[0].detach().cpu().numpy()


def direction(model, grads):
    return torch.cat([t.reshape(-1) for t in model.tangent(grads)])


def train_once(batch, event, anchor, folder, body):
    e = event["factual"]
    key = event["key"]
    a = e["action"]
    i = e["row"]
    h = e["health"]
    weights = anchor[key + "_pre"]
    old = model_from(batch, weights)
    grads, prior = differentiated(batch, old, i, h, a)
    swapped_grad, alt_prior = differentiated(batch, old, swapped(i), h, a)
    g1 = direction(old, grads)
    g2 = direction(old, swapped_grad)
    cosine = float(torch.nn.functional.cosine_similarity(g1, g2, dim=0))
    body_cases = factorial(body, e)
    for label, expected in event["cell_records"].items():
        assert all(
            abs(body_cases[label][k] - expected[k]) < 1e-8
            for k in ("after", "signal", "reward")
        )
    facts = []
    steps = []
    den = float(e["prob"][0][a])
    for cell in CELLS:
        m = model_from(batch, weights)
        # Each of the four independently constructed full consequence records yields its own signed update.
        steps.append(
            advance(m, grads, body_cases[cell]["signal"], den, float(prior[a]))
        )
        facts.append(m)
    idx = 0 if e["maintenance"] else 3
    factual = facts[idx]
    strength = e["signal"]
    assert abs(strength - body_cases[CELLS[idx]]["signal"]) < 1e-8
    swapped_m = model_from(batch, weights)
    alt = advance(swapped_m, swapped_grad, strength, den, float(alt_prior[a]))
    # Same signed tangent multiplier before the manifold retraction, even when gradients differ.
    gnorm = steps[idx]["gradient_norm"]
    nominal = steps[idx]["nominal_l2"]
    signed = np.sign(strength) * nominal / gnorm if gnorm else 0.0
    equal = model_from(batch, weights)
    with torch.no_grad():
        for par, g in zip(equal.live(), equal.tangent(swapped_grad), strict=True):
            par.add_(g, alpha=float(signed))
        if signed:
            equal.retract()
    cell_pre = batch.panel([old, *facts, swapped_m], (1, 0))
    refpost = cell_pre[1 + idx]
    pre_error = float(abs(cell_pre[0] - anchor[key + "_prepanel"]).max())
    factual_error = float(abs(refpost - anchor[key + "_postpanel"]).max())
    post_hash = signature(vector(factual))
    expected_hash = signature(anchor[key + "_post"])
    second = batch.panel([old, swapped_m, equal, None, None, None], (2, 5))
    np.savez_compressed(
        folder / "factorial.npz",
        panels=cell_pre[:5],
        identity_train=np.stack([refpost, second[1], second[2]]),
        pre_weights=weights,
        cell_weights=np.stack([vector(m) for m in facts]),
        swapped_weights=vector(swapped_m),
        equal_weights=vector(equal),
        grad_source=g1.cpu().numpy(),
        grad_swapped=g2.cpu().numpy(),
        train_before=prior,
        train_swapped_before=alt_prior,
    )
    return (
        old,
        facts,
        dict(
            cells=body_cases,
            steps=steps,
            swapped_step=alt,
            equal_multiplier=float(signed),
            gradient_cosine=cosine,
            direct_hash_equal=[
                signature(vector(facts[j])) == signature(vector(facts[j + 1]))
                for j in (0, 2)
            ],
            source_update_hash=post_hash,
            source_expected_hash=expected_hash,
            pre_panel_error=pre_error,
            post_panel_error=factual_error,
        ),
    )


@torch.no_grad()
def mix_panel(batch, old, new, event, folder):
    source = ITEMS[event["factual"]["row"]]["partner"]
    panels = np.empty((6, 4, len(GRID), 4), dtype=np.float32)
    cos = np.empty((4, len(GRID)), dtype=np.float64)
    norms = np.empty((4, len(GRID)), dtype=np.float64)

    def parts(common):
        h = common.float()
        pre = (h + old.delta(h)).to(common.dtype).float()
        post = (h + new.delta(h)).to(common.dtype).float()
        return pre, post, post - pre

    for k, h in enumerate(LEVELS):
        rec = [parts(batch.cache[j, k]) for j in range(len(GRID))]
        for j, ix in enumerate(GRID):
            pre, post, delta = rec[j]
            other = rec[PAIRS[j]][2]
            same = ITEMS[ix]["partner"] == source
            norms[k, j] = float(delta.double().norm())
            den = float(delta.double().norm() * other.double().norm())
            cos[k, j] = (
                float(torch.dot(delta.double(), other.double()) / den)
                if den
                else np.nan
            )

            def patch(raw):
                x, y, d = parts(raw[0])
                return torch.stack(
                    [x, y, y - d, (y - d) + d, x + other, y if same else x]
                ).to(raw.dtype)

            vec = patch(batch.cache[j, k].expand(6, -1))
            p = batch.engine._read(int(ix), vec)
            if not batch.q.fast or j == (3 + k):
                batch.engine.custom = patch
                try:
                    native = batch.q.ordinary(int(ix), h, batch.weights)
                finally:
                    batch.engine.custom = None
                if not torch.equal(p, native):
                    raise RuntimeError(
                        "crossed residual projection differs from full forward"
                    )
                p = native
            panels[:, k, j] = p.cpu().numpy()
    np.savez_compressed(
        folder / "swaps.npz", panels=panels, delta_cosine=cos, norms=norms
    )
    return dict(
        variants=HYBRIDS,
        restore_max=float(abs(panels[1] - panels[3]).max()),
        remove_max=float(abs(panels[0] - panels[2]).max()),
    )


def continue_fork(batch, event, tape, models, folder, anchor):
    t = event["turn"]
    states = (35, len(tape) + 1) if t == 33 else (46, len(tape) + 1)
    key = event["key"]
    idx = 0 if event["factual"]["maintenance"] else 3
    roles = [*models, None, None]
    panels = []
    hashes = []
    errors = []
    for n in range(t + 1, len(tape) + 2):
        if n in states:
            now = batch.panel(roles, (2, n % 32))[:4]
            panels.append(now)
            hashes.append([signature(vector(m)) for m in models])
            errors.append(float(abs(now[idx] - anchor[key + f"_d{n}"]).max()))
        if n <= len(tape):
            e = tape[n - 1]
            p = batch.ordinary(e["row"], e["health"], roles)
            update_roles(roles, p, e, range(4))
    np.savez_compressed(
        folder / "continuation.npz", turns=np.asarray(states), panels=np.stack(panels)
    )
    return dict(states=list(states), hashes=hashes, archive_max=errors)


def run(root):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for f, s in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / f).read_bytes()).hexdigest() == s
    events = json.loads((root / "events.json").read_bytes())
    tapes = json.loads((root / "tapes.json").read_bytes())
    body = BodyValue()
    actor = FrozenQwen(root / "expected-weights.json", out)
    batch = Batch(actor, root)
    summary = []
    try:
        equal_route = float(
            (batch.base.birth[0][0] - batch.base.birth[0][1]).abs().max()
        )
        write(
            out / "status.json",
            dict(stage="partner_identity_representations", count=len(events)),
        )
        capture(batch, out)
        with np.load(root / "anchors.npz") as anchors:
            for n, event in enumerate(events):
                key = event["key"]
                folder = out / "cases" / key
                folder.mkdir(parents=True, exist_ok=False)
                write(
                    out / "status.json",
                    dict(
                        stage="factorial_and_positions",
                        case=key,
                        completed=n,
                        total=len(events),
                    ),
                )
                old, models, info = train_once(batch, event, anchors, folder, body)
                index = 0 if event["factual"]["maintenance"] else 3
                info["patch"] = mix_panel(batch, old, models[index], event, folder)
                if event["focus"]:
                    info["continuation"] = continue_fork(
                        batch, event, tapes[str(event["seed"])], models, folder, anchors
                    )
                info["site"] = train_contrast(batch, old, event, folder)
                write(folder / "receipt.json", info)
                summary.append(key)
    finally:
        batch.close()
    actor.verify_unchanged()
    write(
        out / "receipt.json",
        dict(
            cases=summary,
            gpu=torch.cuda.get_device_name(),
            seconds=time.monotonic() - start,
            frozen_backbone=True,
            birth_route_max_error=equal_route,
            peak_bytes=torch.cuda.max_memory_allocated(),
            qwen=batch.q.stats,
            cache_fallback=batch.fallback_at,
            panel_checks=batch.checks,
            panel_max_error=batch.max_error,
            versions={
                k: version(k) for k in ("numpy", "torch", "transformers", "scipy")
            },
        ),
    )
    write(out / "status.json", dict(stage="complete", cases=len(summary)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["socialdamage"], required=True)
    a = p.parse_args()
    run(a.root)
