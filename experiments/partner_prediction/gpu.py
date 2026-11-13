"""Same-history interventions and 24 autonomous, irreversible Qwen lives."""

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
from experiments.individual_life.design import action_at
from experiments.lifetime_cooperation.design import (
    ITEMS,
    PANEL,
    PILOT,
    physical_step,
    rules,
    schedule,
)
from experiments.loreft_life.adapters import advance, signature, vector
from experiments.loreft_life.gpu import save_events
from experiments.partner_damage.design import swapped
from experiments.social_closed_loop.core import BodyValue

from .design import ARMS, SHADOWS, Forecast
from .model import Actor


class Participants:
    def __init__(self, actor):
        self.policies = {arm: actor.new(arm) for arm in ARMS}
        self.heads = {k: Forecast(k) for k in (*ARMS, *SHADOWS)}


def update_policy(model, gradients, signal, behavior, current):
    return advance(
        model, tuple(g.detach() for g in gradients), signal, behavior, current
    )


def observe(heads, keys, x, e):
    i, a, y = e["row"], e["action"], e["maintenance"]
    rows = {}
    for k in keys:
        model = heads[k]
        before = np.stack([model.read(x, j) for j in (i, swapped(i))])
        if e["after"] > 0:
            model.observe(x, i, a, y)
        after = np.stack([model.read(x, j) for j in (i, swapped(i))])
        rows[k] = {
            "before": before.tolist(),
            "after": after.tolist(),
            "count": model.updates,
        }
    return rows


def replay(actor, seed, people, src, folder):
    logs = []
    # Lane 3 is the original LoReFT lane used in #19. Other lanes have the same frozen backbone.
    slots = {"body": 3, "true": 1, "wrong": 2}
    for k, e in enumerate(src):
        i, h, a = e["row"], e["health"], e["action"]
        heads = people.heads
        roles = [(None, None)] * 6
        for name, j in slots.items():
            roles[j] = (people.policies[name], heads[name].read(actor.x, i))
        p = actor.ordinary(i, h, actor.lanes(roles))
        probs = p.detach().cpu().numpy().copy()
        if k == 0:
            assert np.array_equal(probs, np.broadcast_to(probs[0], probs.shape))
        stats = {}
        if e["after"] > 0:
            grads = {
                name: torch.autograd.grad(
                    p[j, a].log(),
                    people.policies[name].live(),
                    retain_graph=name != "wrong",
                )
                for name, j in slots.items()
            }
            for name, j in slots.items():
                stats[name] = update_policy(
                    people.policies[name],
                    grads[name],
                    e["signal"],
                    e["behavior"][a],
                    float(probs[j, a]),
                )
        observed = observe(heads, (*ARMS, *SHADOWS), actor.x, e)
        logs.append(
            dict(
                **e,
                policy={s: probs[j].tolist() for s, j in slots.items()},
                updates=stats,
                forecasts=observed,
            )
        )
    save_events(folder / "replay.jsonl", logs)
    return np.stack([v["policy"]["body"] for v in logs])


def live(actor, seed, arm, people, tape, body, folder):
    ids, uniforms, noises, _ = tape
    h, total = 350.0, 0.0
    logs = []
    policy = people.policies[arm]
    head = people.heads[arm]
    for k, idx in enumerate(ids):
        i = int(idx)
        roles = [(None, None)] * 6
        roles[3] = (policy, head.read(actor.x, i))
        p = actor.ordinary(i, h, actor.lanes(roles))
        probs = p[3].detach().cpu().numpy().copy()
        birth = p[0].detach().cpu().numpy().copy()
        if k == 0:
            assert np.array_equal(probs, birth)
        u, v = map(float, uniforms[k])
        a = action_at(probs, u)
        bits = rules(seed, "stable", k)
        after, _, maintained, rescued = physical_step(
            h, a, ITEMS[i], bits, v, noises[k]
        )
        reward = body.rul(after) - body.rul(h)
        baseline = (-8 + total) / (8 + k)
        signal = reward - baseline
        e = {
            "turn": k + 1,
            "row": i,
            "health": h,
            "after": after,
            "action": a,
            "maintenance": maintained,
            "noise": noises[k].tolist(),
            "action_u": u,
            "partner_u": v,
            "rescued_answer": rescued,
            "rules": bits,
            "reward": reward,
            "baseline": baseline,
            "signal": signal,
            "policy": probs.tolist(),
            "birth": birth.tolist(),
        }
        if after > 0:
            grad = torch.autograd.grad(p[3, a].log(), policy.live())
            e["update"] = update_policy(
                policy, grad, signal, float(probs[a]), float(probs[a])
            )
        keys = (arm, *SHADOWS) if arm == "true" else (arm,)
        e["forecasts"] = observe(people.heads, keys, actor.x, e)
        logs.append(e)
        total += reward
        h = after
        if h == 0:
            break
    assert h == 0 and abs(total + body.rul(350)) < 1e-8
    save_events(folder / (arm + ".jsonl"), logs)
    return len(logs)


def snapshot(actor, people, folder, check):
    y = actor.panels(people.policies, people.heads, check)
    forecast = {
        k: np.stack([[s.read(actor.x, int(i)) for i in ix] for ix in PANEL])
        for k, s in people.heads.items()
    }
    np.savez_compressed(folder / "panel.npz", policy=y, **forecast)
    weights = {"policy_" + k: vector(m) for k, m in people.policies.items()}
    weights.update({"head_" + k: v.w for k, v in people.heads.items()})
    np.savez_compressed(folder / "weights.npz", **weights)
    write(
        folder / "counts.json",
        {
            "forecast_updates": {k: m.updates for k, m in people.heads.items()},
            "policy_hash": {
                k: signature(vector(m)) for k, m in people.policies.items()
            },
        },
    )
    return y


def run(root, mode):
    assert mode == "partnerpred"
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for name, digest in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    tapes = json.loads((root / "tapes.json").read_bytes())
    body = BodyValue()
    write(out / "status.json", {"stage": "frozen_model_and_features"})
    frozen = FrozenQwen(root / "expected-weights.json", out)
    actor = Actor(frozen, root, out)
    sizes = {}
    with np.load(root / "archive-ends.npz") as prev:
        try:
            for si, seed in enumerate(PILOT):
                folder = out / "lives" / f"s{seed}"
                rep, aut = folder / "replay", folder / "live"
                rep.mkdir(parents=True)
                aut.mkdir()
                write(
                    out / "status.json",
                    {"stage": "paired_replay", "seed": seed, "completed": si},
                )
                people = Participants(actor)
                orig = replay(actor, seed, people, tapes[str(seed)], rep)
                values = snapshot(actor, people, rep, si)
                diff = abs(orig - prev[f"s{seed}_actual"])
                panel_diff = abs(values[0, 1] - prev[f"s{seed}_held"])
                write(
                    rep / "legacy.json",
                    {
                        "max_actual": float(diff.max()),
                        "mean_actual": float(diff.mean()),
                        "max_held": float(panel_diff.max()),
                        "mean_held": float(panel_diff.mean()),
                    },
                )
                people = Participants(actor)
                tape = schedule(seed)
                sizes[seed] = {}
                for name in ARMS:
                    write(
                        out / "status.json",
                        {
                            "stage": "autonomous_life",
                            "seed": seed,
                            "arm": name,
                            "completed": si,
                        },
                    )
                    sizes[seed][name] = live(actor, seed, name, people, tape, body, aut)
                snapshot(actor, people, aut, si + 11)
        finally:
            actor.b.close()
    frozen.verify_unchanged()
    write(
        out / "receipt.json",
        {
            "mode": mode,
            "device": torch.cuda.get_device_name(),
            "seeds": PILOT,
            "autonomous_lengths": sizes,
            "elapsed_seconds": time.monotonic() - started,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "panel_checks": actor.checks,
            "max_checked_error": actor.error,
            "panel_fallback": actor.fallback,
            "initial_fallback": actor.b.engine.fallback,
            "frozen_backbone": True,
            "qwen_stats": actor.b.q.stats,
            "versions": {
                k: version(k) for k in ("torch", "numpy", "scipy", "transformers")
            },
        },
    )
    write(
        out / "status.json",
        {
            "stage": "complete",
            "histories": len(PILOT),
            "autonomous_lives": 3 * len(PILOT),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["partnerpred"], required=True)
    a = p.parse_args()
    run(a.root, a.mode)
