"""Chronological LoReFT counterfactuals. A skipped write never changes the factual body or reward."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import NumericalMismatch
from experiments.lifetime_cooperation.design import LEVELS
from experiments.loreft_life.adapters import LoReFT, advance, frame, signature, vector
from experiments.loreft_life.engine import Engine, People
from experiments.loreft_life.gpu import save_events
from .design import (
    GRID,
    PILOT,
    branches,
    branch_key,
    event_plan,
    histories,
    reference_age,
)


def load_vector(model, array):
    offset = 0
    with torch.no_grad():
        for t in model.live():
            end = offset + t.numel()
            t.copy_(
                torch.as_tensor(
                    array[offset:end].reshape(tuple(t.shape)), device=t.device
                )
            )
            offset = end
    assert offset == len(array)


class Batch:
    def __init__(self, actor, root):
        self.engine = Engine(actor, root)
        self.q = self.engine.q
        try:
            self.q.prepare(sorted(set(GRID.tolist())))
        except NumericalMismatch as exc:
            self.q.fast = False
            self.engine.fallback.append(str(exc))
        self.base = People(self.q)
        self.engine.bind(self.base)
        self.weights = [self.base.birth] * 6
        with torch.no_grad():
            self.cache = {
                (j, k): self.engine.common_residual(int(i), h).clone()
                for k, h in enumerate(LEVELS)
                for j, i in enumerate(GRID)
            }
        self.panel_calls = 0
        self.checks = 0
        self.max_error = 0.0
        self.fallback_at = None

    def new(self):
        return LoReFT(frame("cuda"))

    def close(self):
        self.engine.close()

    @staticmethod
    def inject(x, roles):
        h = x.float()
        return torch.stack(
            [
                h[j] + m.delta(h[j]) if m is not None else h[j]
                for j, m in enumerate(roles)
            ]
        ).to(x.dtype)

    def ordinary(self, i, h, roles):
        self.engine.custom = lambda x: self.inject(x, roles)
        try:
            return self.q.ordinary(i, h, self.weights)
        finally:
            self.engine.custom = None

    @torch.no_grad()
    def panel(self, roles, check):
        # Each lane is a history fork. All six lanes have identical prompt and body.
        result = np.empty((6, len(LEVELS), len(GRID), 4), dtype=np.float32)
        device_values = []
        self.panel_calls += 1
        for k, h in enumerate(LEVELS):
            for j, idx in enumerate(GRID):
                i = int(idx)
                if not self.q.fast:
                    p = self.ordinary(i, h, roles)
                else:
                    common = self.cache[j, k]
                    rr = self.inject(common.expand(6, -1), roles)
                    p = self.engine._read(i, rr)
                    if check is True or (isinstance(check, tuple) and (k, j) == check):
                        full = self.ordinary(i, h, roles)
                        err = float((p - full).abs().max())
                        self.max_error = max(self.max_error, err)
                        self.checks += 1
                        if err:
                            self.fallback_at = [self.panel_calls, k, j, err]
                            self.q.fast = False
                            return self.panel(roles, False)
                        p = full
                device_values.append(p)
        result[:] = (
            torch.stack(device_values)
            .reshape(len(LEVELS), len(GRID), 6, 4)
            .permute(2, 0, 1, 3)
            .cpu()
            .numpy()
        )
        return result


def decision(batch, roles, e):
    p = batch.ordinary(e["row"], e["health"], roles)
    return p, p.detach().cpu().numpy().copy()


def update_roles(roles, p, e, indices):
    jobs = []
    for pos, k in enumerate(indices):
        m = roles[k]
        a = e["action"]
        grads = torch.autograd.grad(
            p[k, a].log(), m.live(), retain_graph=pos < len(indices) - 1
        )
        jobs.append((k, tuple(t.detach() for t in grads)))
    return {
        k: advance(
            roles[k],
            grad,
            e["signal"],
            float(e["prob"][0][e["action"]]),
            float(p[k, e["action"]].detach()),
        )
        for k, grad in jobs
    }


def reference(batch, src, ref, oldbirth, oldfinal, folder):
    model = batch.new()
    roles = [None, None, None, model, None, None]
    trace = []
    weights = []
    preqs = []
    logs = []
    birth_actual = []
    max_action = 0.0
    hash_hits = 0
    for k, e in enumerate(src):
        weights.append(vector(model))
        check = True if k == 0 else (k % 4, (7 * k) % 32)
        trace.append(batch.panel(roles, check)[3])
        p, numbers = decision(batch, roles, e)
        preqs.append(numbers[3])
        birth_actual.append(numbers[0])
        if k >= len(ref):
            break
        max_action = max(
            max_action,
            float(abs(numbers[3] - np.array(ref[k]["probabilities"][3])).max()),
        )
        stats = update_roles(roles, p, e, [3])[3]
        hash_hits += int(
            stats["before"] == ref[k]["updates"]["loreft"]["before"]
            and stats["after"] == ref[k]["updates"]["loreft"]["after"]
        )
        logs.append(dict(turn=k + 1, **stats))
    trace = np.asarray(trace)
    weights = np.asarray(weights)
    preqs = np.asarray(preqs)
    birth_actual = np.asarray(birth_actual)
    np.savez_compressed(
        folder / "normal.npz",
        panels=trace,
        actual=preqs,
        birth_actual=birth_actual,
        weights=weights,
    )
    save_events(folder / "normal-updates.jsonl", logs)
    comparison = dict(
        turns=len(ref),
        hash_hits=hash_hits,
        max_action_drift=max_action,
        birth_panel_error=float(abs(trace[0] - oldbirth).max()),
        last_panel_error=float(abs(trace[-1] - oldfinal).max()),
    )
    write(folder / "anchors.json", comparison)
    return trace, weights, preqs, birth_actual, comparison


def fork_group(batch, src, weights, panels, original, group, folder):
    omits = [set(o) for o in group]
    starts = [min(v) for v in omits]
    models = [batch.new() for _ in group] + [None] * (6 - len(group))
    output = [dict(panels=[], actual=[], logs=[]) for _ in group]
    n = len(src)
    earliest = min(starts)
    for t in range(earliest, n + 1):
        e = src[t - 1]
        for k, first in enumerate(starts):
            if t == first:
                load_vector(models[k], weights[t - 1])
        active = [k for k, s in enumerate(starts) if t >= s]
        roles = [models[k] if k in active else None for k in range(6)]
        p, numbers = decision(batch, roles, e)
        if t > earliest:
            views = batch.panel(roles, ((t + earliest) % 4, (t * 11 + earliest) % 32))
        for k in active:
            if t == starts[k]:
                # Clone equality is checked on this branch's own current input as well.
                assert np.array_equal(numbers[k], original[t - 1])
                output[k]["panels"].append(panels[t - 1])
            else:
                output[k]["panels"].append(views[k])
            output[k]["actual"].append(numbers[k])
        if t == n:
            break
        apply = [k for k in active if t not in omits[k]]
        result = update_roles(roles, p, e, apply)
        for k in active:
            if k not in result:
                checksum = signature(vector(models[k]))
                result[k] = dict(before=checksum, after=checksum, actual_l2=0.0)
            output[k]["logs"].append(dict(turn=t, skipped=k not in apply, **result[k]))
    for k, om in enumerate(group):
        dest = folder / branch_key(om)
        dest.mkdir(parents=True)
        np.savez_compressed(
            dest / "trace.npz",
            panels=np.asarray(output[k]["panels"]),
            actual=np.asarray(output[k]["actual"]),
            final_weights=vector(models[k]),
            first=starts[k],
            omitted=om,
        )
        save_events(dest / "updates.jsonl", output[k]["logs"])


def run(root, mode):
    assert mode == "omitstable"
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tapes = histories(root)
    frac = reference_age(tapes)
    for f, h in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / f).read_bytes()).hexdigest() == h
    actor = FrozenQwen(root / "expected-weights.json", out)
    write(out / "status.json", dict(stage="calibrating", seed_count=len(tapes)))
    batch = Batch(actor, root)
    nbranch = 0
    ns = 0
    try:
        for seed in PILOT:
            folder = out / "lives" / f"s{seed}"
            folder.mkdir(parents=True)
            src, ref, oldbirth, oldfinal = tapes[seed]
            write(
                out / "status.json",
                dict(
                    stage="normal_stepwise", seed=seed, completed=ns, branches=nbranch
                ),
            )
            grid, weights, probs, born, comp = reference(
                batch, src, ref, oldbirth, oldfinal, folder
            )
            events = event_plan(src, probs, born, grid, frac)
            todo = branches(events)
            write(
                folder / "plan.json",
                dict(
                    seed=seed,
                    total_states=len(src),
                    reference_fraction=frac,
                    events=events,
                    forks={branch_key(k): v for k, v in todo.items()},
                ),
            )
            seq = list(todo)
            for first in range(0, len(seq), 6):
                group = seq[first : first + 6]
                write(
                    out / "status.json",
                    dict(
                        stage="matched_omissions",
                        seed=seed,
                        branches_done=first,
                        branches_total=len(seq),
                        completed=ns,
                        total_branches_done=nbranch,
                    ),
                )
                fork_group(batch, src, weights, grid, probs, group, folder / "forks")
                nbranch += len(group)
            ns += 1
    finally:
        batch.close()
    actor.verify_unchanged()
    write(
        out / "receipt.json",
        dict(
            mode=mode,
            seeds=PILOT,
            total_forks=nbranch,
            reference_fraction=frac,
            elapsed_seconds=time.monotonic() - start,
            panel_calls=batch.panel_calls,
            ordinary_panel_checks=batch.checks,
            max_panel_probability_error=batch.max_error,
            initial_fallbacks=batch.engine.fallback,
            fallback_at=batch.fallback_at,
            qwen_stats=batch.q.stats,
            frozen_backbone=True,
            device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_allocated(),
            versions={
                k: version(k) for k in ("torch", "numpy", "scipy", "transformers")
            },
        ),
    )
    write(out / "status.json", dict(stage="complete", seeds=ns, forks=nbranch))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["omitstable"], required=True)
    args = p.parse_args()
    run(args.root, args.mode)
