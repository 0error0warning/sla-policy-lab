"""Restoration, autonomous CPU lives, and own-gradient prefix sensitivity replay."""

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from experiments.consequence_policy.experiment import evaluation_reference, read_inputs
from experiments.consequence_policy.lifetime import (
    WORLDS,
    PersonalActor,
    environment_tape,
    run_life,
)
from experiments.consequence_policy.policy import deployed
from experiments.consequence_policy.study import LIVE_CAP, PARTNERS, TEST_H
from experiments.consequence_warmstart.design import (
    ARMS,
    BIRTH_SEEDS,
    BLOCKS,
    CHECKPOINTS,
    ENV_SEEDS,
    TIME_LIMIT,
    permutation,
    selected,
    tape,
    verify_lock,
)
from experiments.direct_policy.protocol import probabilities, settle, write
from experiments.social_closed_loop.core import BodyValue


def effective(actor):
    return actor.main.weights + actor.u[:, None, None, None] * actor.direction.weights


def check_time(started):
    if time.monotonic() - started > TIME_LIMIT:
        raise TimeoutError("Locked CPU wall budget exhausted; no replacements")


def restore(source, rows, native, x, body, started):
    result = {
        "lives": 0,
        "turns": 0,
        "max_probability_error": 0.0,
        "max_scalar_feedback_error": 0.0,
        "max_scratch_weight_error": 0.0,
    }
    out = source / "results"
    for seed in BIRTH_SEEDS:
        with (
            np.load(out / f"life-{seed}-traces.npz") as saved,
            np.load(out / f"life-{seed}-scratch.npz") as weights,
        ):
            for world in WORLDS:
                for rep in range(2):
                    stream = environment_tape(seed, world, rep, rows)
                    key = f"w{world[0]}{world[1]}-r{rep}"
                    for arm in ("prior", "personal", "scratch"):
                        check_time(started)
                        actor = PersonalActor(out, seed, world, arm)
                        trace, _ = run_life(
                            actor, seed, world, rep, stream, rows, native, x, body
                        )
                        old = saved[f"{key}-{arm}"]
                        np.testing.assert_array_equal(trace[:, :6], old[:, :6])
                        pe = float(np.max(abs(trace[:, 6:10] - old[:, 6:10])))
                        se = float(np.max(abs(trace[:, 10:] - old[:, 10:])))
                        if pe > 2e-6 or se > 1e-8:
                            raise ValueError(f"Restoration failed: {seed}/{key}/{arm}")
                        result["max_probability_error"] = max(
                            result["max_probability_error"], pe
                        )
                        result["max_scalar_feedback_error"] = max(
                            result["max_scalar_feedback_error"], se
                        )
                        if arm == "scratch":
                            error = float(
                                np.max(abs(actor.main.weights - weights[key]))
                            )
                            if error > 2e-6:
                                raise ValueError("Scratch restoration weights differ")
                            result["max_scratch_weight_error"] = max(
                                result["max_scratch_weight_error"], error
                            )
                        result["lives"] += 1
                        result["turns"] += len(trace)
        print(f"restored {seed}", flush=True)
    result["passed"] = True
    return result


class Probes:
    def __init__(self, rows, native, x, body):
        self.rows, self.native, self.x = rows, native, x
        self.indices = {
            block: {
                part: selected(rows, tw) for part, tw in zip(("live", "heldout"), pairs)
            }
            for block, pairs in BLOCKS.items()
        }
        self.references = {}
        for block, parts in self.indices.items():
            for part, idx in parts.items():
                contexts = [rows[i] for i in idx]
                for mapping in (0, 1):
                    self.references[block, part, mapping] = evaluation_reference(
                        body, contexts, mapping
                    )

    def measure(self, actor, block, world):
        panels, arrays = {}, {}
        for part, idx in self.indices[block].items():
            p = np.array([self.rows[i]["partner"] for i in idx])
            model = actor.main.__class__(self.x.shape[1])
            model.weights[:] = effective(actor)
            pi = deployed(model.grid_logits(self.native[idx], self.x[idx], p, TEST_H))
            arrays[part] = pi
            risk = np.empty_like(pi)
            favored = np.empty(len(idx), dtype=int)
            for partner in (0, 1):
                mask = p == partner
                r, _, f = self.references[block, part, world[partner]]
                risk[:, mask] = r[:, mask]
                favored[mask] = f[mask]
            good = pi[:, np.arange(len(idx)), favored]
            answer = pi[:, :, 2:].sum(-1)
            death = (pi * risk).sum(-1)
            panels[part] = {
                "health": list(TEST_H),
                "answer": answer.mean(1).tolist(),
                "style": good.mean(1).tolist(),
                "conditional_style": (good.sum(1) / answer.sum(1)).tolist(),
                "death": death.mean(1).tolist(),
                "partner_style": [good[:, p == j].mean(1).tolist() for j in (0, 1)],
                "partner_conditional_style": [
                    (good[:, p == j].sum(1) / answer[:, p == j].sum(1)).tolist()
                    for j in (0, 1)
                ],
            }
        return panels, arrays


def snapshot(actor, count, label, block, world, probes, arrays):
    panels, values = probes.measure(actor, block, world)
    arrays[f"{label}-weights"] = effective(actor).copy()
    for name, value in values.items():
        arrays[f"{label}-{name}"] = value
    return {"updates": count, "panels": panels, "terminal": label == "terminal"}


def life(actor, stream, rows, native, x, body, block, world, probes, started):
    ids, uniforms, noises = stream
    arrays, checkpoints, records = {}, {}, []
    checkpoints["birth"] = snapshot(actor, 0, "birth", block, world, probes, arrays)
    health = 350.0
    for turn, i in enumerate(ids):
        check_time(started)
        row = rows[i]
        p = row["partner"]
        pi, logits, direction = actor.score(i, row, health, native, x)
        action = int((uniforms[turn, 0] > np.cumsum(pi)[:3]).sum())
        private = probabilities(PARTNERS[p], row["frame"], world[p])
        maintained = bool(uniforms[turn, 1] < private[action - 1]) if action else None
        after, _ = settle(health, action, maintained, noises[turn])
        observed = body.rul(after) - body.rul(health)
        normalized = observed / min(1.0, body.rul(health))
        if after == 0.0 or turn + 1 == LIVE_CAP:
            checkpoints["last_preterminal"] = snapshot(
                actor, turn, "last_preterminal", block, world, probes, arrays
            )
        before = effective(actor).copy()
        actor.update(p, action, health, x[i], pi, logits, direction, normalized)
        delta = effective(actor) - before
        if not np.isfinite(delta).all():
            raise FloatingPointError("Nonfinite update")
        if np.any(delta[1 - p]):
            raise ValueError("Unexpected cross-partner weight update")
        next_pi, _, _ = actor.score(i, row, health, native, x)
        records.append(
            [
                i,
                health,
                action,
                -1 if maintained is None else int(maintained),
                after,
                observed,
                *pi,
                normalized,
                float(np.linalg.norm(delta)),
                *next_pi,
            ]
        )
        health = after
        count = turn + 1
        if health > 0 and count in CHECKPOINTS:
            key = f"after_{count}"
            checkpoints[key] = snapshot(actor, count, key, block, world, probes, arrays)
        if health <= 0:
            break
    trace = np.array(records, dtype=np.float64)
    np.testing.assert_array_equal(trace[1:, 1], trace[:-1, 4])
    if not np.all(trace[:, 4] <= trace[:, 1]):
        raise ValueError("Body healed")
    checkpoints["terminal"] = snapshot(
        actor, len(trace), "terminal", block, world, probes, arrays
    )
    actions = trace[:, 2].astype(int)
    metrics = {
        "turns": len(trace),
        "tasks": int((actions >= 2).sum()),
        "maintenances": int((trace[:, 3] == 1).sum()),
        "capped": bool(health > 0),
        "injury_then_another_choice": int((trace[:-1, 4] < trace[:-1, 1]).sum()),
        "nonterminal_updates": len(trace) - int(health <= 0),
        "maintenance_requests": int((actions != 0).sum()),
        "refusals": int((trace[:, 3] == 0).sum()),
        "checkpoints": checkpoints,
    }
    arrays.update(trace=trace, context=ids, uniforms=uniforms, noises=noises)
    return metrics, arrays


def replay(actor, trace, order, rows, native, x):
    """Recompute recipient scores; never use donor log-gradients."""
    rewards = trace[order, 10]
    for entry, outcome in zip(trace, rewards):
        i, health, action = int(entry[0]), entry[1], int(entry[2])
        row = rows[i]
        pi, logits, direction = actor.score(i, row, health, native, x)
        actor.update(
            row["partner"], action, health, x[i], pi, logits, direction, outcome
        )
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    verify_lock(repo, args.source, args.lock)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        rows, native, x, _, _ = read_inputs(
            args.source / "capture/output", args.source / "results"
        )
        body = BodyValue()
        restored = restore(args.source, rows, native, x, body, started)
        write(args.out / "restoration.json", restored)
        probes = Probes(rows, native, x, body)
        all_lives = []
        for birth, seed in zip(BIRTH_SEEDS, ENV_SEEDS):
            for block in BLOCKS:
                for world in WORLDS:
                    stream = tape(seed, world, block, rows)
                    case = {
                        "birth_seed": birth,
                        "environment_seed": seed,
                        "block": block,
                        "world": list(world),
                        "arms": {},
                    }
                    key = f"s{seed}-{block}-w{world[0]}{world[1]}"
                    for arm in ARMS:
                        actor = PersonalActor(
                            args.source / "results", birth, world, arm
                        )
                        metrics, arrays = life(
                            actor,
                            stream,
                            rows,
                            native,
                            x,
                            body,
                            block,
                            world,
                            probes,
                            started,
                        )
                        if arm == "warmstart_adapter":
                            metrics["replays"] = {}
                            trace = arrays["trace"]
                            for label, checkpoint in metrics["checkpoints"].items():
                                check_time(started)
                                count = checkpoint["updates"]
                                prefix = trace[:count]
                                perm, detail = permutation(
                                    trace, rows, seed, world, block, count
                                )
                                detail["permutation"] = perm.tolist()
                                metrics["replays"][label] = {"shuffle": detail}
                                for name, target, order in (
                                    ("true", "warmstart_adapter", np.arange(count)),
                                    ("shuffle", "warmstart_adapter", perm),
                                    ("scratch", "scratch", np.arange(count)),
                                ):
                                    recipient = PersonalActor(
                                        args.source / "results", birth, world, target
                                    )
                                    replay(recipient, prefix, order, rows, native, x)
                                    if name == "true":
                                        np.testing.assert_array_equal(
                                            effective(recipient),
                                            arrays[f"{label}-weights"],
                                        )
                                    panels, values = probes.measure(
                                        recipient, block, world
                                    )
                                    metrics["replays"][label][name + "_panels"] = panels
                                    arrays[f"{label}-replay-{name}-weights"] = (
                                        effective(recipient).copy()
                                    )
                                    for part, value in values.items():
                                        arrays[f"{label}-replay-{name}-{part}"] = value
                        np.savez_compressed(args.out / f"{key}-{arm}.npz", **arrays)
                        case["arms"][arm] = metrics
                    write(args.out / f"{key}.json", case)
                    all_lives.append(case)
            print(f"complete development cluster {seed}", flush=True)
        write(args.out / "index.json", all_lives)
        write(
            args.out / "execution.json",
            {
                "complete": True,
                "seconds": time.monotonic() - started,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "source_mode": "archived batched frozen Qwen prefixes, CPU float32 adapters",
                "gpu_allocations": 0,
                "autonomous_lives": len(all_lives) * len(ARMS),
                "encounters": sum(
                    a["turns"] for c in all_lives for a in c["arms"].values()
                ),
            },
        )
    except Exception as exc:
        write(
            args.out / "execution.json",
            {
                "complete": False,
                "seconds": time.monotonic() - started,
                "error": repr(exc),
                "stop": "no_expansion",
            },
        )
        raise


if __name__ == "__main__":
    main()
