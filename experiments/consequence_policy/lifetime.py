"""Run discrete mortal lives and test what they leave in Qwen's output parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from experiments.consequence_policy.experiment import read_inputs
from experiments.consequence_policy.policy import Policy, deployed, features, softmax
from experiments.consequence_policy.study import (
    CONFIRM,
    EXPLORATION,
    LIVE_CAP,
    PARTNERS,
    PILOT,
    TEMP,
    TEST_H,
    basis,
)
from experiments.direct_policy.protocol import probabilities, rng, settle, write
from experiments.social_closed_loop.core import BodyValue

WORLDS = ((0, 0), (0, 1), (1, 0), (1, 1))
ARMS = ("prior", "personal", "frozen", "population", "shuffled_population", "scratch")
COLUMNS = (
    "context",
    "health",
    "action",
    "response",
    "after",
    "reward",
    "prob0",
    "prob1",
    "prob2",
    "prob3",
    "u17_before",
    "u42_before",
    "score",
    "fisher",
    "normalized_reward",
    "u17_after",
    "u42_after",
)


def environment_tape(seed, world, replica, rows):
    gen = rng(seed, "new-continuous-life-v1", list(world), replica)
    keys = {
        (r["partner"], r["frame"], r["layout"]): i
        for i, r in enumerate(rows)
        if r["split"] == "test" and r["task"] == 4
    }
    ids = []
    pairs = [(p, f) for p in range(2) for f in range(2)] * 2
    layouts = np.concatenate(
        [gen.permutation(24) for _ in range((LIVE_CAP + 23) // 24)]
    )
    for block in range((LIVE_CAP + 7) // 8):
        for offset, index in enumerate(gen.permutation(8)):
            p, f = pairs[index]
            ids.append(keys[p, f, layouts[block * 8 + offset]])
    u = gen.random((LIVE_CAP, 2))
    noise = 0.1 * ndtri(
        ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * gen.random((LIVE_CAP, 5))
    )
    return np.asarray(ids, dtype=int), u, noise


def combined(a, b):
    policy = Policy(a.shape[2])
    policy.weights[:] = (a + b) / 2.0
    direction = Policy(a.shape[2])
    direction.weights[:] = (a - b) / 2.0
    return policy, direction


class PersonalActor:
    def __init__(self, checkpoint_dir, seed, world, arm):
        def load(mapping, name):
            return Policy.load(
                checkpoint_dir / f"s{seed}-m{mapping}" / (name + ".npz")
            ).weights

        self.arm = arm
        a, b = load(0, "real"), load(1, "real")
        self.main, self.direction = combined(a, b)
        self.u = np.zeros(2)
        if arm in ("frozen", "scratch"):
            self.main.weights.fill(0.0)
        elif arm in ("population", "shuffled_population"):
            source = "real" if arm == "population" else "shuffle"
            for p in range(2):
                self.main.weights[p] = load(world[p], source)[p]
        elif arm not in ("prior", "personal"):
            raise ValueError(arm)
        if arm != "personal":
            self.direction.weights.fill(0.0)

    def score(self, i, row, health, native, x):
        p = row["partner"]
        residual = self.main.residual(x[i], p, health)
        d = self.direction.residual(x[i], p, health)
        logits = native[i] / TEMP + residual + self.u[p] * d
        return deployed(logits), logits, d

    def update(self, p, a, health, x, pi, logits, direction, normalized):
        s = softmax(logits)
        z = (1 - EXPLORATION) * s / pi
        score = z * (direction - np.dot(s, direction))
        fisher = float(pi @ (score**2))
        gradient = float(score[a])
        if self.arm == "personal":
            self.u[p] = advance(self.u[p], normalized, gradient, fisher)
        elif self.arm == "scratch":
            derivative = -s.copy()
            derivative[a] += 1.0
            derivative *= z[a]
            g = basis([health])[0]
            norm = max(1.0, 0.25 * np.dot(g, g) * np.dot(x, x))
            self.main.weights[p] += (
                0.7
                * (normalized + 0.9)
                * np.einsum("b,d,a->bda", g, x, derivative)
                / norm
            )
        return gradient, fisher


def advance(u, outcome, score, fisher):
    return float(np.clip(u + 0.5 * (outcome + 0.9) * score / (1.0 + fisher), -1.0, 1.0))


def post(actor, native, x, rows, world, u=None):
    idx = np.asarray(
        [i for i, r in enumerate(rows) if r["split"] == "test" and r["task"] == 5]
    )
    p = np.array([rows[i]["partner"] for i in idx])
    coeff = actor.u if u is None else np.asarray(u)
    w = Policy(x.shape[1])
    w.weights[:] = (
        actor.main.weights + coeff[:, None, None, None] * actor.direction.weights
    )
    pi = deployed(w.grid_logits(native[idx], x[idx], p, TEST_H))
    favored = [
        2
        if probabilities(
            PARTNERS[rows[i]["partner"]], rows[i]["frame"], world[rows[i]["partner"]]
        )[1]
        > 0.5
        else 3
        for i in idx
    ]
    answer = pi[:, :, 2:].sum(axis=-1)
    correct = pi[:, np.arange(len(idx)), favored]
    return {
        "healthy_answer": float(answer[-2:].mean()),
        "danger_answer": float(answer[0].mean()),
        "healthy_style": float(correct[-2:].mean()),
        "partner_style": [float(correct[-2:, p == j].mean()) for j in (0, 1)],
    }


def replay(trace, rows, seed, world, replica):
    rewards = trace[:, 14].copy()
    partners = np.array([rows[int(i)]["partner"] for i in trace[:, 0]])
    bands = np.digitize(trace[:, 1], [6.0, 15.0])
    gen = rng(seed, "personal-tape-shuffle", list(world), replica)
    perm = np.arange(len(trace))
    for p in range(2):
        for band in np.unique(bands):
            idx = np.flatnonzero((partners == p) & (bands == band))
            perm[idx] = gen.permutation(idx)
    coefficients = []
    for sequence in (rewards, rewards[perm]):
        values = np.zeros(2)
        for row, p, r in zip(trace, partners, sequence):
            values[p] = advance(values[p], r, row[12], row[13])
        coefficients.append(values)
    assert np.allclose(coefficients[0], trace[-1, -2:], rtol=0.0, atol=1e-10)
    return coefficients[1], perm


def run_life(actor, seed, world, replica, tape, rows, native, x, body, override=None):
    ids, u, noise = tape
    h, records = 350.0, []
    for t, i in enumerate(ids):
        row = rows[i]
        p = row["partner"]
        pi, logits, direction = (
            actor.score(i, row, h, native, x)
            if override is None
            else override(i, row, h, actor.u)
        )
        a = int((u[t, 0] > np.cumsum(pi)[:3]).sum())
        private = probabilities(PARTNERS[p], row["frame"], world[p])
        accepted = bool(u[t, 1] < private[a - 1]) if a else None
        after, _ = settle(h, a, accepted, noise[t])
        observed = body.rul(after) - body.rul(h)
        normalized = observed / min(1.0, body.rul(h))
        before = actor.u.copy()
        grad, fisher = actor.update(p, a, h, x[i], pi, logits, direction, normalized)
        if override is not None:
            actual_u, grad, fisher = override.commit(
                p, a, normalized, actor.u, grad, fisher
            )
            actor.u[:] = actual_u
        records.append(
            [
                i,
                h,
                a,
                -1 if accepted is None else int(accepted),
                after,
                observed,
                *pi,
                *before,
                grad,
                fisher,
                normalized,
                *actor.u,
            ]
        )
        h = after
        if h <= 0.0:
            break
    trace = np.asarray(records, dtype=np.float64)
    assert len(trace) == LIVE_CAP or trace[-1, 4] == 0.0
    assert np.array_equal(trace[1:, 1], trace[:-1, 4])
    actions = trace[:, 2].astype(int)
    near = trace[:, 1] <= 3.0
    metrics = {
        "turns": len(trace),
        "tasks": int((actions >= 2).sum()),
        "maintenances": int((trace[:, 3] == 1).sum()),
        "capped": bool(h > 0.0),
        "danger_encounters": int(near.sum()),
        "dangerous_answers": int(((actions >= 2) & near).sum()),
        "final_u": actor.u.tolist(),
        "post": post(actor, native, x, rows, world),
    }
    if actor.arm == "personal":
        shuffled_u, permutation = replay(trace, rows, seed, world, replica)
        metrics.update(
            shuffle_u=shuffled_u.tolist(),
            shuffle_permutation=permutation.tolist(),
            shuffle_post=post(actor, native, x, rows, world, u=shuffled_u),
        )
        maps = [1 if m == 0 else -1 for m in world]
        metrics["correct_signs"] = (actor.u * maps > 0).tolist()
    return trace, metrics


def lifetime_summary(output, seeds):
    clusters = []
    absolute = {a: [] for a in ARMS}
    for seed in seeds:
        runs = json.loads((output / f"life-{seed}.json").read_bytes())
        differences = []
        for run in runs:
            arms = run["arms"]
            for a in ARMS:
                absolute[a].append(arms[a])
            personal = arms["personal"]
            differences.append(
                {
                    "style_vs_prior": personal["post"]["healthy_style"]
                    - arms["prior"]["post"]["healthy_style"],
                    "style_vs_shuffle": personal["post"]["healthy_style"]
                    - personal["shuffle_post"]["healthy_style"],
                    "life_vs_prior": personal["turns"] - arms["prior"]["turns"],
                    "tasks_vs_prior": personal["tasks"] - arms["prior"]["tasks"],
                    "tasks_vs_frozen": personal["tasks"] - arms["frozen"]["tasks"],
                    "life_vs_frozen": personal["turns"] - arms["frozen"]["turns"],
                    "danger_vs_prior": personal["post"]["danger_answer"]
                    - arms["prior"]["post"]["danger_answer"],
                    "correct_partner_sign": np.mean(personal["correct_signs"]),
                    "both_partner_signs": float(all(personal["correct_signs"])),
                }
            )
        clusters.append(
            {k: float(np.mean([d[k] for d in differences])) for k in differences[0]}
        )
    indices = rng(63996, "lifetime-bootstrap").integers(
        len(seeds), size=(12000, len(seeds))
    )
    stats = {}
    for k in clusters[0]:
        data = np.array([r[k] for r in clusters])
        ci = np.quantile(data[indices].mean(axis=1), [0.025, 0.975])
        stats[k] = {
            "mean": float(data.mean()),
            "ci95": ci.tolist(),
            "positive": int((data > 0).sum()),
        }
    summary = {}
    for arm, obs in absolute.items():
        summary[arm] = {
            k: float(np.mean([v[k] for v in obs]))
            for k in ("turns", "tasks", "maintenances", "capped")
        }
        summary[arm].update(
            {
                k: float(np.mean([v["post"][k] for v in obs]))
                for k in ("danger_answer", "healthy_answer", "healthy_style")
            }
        )
        for key in ("danger_encounters", "dangerous_answers"):
            summary[arm][key] = int(sum(v[key] for v in obs))
    gates = [stats[k] for k in ("style_vs_prior", "style_vs_shuffle")]
    passed = all(
        d["mean"] >= 0.10
        and d["ci95"][0] > 0
        and d["positive"] >= (16 if len(seeds) == 20 else 3)
        for d in gates
    )
    return {
        "seeds": list(seeds),
        "clusters": clusters,
        "stats": stats,
        "absolute": summary,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave", choices=["pilot", "confirm"], required=True)
    parser.add_argument("--fresh-prefixes", type=Path)
    parser.add_argument("--life-output", type=Path)
    args = parser.parse_args()
    rows, native, x, _, _ = read_inputs(args.cache, args.output)
    destination = args.life_output or args.output
    if args.fresh_prefixes:
        if destination.resolve() == args.output.resolve():
            raise ValueError(
                "Provide a separate output path for the singleton deployment."
            )
        with (
            np.load(args.fresh_prefixes) as fresh,
            np.load(args.output / "projection.npz") as projection,
        ):
            index = fresh["context"]
            assert np.array_equal(
                index, [i for i, r in enumerate(rows) if r["split"] == "test"]
            )
            native[index] = fresh["logits"]
            x[index] = features(
                fresh["hidden"], projection["center"], projection["matrix"]
            )
    destination.mkdir(parents=True, exist_ok=True)
    seeds = PILOT if args.wave == "pilot" else CONFIRM
    body = BodyValue()
    for seed in seeds:
        results, trajectories, checkpoints = [], {}, {}
        for world in WORLDS:
            for replica in range(2):
                tape = environment_tape(seed, world, replica, rows)
                arms = {}
                key = f"w{world[0]}{world[1]}-r{replica}"
                for arm in ARMS:
                    actor = PersonalActor(args.output, seed, world, arm)
                    trace, metrics = run_life(
                        actor, seed, world, replica, tape, rows, native, x, body
                    )
                    trajectories[f"{key}-{arm}"] = trace
                    arms[arm] = metrics
                    if arm == "scratch":
                        checkpoints[key] = actor.main.weights
                results.append({"world": list(world), "replica": replica, "arms": arms})
        np.savez_compressed(destination / f"life-{seed}-traces.npz", **trajectories)
        np.savez_compressed(destination / f"life-{seed}-scratch.npz", **checkpoints)
        write(destination / f"life-{seed}.json", results)
        print("complete life seed", seed, flush=True)
    report = lifetime_summary(destination, seeds)
    write(destination / f"life-{args.wave}.json", report)
    print(json.dumps(report["stats"]), flush=True)


if __name__ == "__main__":
    main()
