"""Keep both parameter banks and replace only the routing key."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import PARTNERS
from experiments.direct_policy.protocol import probabilities, rng, write
from experiments.internal_body.credit import CONFIRM, Learner
from experiments.internal_body.evidence import stat
from experiments.internal_body.lives import WORLDS, life_tape, post, run_one
from experiments.internal_body.personal import LifeInputs
from experiments.social_closed_loop.core import BodyValue


def identity_or_layout(data, folder, seed, kind, partition):
    a = Learner(data, folder, seed, "predict")
    if kind == "layout":
        a.routing_slot = np.array([partition[r["layout"]] for r in data.rows])
    if kind == "relabel":
        a.routing_slot = 1 - data.slot
    return a


def main():
    p = argparse.ArgumentParser()
    for k in ("cache", "credit", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    data = LifeInputs(a.cache)
    body = BodyValue()
    clusters = []
    for seed in CONFIRM:
        env_seed = seed + 1000
        partition = rng(seed, "capacity-route-layout").permutation([0] * 12 + [1] * 12)
        obs = []
        weights = {}
        raw = {}
        with (
            np.load(a.credit / f"credit-{seed}-raw.npz") as trajectories,
            np.load(a.credit / f"credit-{seed}-parameters.npz") as old,
        ):
            for j, world in enumerate(WORLDS):
                src = trajectories[f"w{j}-v0-predict"]
                fixed = {}
                for kind in ("id", "layout", "relabel"):
                    actor = identity_or_layout(data, a.credit, seed, kind, partition)
                    for r in src:
                        i, h, act = int(r[0]), float(r[1]), int(r[2])
                        pi = actor.policy(i, h)
                        if kind != "layout":
                            np.testing.assert_array_equal(pi[0][0], r[7:11])
                        actor.update(i, act, float(r[6]), pi)
                    if kind == "relabel":
                        np.testing.assert_array_equal(
                            actor.W[::-1], old[f"w{j}-v0-predict"]
                        )
                    if kind == "id":
                        np.testing.assert_array_equal(actor.W, old[f"w{j}-v0-predict"])
                    if kind == "layout":
                        weights[f"w{j}-fixed-layout"] = actor.W
                    fixed[kind] = post(actor, world)
                env = life_tape(env_seed, world, data.rows)
                raw[f"w{j}-context"], raw[f"w{j}-uniform"], raw[f"w{j}-noise"] = env
                alive = {}
                for kind in ("id", "layout"):
                    actor = identity_or_layout(data, a.credit, seed, kind, partition)
                    tr, r = run_one(actor, env_seed, world, 0, env, body)
                    weights[f"w{j}-live-{kind}"] = actor.W
                    raw[f"w{j}-{kind}"] = tr
                    alive[kind] = r
                same = (
                    probabilities(PARTNERS[0], 0, world[0])[1]
                    == probabilities(PARTNERS[1], 0, world[1])[1]
                )
                obs.append(
                    {
                        "world": list(world),
                        "same_preferences": same,
                        "fixed_tape": fixed,
                        "on_policy": alive,
                    }
                )
        row = {}
        for same in (False, True):
            group = [r for r in obs if r["same_preferences"] == same]
            key = "same" if same else "opposite"
            for k in ("style", "danger", "healthy"):
                row[f"{key}_{k}_fixed_tape"] = float(
                    np.mean(
                        [
                            v["fixed_tape"]["id"][k] - v["fixed_tape"]["layout"][k]
                            for v in group
                        ]
                    )
                )
                row[f"{key}_{k}_live"] = float(
                    np.mean(
                        [
                            v["on_policy"]["id"]["post"][k]
                            - v["on_policy"]["layout"]["post"][k]
                            for v in group
                        ]
                    )
                )
            for k in ("tasks", "turns"):
                row[f"{key}_{k}_live"] = float(
                    np.mean(
                        [
                            v["on_policy"]["id"][k] - v["on_policy"]["layout"][k]
                            for v in group
                        ]
                    )
                )
        clusters.append(row)
        np.savez_compressed(a.output / f"routing-{seed}-raw.npz", **raw)
        np.savez_compressed(a.output / f"routing-{seed}-parameters.npz", **weights)
        write(
            a.output / f"routing-{seed}.json",
            {
                "population_seed": seed,
                "environment_seed": env_seed,
                "partition": partition.tolist(),
                "worlds": obs,
            },
        )
    result = {k: stat([r[k] for r in clusters]) for k in clusters[0]}
    write(
        a.output / "confirm.json",
        {
            "seeds": list(CONFIRM),
            "clusters": clusters,
            "stats": result,
            "consistent_relabel_bitwise": True,
        },
    )
    print(json.dumps({k: v for k, v in result.items() if "style" in k}))


if __name__ == "__main__":
    main()
