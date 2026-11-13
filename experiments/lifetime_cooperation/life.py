"""Drive one autonomous body; all policy measurements are read-only forks."""

from hashlib import sha256
import json
import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.design import action_at
from .design import ITEMS, checkpoints, physical_step, rules
from .simple import LinearGroup


def hashes(weights):
    return {
        k: sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        for k, v in weights.items()
    }


def run_life(folder, learner, kind, learning, seed, world, tape, body):
    folder.mkdir(parents=True, exist_ok=False)
    replay = LinearGroup() if kind == "qwen" and learning else None
    h, cumulative, seen_low = 350.0, 0.0, False
    events = []
    metadata = []
    snapshots = {}
    grids = {}
    ids, uniforms, noise, offsets = tape
    for k, raw_i in enumerate(ids):
        i = int(raw_i)
        u, v = (float(x) for x in uniforms[k])
        row = ITEMS[i]
        bits = rules(seed, world, k)
        p = learner.decision(i, h)
        behavior = p[0]
        a = action_at(behavior, u)
        bad = (a + int(offsets[k])) % 4
        after, stages, maintained, rescued = physical_step(h, a, row, bits, v, noise[k])
        reward = body.rul(after) - body.rul(h)
        baseline = (-8.0 + cumulative) / (8 + k)
        signal = reward - baseline
        after_death = after == 0.0
        tags = checkpoints(k, h, not seen_low, after_death) if learning else []
        if tags:
            key = f"k{k}"
            # Probes do not consume randomness or change the actor or physical state.
            grids[key] = learner.probe()
            if replay:
                grids[key + "_on_tape"] = replay.probe()
            weights = learner.weights()
            if "birth" in tags or "last_living" in tags:
                for name, values in weights.items():
                    snapshots[key + "_" + name] = values.copy()
                if replay:
                    for name, values in replay.weights().items():
                        snapshots[key + "_on_tape_" + name] = values.copy()
            metadata.append(
                dict(
                    key=key,
                    completed=k,
                    health=h,
                    tags=tags,
                    bits=bits,
                    parameters=hashes(weights),
                )
            )
        evidence = dict(
            turn=k + 1,
            row=i,
            health=h,
            after=after,
            stages=stages,
            noise=noise[k].tolist(),
            action=a,
            wrong_action=bad,
            action_u=u,
            partner_u=v,
            maintenance=maintained,
            rescued_answer=rescued,
            rules=bits,
            prob=p.tolist(),
            reward=reward,
            baseline=baseline,
            signal=signal,
        )
        if learning:
            if replay:
                evidence["linear_on_tape_prob"] = replay.decision(i, h).tolist()
                evidence["linear_on_tape_steps"] = replay.learn(
                    i, h, a, bad, signal, behavior
                )
            evidence["steps"] = learner.learn(i, h, a, bad, signal, behavior)
            evidence["after_parameters"] = hashes(learner.weights())
        events.append(evidence)
        cumulative += reward
        seen_low |= h <= 10
        h = after
        if after_death:
            break
    assert h == 0 and abs(cumulative + body.rul(350)) < 1e-8
    if learning:
        np.savez_compressed(folder / "panels.npz", **grids)
        np.savez_compressed(folder / "parameters.npz", **snapshots)
    with (folder / "events.jsonl").open("w", encoding="utf8", newline="\n") as out:
        for event in events:
            out.write(
                json.dumps(
                    event, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    healthy = [e for e in events if e["health"] >= 90]

    def subsequent(records):
        return sum(
            e["after"] > 0 and e["after"] < e["health"] and e["turn"] < len(events)
            for e in records
        )

    summary = dict(
        seed=seed,
        world=world,
        kind=kind,
        learning=learning,
        turns=len(events),
        physical_death=True,
        injured_then_acted=subsequent(events),
        low_then_acted=subsequent([e for e in events if e["health"] <= 10]),
        low_requests=sum(e["health"] <= 10 for e in events),
        rescued=sum(e["rescued_answer"] for e in events),
        healthy_rounds=len(healthy),
        healthy_tasks=sum(e["action"] >= 2 and e["after"] > 0 for e in healthy),
        tasks=sum(e["action"] >= 2 and e["after"] > 0 for e in events),
        checkpoints=metadata,
    )
    write(folder / "life.json", summary)
    return summary
