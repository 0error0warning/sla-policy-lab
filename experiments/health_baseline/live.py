"""Free, non-resetting lives with two within-tape counterfactual adapters on true-credit arms."""

import json
import os
from hashlib import sha256
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import render, write
from experiments.first_fatal.design import BIRTHS, first_fatal, practice, settle, world
from experiments.first_fatal.study import arrivals, hashed, probe
from experiments.health_baseline.design import ARMS, BASELINES, Past
from experiments.health_baseline.replay import write_events
from experiments.individual_life.design import action_at, offers
from experiments.individual_life.model import derivatives, update


def run_life(
    d, backend, body, seed, birth, arm, whole, rules=BASELINES, extra_rule=None
):
    d.mkdir(parents=True, exist_ok=True)
    primary, secondary = rules
    name, credit = (primary.name, "frozen") if arm == "frozen" else arm.split("_")
    if credit not in ("frozen", "true", "wrong") or name not in {
        primary.name,
        secondary.name,
    }:
        raise ValueError(arm)
    policy_rule, other_rule = (
        (primary, secondary) if name == primary.name else (secondary, primary)
    )
    main = backend.new(birth)
    wrong = backend.new(birth) if credit == "true" else None
    other = backend.new(birth) if credit == "true" else None
    extra = backend.new(birth) if extra_rule is not None and credit == "true" else None
    backend.bind(main, wrong, other)
    born = main.snapshot()
    entry = born
    store = {"birth": main.bits()}
    hashes = [hashed(store["birth"])] * (
        1 + 2 * int(wrong is not None) + int(extra is not None)
    )
    entry_t = endpoint = None
    events = []
    past = Past()
    h = 350.0
    try:
        for t, (i, u, noise, offset) in enumerate(arrivals(seed, backend.rows), 1):
            criterion = policy_rule.at_health(h)
            other_criterion = other_rule.at_health(h)
            row = backend.rows[i]
            near = practice(h)
            if near and entry_t is None:
                entry = main.snapshot()
                store["entry"] = main.bits()
                entry_t = t
            weights = [
                born,
                main.live(),
                wrong.live() if wrong is not None else born,
                other.live() if other is not None else born,
                extra.live() if extra is not None else entry,
                tuple(v.detach() for v in main.live()),
            ]
            first = first_fatal(h) and endpoint is None
            if first:
                endpoint = t
                store["first"] = main.bits()
                if wrong is not None:
                    store["first_wrong"] = wrong.bits()
                    store["first_other"] = other.bits()
                if extra is not None:
                    store["first_extra"] = extra.bits()
                focal = probe(backend, weights, i, h, seed, t)
                np.savez_compressed(d / "predecision.npz", **focal)
            with torch.set_grad_enabled(credit != "frozen"):
                requested = (
                    [(1, (seed + t) % 4, main)]
                    if credit != "frozen" and (first or (whole and t % 13 == 0))
                    else []
                )
                if requested and extra is not None:
                    requested.append((4, (seed + t) % 4, extra))
                panel = backend.panel(
                    i, h, weights, check=whole or first or t == 1, requests=requested
                )
            prob = panel.detach().cpu().numpy().copy()
            np.testing.assert_array_equal(prob[1], prob[5])
            if first:
                np.testing.assert_array_equal(prob, focal["same_request"][0])
            a = action_at(prob[1], u[0])
            chances, _ = offers(row, world(seed))
            maintained = bool(u[1] < chances[a]) if a else None
            gb, lb, n = past.baselines(h)
            after, path, peak = settle(h, a, maintained, noise)
            reward = body.rul(after) - body.rul(h)
            baseline = criterion.baseline(gb, lb)
            credited = []
            before = hashes.copy()
            if credit != "frozen":
                mis = (a + offset) % 4
                roles = [(1, a if credit == "true" else mis, main)]
                modes = [criterion]
                if wrong is not None:
                    roles.extend([(2, mis, wrong), (3, a, other)])
                    modes.extend([criterion, other_criterion])
                if extra is not None:
                    roles.append((4, a, extra))
                    modes.append(extra_rule.at_health(h))
                for j, ((k, label, model), grad, center) in enumerate(
                    zip(roles, derivatives(panel, roles), modes, strict=True)
                ):
                    signal = center.signal(reward, gb, lb)
                    info = update(
                        model,
                        grad,
                        signal,
                        float(prob[1, label]),
                        float(prob[k, label]),
                    )
                    hashes[j] = hashed(model.bits())
                    credited.append(
                        dict(
                            credited_action=label,
                            criterion=center.name,
                            signal=signal,
                            **info,
                        )
                    )
                    if rules != BASELINES:
                        credited[-1].update(
                            weighted_feedback=center.damage_weight * reward,
                            damage_weight=center.damage_weight,
                        )
            events.append(
                {
                    "turn": t,
                    "index": i,
                    "partner": row["partner"],
                    "frame": row["frame"],
                    "layout": row["layout"],
                    "action": a,
                    "word": WORDS[a],
                    "utterance": render(a, row["task"], 0),
                    "health": h,
                    "after": after,
                    "subevents": path,
                    "noise": noise.tolist(),
                    "u": u.tolist(),
                    "offset": offset,
                    "maintenance": maintained,
                    "peak_collapse": peak,
                    "practice": near,
                    "first_fatal": first,
                    "probs": prob[:5].tolist(),
                    "reward": reward,
                    "baseline": baseline,
                    "global_baseline": gb,
                    "local_baseline": lb,
                    "eligible": n,
                    "signal": criterion.signal(reward, gb, lb),
                    "version_at_choice": t - 1 if credit != "frozen" else 0,
                    "update": credited,
                    "before_hash": before,
                    "after_hash": hashes.copy(),
                }
            )
            if rules != BASELINES:
                events[-1]["weighted_feedback"] = criterion.damage_weight * reward
            past.observe(h, reward)
            h = after
            if h <= 0:
                break
        store["final"] = main.bits()
        np.savez_compressed(d / "parameters.npz", **store)
        write_events(d / "events.jsonl.gz", events)
        info = {
            "seed": seed,
            "birth": birth,
            "rule": arm,
            "world": world(seed),
            "turns": len(events),
            "practice_entry": entry_t,
            "first_fatal": endpoint,
            "practice_count": sum(e["practice"] for e in events),
            "natural_death": h == 0,
            "main_updates": len(events) if credit != "frozen" else 0,
            "full_live_checks": whole,
            "snapshot_hashes": {k: hashed(v) for k, v in store.items()},
        }
        write(d / "life.json", info)
        return len(events)
    finally:
        backend.close()


def execute(out, backend, body, seeds, arms=ARMS, rules=BASELINES, arm_overrides=None):
    n = turns = 0
    backup = out.parent / "seed-backups"
    backup.mkdir(exist_ok=True)
    for seed in seeds:
        bucket = out / "lives" / f"s{seed}"
        for birth in BIRTHS:
            for arm in arms:
                selected, extra = (arm_overrides or {}).get(arm, (rules, None))
                turns += run_life(
                    bucket / birth / arm,
                    backend,
                    body,
                    seed,
                    birth,
                    arm,
                    seed == seeds[0],
                    selected,
                    extra,
                )
                n += 1
                write(
                    out / "status.json",
                    {
                        "stage": "living",
                        "completed": n,
                        "seed": seed,
                        "birth": birth,
                        "arm": arm,
                    },
                )
        members = {}
        temp = backup / f"s{seed}.tmp"
        with ZipFile(temp, "w", ZIP_DEFLATED) as z:
            for f in sorted(bucket.glob("*/*/*")):
                name = f.relative_to(bucket).as_posix()
                data = f.read_bytes()
                members[name] = sha256(data).hexdigest()
                z.writestr(name, data)
            z.writestr("members.json", json.dumps(members, sort_keys=True))
        os.replace(temp, backup / f"s{seed}.zip")
    return {"assigned_lives": n, "encounters": turns, "seeds": list(seeds)}
