"""Follow an actual private policy from birth through its first lethal decision and death."""

import gzip
import json
import os
from hashlib import sha256
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import render, rng, write
from experiments.first_fatal.design import (
    BIRTHS,
    HEALTHY,
    RULES,
    first_fatal,
    practice,
    settle,
    world,
)
from experiments.individual_life.design import action_at, offers
from experiments.individual_life.model import derivatives, update


def arrivals(seed, rows):
    from scipy.special import ndtr, ndtri

    lookup = {
        (r["partner"] - 2, r["frame"], r["layout"]): j
        for j, r in enumerate(rows)
        if (r["task"], r["wording"]) == (4, 3)
    }
    block = 0
    while True:
        g = rng(seed, "first-fatal-arrivals-v1", block)
        layouts = g.permutation(24)
        cells = np.concatenate(
            [g.permutation(np.tile(np.arange(4), 2)) for _ in range(3)]
        )
        u = g.random((24, 2))
        noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * g.random((24, 5)))
        offsets = rng(seed, "first-fatal-wrong-label-v1", block).integers(1, 4, 24)
        for j, cell in enumerate(cells):
            yield (
                int(lookup[cell // 2, cell % 2, layouts[j]]),
                u[j],
                noise[j],
                int(offsets[j]),
            )
        block += 1


def hashed(a):
    return sha256(a.tobytes()).hexdigest()


def bits(pair):
    return (
        np.concatenate([t.detach().cpu().numpy().ravel() for t in pair])
        .astype("<f4", copy=False)
        .view("<u4")
        .copy()
    )


def probe(backend, weights, context, h, seed, turn):
    """Read-only. All roles see identical text, body, order and upstream tensors."""
    ix = backend.probe_ix
    grid = []
    same = []
    with torch.no_grad():
        for level_index, level in enumerate((h, *HEALTHY)):
            same.append(
                backend.panel(context, level, weights, check=True)
                .detach()
                .cpu()
                .numpy()
            )
            witness = (seed + turn + 37 * level_index) % len(ix)
            grid.append(
                np.stack(
                    [
                        backend.panel(int(i), level, weights, check=j == witness)
                        .detach()
                        .cpu()
                        .numpy()
                        for j, i in enumerate(ix)
                    ]
                )
            )
    return {
        "health": np.array([h, *HEALTHY]),
        "same_request": np.stack(same),
        "panel": np.stack(grid),
        "indices": np.asarray(ix),
    }


def run_life(d, backend, body, seed, birth, rule, whole):
    d.mkdir(parents=True, exist_ok=True)
    main = backend.new(birth)
    wrong = backend.new(birth) if rule == "true1" else None
    near = backend.new(birth) if rule == "true1" else None
    backend.bind(main, wrong, near)
    born = main.snapshot()
    entry = born
    store = {"birth": main.bits()}
    birth_hash = hashed(store["birth"])
    current_hash = [birth_hash] * (3 if wrong is not None else 1)
    h = 350.0
    cum = 0.0
    step = 0
    entry_t = None
    endpoint = None
    probes = None
    events = []
    try:
        for t, (i, u, noise, offset) in enumerate(arrivals(seed, backend.rows), 1):
            row = backend.rows[i]
            is_near = practice(h)
            if is_near and entry_t is None:
                entry = main.snapshot()
                store["entry"] = main.bits()
                entry_t = t
                if near is not None:
                    np.testing.assert_array_equal(near.bits(), store["entry"])
            first = first_fatal(h) and endpoint is None
            weights = [
                born,
                main.live(),
                wrong.live() if wrong is not None else born,
                near.live() if near is not None else born,
                entry,
                tuple(p.detach() for p in main.live()),
            ]
            if first:
                endpoint = t
                store["first"] = main.bits()
                if wrong is not None:
                    store["first_wrong"] = wrong.bits()
                    store["first_near_wrong"] = near.bits()
                probes = probe(backend, weights, i, h, seed, t)
            with torch.set_grad_enabled(rule != "frozen"):
                witness = (
                    [(1, (seed + t) % 4, main)]
                    if rule != "frozen" and (first or t % 13 == 0 and whole)
                    else []
                )
                panel = backend.panel(
                    i, h, weights, check=whole or first or t == 1, requests=witness
                )
            prob = panel.detach().cpu().numpy().copy()
            np.testing.assert_array_equal(prob[1], prob[5])
            if first:
                np.testing.assert_array_equal(prob, probes["same_request"][0])
            a = action_at(prob[1], u[0])
            response_p, _ = offers(row, world(seed))
            maintained = bool(u[1] < response_p[a]) if a else None
            after, path, over = settle(h, a, maintained, noise)
            reward = body.rul(after) - body.rul(h)
            baseline = (-8.0 + cum) / (7 + t)
            signal = reward - baseline
            credited = []
            before = current_hash.copy()
            if rule != "frozen":
                mis = (a + offset) % 4
                roles = [(1, a if rule == "true1" else mis, main)]
                if wrong is not None:
                    roles.extend([(2, mis, wrong), (3, mis if is_near else a, near)])
                grads = derivatives(panel, roles)
                for k, ((j, label, policy), grad) in enumerate(
                    zip(roles, grads, strict=True)
                ):
                    info = update(
                        policy,
                        grad,
                        signal,
                        float(prob[1, label]),
                        float(prob[j, label]),
                    )
                    current_hash[k] = hashed(policy.bits())
                    credited.append({"credited_action": label, **info})
                step += 1
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
                    "peak_collapse": over,
                    "practice": is_near,
                    "first_fatal": first,
                    "probs": prob[:5].tolist(),
                    "reward": reward,
                    "baseline": baseline,
                    "signal": signal,
                    "version_at_choice": step - int(rule != "frozen"),
                    "update": credited,
                    "before_hash": before,
                    "after_hash": current_hash.copy(),
                }
            )
            cum += reward
            h = after
            del panel
            if h <= 0:
                break
        store["final"] = main.bits()
        info = {
            "seed": seed,
            "birth": birth,
            "rule": rule,
            "world": world(seed),
            "turns": len(events),
            "practice_entry": entry_t,
            "first_fatal": endpoint,
            "practice_count": sum(e["practice"] for e in events),
            "natural_death": h == 0,
            "main_updates": step,
            "full_live_checks": whole,
            "snapshot_hashes": {k: hashed(v) for k, v in store.items()},
        }
        np.savez_compressed(d / "parameters.npz", **store)
        if probes is not None:
            np.savez_compressed(d / "predecision.npz", **probes)
        tape = b"".join(
            (
                json.dumps(
                    e,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf8")
            for e in events
        )
        (d / "events.jsonl.gz").write_bytes(
            gzip.compress(tape, compresslevel=5, mtime=0)
        )
        write(d / "complete.tmp", info)
        os.replace(d / "complete.tmp", d / "life.json")
        return info
    finally:
        backend.close()


def execute(out, backend, body, seeds):
    n = 0
    encounters = 0
    for seed in seeds:
        for birth in BIRTHS:
            for rule in RULES:
                d = out / "lives" / f"s{seed}" / birth / rule
                v = run_life(d, backend, body, seed, birth, rule, seed == seeds[0])
                n += 1
                encounters += v["turns"]
                write(
                    out / "status.json",
                    {
                        "stage": "finite_lives",
                        "completed": n,
                        "seed": seed,
                        "birth": birth,
                        "rule": rule,
                    },
                )
        backup = out.parent / "seed-backups"
        backup.mkdir(exist_ok=True)
        bucket = out / "lives" / f"s{seed}"
        members = {}
        partial = backup / f"s{seed}.tmp"
        with ZipFile(partial, "w", ZIP_DEFLATED) as z:
            for f in sorted(bucket.glob("*/*/*")):
                name = f.relative_to(bucket).as_posix()
                data = f.read_bytes()
                members[name] = sha256(data).hexdigest()
                z.writestr(name, data)
            z.writestr("members.json", json.dumps(members, sort_keys=True))
        os.replace(partial, backup / f"s{seed}.zip")
        print(f"completed all private lives for seed {seed}", flush=True)
    return {"assigned_lives": n, "encounters": encounters, "seeds": list(seeds)}
