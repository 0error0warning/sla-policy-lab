"""Teacher-forced personal histories with paired, discarded one-update interventions."""

import gzip
import json
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import torch

from experiments.direct_policy.protocol import write
from experiments.first_fatal.design import BIRTHS, PILOT, REPLICATION
from experiments.first_fatal.study import bits, hashed
from experiments.health_baseline.design import Past
from experiments.individual_life.model import derivatives, update

# All six roles score the next living state before its outcome is read.
ROLES = (
    "global_before",
    "global",
    "local_before",
    "local",
    "local_from_global",
    "global_duplicate",
)


def write_events(path, rows):
    buf = "".join(
        json.dumps(e, allow_nan=False, separators=(",", ":")) + "\n" for e in rows
    )
    path.write_bytes(gzip.compress(buf.encode(), compresslevel=5, mtime=0))


def step_size(before, after):
    return float(
        np.linalg.norm(
            after.view("<f4").astype(float) - before.view("<f4").astype(float)
        )
    )


def replay(d, backend, ev, meta, whole):
    d.mkdir(parents=True, exist_ok=True)
    g, l, f = [backend.new(meta["birth"]) for _ in range(3)]
    backend.bind(g, l, f)
    bg, bl = g.snapshot(), l.snapshot()
    past = Past()
    logs = []
    witness = None
    try:
        for k, e in enumerate(ev):
            i, h, a = e["index"], e["health"], e["action"]
            weights = [
                bg,
                g.live(),
                bl,
                l.live(),
                f.live(),
                tuple(x.detach() for x in g.live()),
            ]
            focal = (
                k > 0
                and witness is None
                and ev[k - 1]["practice"]
                and ev[k - 1]["action"] >= 2
            )
            requests = (
                [(1, a, g), (3, a, l)] if (whole and k % 13 == 0) or focal else []
            )
            panel = backend.panel(
                i, h, weights, check=whole or focal or k == 0, requests=requests
            )
            prob = panel.detach().cpu().numpy().copy()
            np.testing.assert_array_equal(prob[1], prob[5])
            if focal:
                witness = {"turn": k + 1, "index": i, "health": h, "u": e["u"][0]}
                stash = {
                    name: bits(pair) for name, pair in zip(ROLES, weights, strict=True)
                }
                witness["hashes"] = {n: hashed(v) for n, v in stash.items()}
                np.savez_compressed(d / "successor.npz", **stash, probs=prob)
            gg, gl = derivatives(panel, [(1, a, g), (3, a, l)])
            bg, bl = g.snapshot(), l.snapshot()
            pg, pl = bits(bg), bits(bl)
            with torch.no_grad():
                for x, y in zip(f.live(), bg, strict=True):
                    x.copy_(y)
            gb, lb, n = past.baselines(h)
            assert abs(gb - e["baseline"]) < 1e-9
            denom = e["probs"][1][a]
            updates = []
            for model, grad, adv, pi, initial in (
                (g, gg, e["reward"] - gb, prob[1, a], pg),
                (l, gl, e["reward"] - lb, prob[3, a], pl),
                (f, gg, e["reward"] - lb, prob[1, a], pg),
            ):
                info = update(model, grad, adv, denom, float(pi))
                end = model.bits()
                updates.append(
                    dict(
                        **info,
                        actual_step_norm=step_size(initial, end),
                        after_hash=hashed(end),
                    )
                )
            logs.append(
                {
                    "turn": k + 1,
                    "index": i,
                    "health": h,
                    "action": a,
                    "u": e["u"][0],
                    "practice": e["practice"],
                    "after": e["after"],
                    "reward": e["reward"],
                    "global_baseline": gb,
                    "local_baseline": lb,
                    "eligible": n,
                    "probs": prob[:5].tolist(),
                    "before_hash": [hashed(pg), hashed(pl)],
                    "updates": updates,
                    "source_global_max_error": float(
                        np.max(
                            np.abs(prob[1].astype(float) - np.asarray(e["probs"][1]))
                        )
                    ),
                    "source_global_hash_match": updates[0]["after_hash"]
                    == e["after_hash"][0],
                }
            )
            past.observe(h, e["reward"])
        assert ev[-1]["after"] == 0 and witness is not None
        write_events(d / "replay.jsonl.gz", logs)
        record = {
            "seed": meta["seed"],
            "birth": meta["birth"],
            "turns": len(logs),
            "living_successors": len(logs) - 1,
            "source_rule": meta["rule"],
            "witness": witness,
            "full_live_checks": whole,
        }
        write(d / "life.json", record)
        return len(logs)
    finally:
        backend.close()


def execute(out, backend, root):
    n = total = 0
    with ZipFile(root / "tapes.zip") as z:
        provenance = json.loads(z.read("sources.json"))
        assert all(
            sha256(z.read(k)).hexdigest() == v for k, v in provenance["files"].items()
        )
        for seed in (*PILOT, *REPLICATION):
            for birth in BIRTHS:
                key = f"s{seed}/{birth}"
                meta = json.loads(z.read(key + "/life.json"))
                events = [
                    json.loads(l)
                    for l in gzip.decompress(
                        z.read(key + "/events.jsonl.gz")
                    ).splitlines()
                ]
                assert (meta["seed"], meta["birth"], meta["rule"]) == (
                    seed,
                    birth,
                    "true1",
                )
                total += replay(
                    out / "replays" / key,
                    backend,
                    events,
                    meta,
                    seed in (PILOT[0], REPLICATION[0]),
                )
                n += 1
                write(
                    out / "status.json",
                    {
                        "stage": "replaying_fixed_histories",
                        "completed": n,
                        "seed": seed,
                        "birth": birth,
                    },
                )
    return {
        "assigned_lives": n,
        "source_turns": total,
        "living_successors": total - n,
        "source_archives": provenance["parent_archives"],
    }
