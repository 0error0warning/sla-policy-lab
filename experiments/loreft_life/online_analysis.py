"""Independent scoring and physical verification for the conditional fresh-life wave."""

import argparse
from pathlib import Path
import json
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.individual_life.design import action_at
from experiments.individual_life.analysis import describe
from experiments.lifetime_cooperation.audit import physical
from experiments.lifetime_cooperation.design import schedule
from experiments.social_closed_loop.core import BodyValue
from .analysis import arm_stats, csv_out
from .design import ITEMS, ROLES, LIVE_SEEDS


def verify(root, dest):
    receipts = json.loads((root / "receipt.json").read_bytes())
    assert receipts["mode"] == "samealive" and receipts["frozen_backbone"]
    rows = []
    seeddiffs = []
    body = BodyValue()
    for seed in LIVE_SEEDS:
        source = {}
        ix, u, noise, offset = schedule(seed)
        for arm in ("birth", "native_lora", "loreft"):
            slot = ROLES.index(arm)
            d = root / "lives" / f"s{seed}" / arm
            m = json.loads((d / "life.json").read_bytes())
            es = list(map(json.loads, (d / "events.jsonl").read_bytes().splitlines()))
            assert es[-1]["after"] == 0 and len(es) == m["turns"]
            samples = 0
            tasks = 0
            low = 0
            inj = 0
            h = 350.0
            total = 0.0
            for k, e in enumerate(es):
                assert (
                    e["health"] == h
                    and e["action_u"] == u[k, 0]
                    and e["partner_u"] == u[k, 1]
                    and ix[k] == e["row"]
                )
                i = e["row"]
                r = ITEMS[i]
                p = np.array(e["prob"])
                a = action_at(p[slot], e["action_u"])
                assert a == e["action"] and (a + offset[k]) % 4 == e["wrong_action"]
                b = divmod((seed - LIVE_SEEDS[0]) % 4, 2)
                j = r["partner"] - 2
                fav = 2 + (j ^ r["frame"] ^ b[j])
                prob = 0.15 if a == 1 else 0.9 if a == fav else 0.1
                accepted = bool(u[k, 1] < prob) if a else None
                assert e["maintenance"] == accepted
                np.testing.assert_allclose(e["noise"], noise[k], rtol=0, atol=5e-15)
                path = physical(h, a, accepted, e["noise"])
                np.testing.assert_allclose(e["stages"], path, rtol=0, atol=1e-10)
                after = e["after"]
                assert abs(path[-1] - after) < 1e-10
                assert (after > 0) == (k < len(es) - 1)
                reward = body.rul(after) - body.rul(h)
                assert abs(e["signal"] - reward + (-8 + total) / (8 + k)) < 1e-9
                samples += int(h >= 90)
                tasks += int(h >= 90 and a >= 2 and after > 0)
                low += int(h <= 10 and after > 0)
                inj += int(after > 0 and after < h)
                total += reward
                h = after
            assert (samples, tasks, low, inj) == (
                m["healthy_rounds"],
                m["healthy_tasks"],
                m["low_then_again"],
                m["injured_then_again"],
            )
            with np.load(d / "panels.npz") as z:
                g = z["last"]
                saved = {r: arm_stats(g[ROLES.index(r)], seed, "stable") for r in ROLES}
            own = saved[arm]
            source[arm] = {"healthy_actual": 100 * tasks / samples, **own}
            rows.append(
                {
                    "seed": seed,
                    "agent": arm,
                    "turns": len(es),
                    "low_injury_then_next": low,
                    "all_injury_then_next": inj,
                    "healthy_completed": tasks,
                    "healthy_requests": samples,
                    **source[arm],
                }
            )
            if arm == "loreft":
                dif = {"seed": seed}
                for ref in ("birth", "loreft_wrong"):
                    dif.update({f"{k}_vs_{ref}": own[k] - saved[ref][k] for k in own})
        dif["actual_healthy_vs_frozen"] = (
            source["loreft"]["healthy_actual"] - source["birth"]["healthy_actual"]
        )
        seeddiffs.append(dif)
    dest.mkdir(parents=True, exist_ok=True)
    csv_out(dest / "autonomous.csv", rows)
    csv_out(dest / "paired.csv", seeddiffs)
    write(
        dest / "summary.json",
        {k: describe([d[k] for d in seeddiffs]) for k in seeddiffs[0] if k != "seed"},
    )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print("verified fresh lives", len(verify(a.root, a.out)))
