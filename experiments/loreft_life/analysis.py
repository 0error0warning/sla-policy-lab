"""Score frozen, matched-replay, partner-replacement and causal intervention contrasts."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import describe
from .design import ITEMS, PANEL, PILOT, WORLDS, ROLES, INTERVENTIONS, orientation

SOCIAL = (1, 2)  # H 6,90
BODY = ("cooperation", "death", "healthy")


def arm_stats(p, seed, world):
    result = {}
    for w, split in enumerate(("source", "held")):
        ctx = [ITEMS[i] for i in PANEL[w]]
        signs = np.array(
            [
                orientation(seed, r)
                * (-1 if world == "changed" and r["partner"] == 2 else 1)
                for r in ctx
            ]
        )
        h = p[w].astype(float)
        margin = (h[:, :, 2] - h[:, :, 3]) * signs
        q_appeal = np.where(signs == 1, 0.9, 0.1)
        vals = {
            "social": margin[list(SOCIAL)].mean(),
            "cooperation": (h[1, :, 2] * q_appeal + h[1, :, 3] * (1 - q_appeal)).mean(),
            "death": (h[0, :, 0] + 0.85 * h[0, :, 1] + h[0, :, 2:].sum(-1)).mean(),
            "healthy": h[2:, :, 2:].sum(-1).mean(),
        }
        for name, v in vals.items():
            result[split + "_" + name] = 100 * float(v)
    return result


def history_contrast(stable, changed, seed, word=1, task=None):
    rows = [ITEMS[i] for i in PANEL[word]]
    m = lambda x: (x[word, :, :, 2].astype(float) - x[word, :, :, 3].astype(float))[
        list(SOCIAL)
    ]
    difference = 0.5 * (m(stable) - m(changed))
    selected = np.array([task is None or r["task"] == task for r in rows])
    target = selected & np.array([r["partner"] == 2 for r in rows])
    other = selected & ~target
    signs = np.array([orientation(seed, r) for r in rows])
    return 100 * float((difference[:, target] * signs[target]).mean()), 100 * float(
        abs(difference[:, other]).mean()
    )


def describe_values(x):
    return describe(np.asarray(x, dtype=float))


def summarize(per_seed):
    return {
        key: describe_values([row[key] for row in per_seed])
        for key in per_seed[0]
        if key != "seed"
    }


def behavioral_gate(seed_rows):
    n = len(seed_rows)
    s = summarize(seed_rows)
    ok = {}
    for ref in ("birth", "loreft_wrong"):
        for word in ("source", "held"):
            v = s[f"{word}_social_vs_{ref}"]
            ok[f"{word}_social_{ref}"] = (
                bool(v["mean"] >= 3 and v["positive"] >= 3 * n // 4)
                if word == "held"
                else bool(v["mean"] > 0)
            )
            ok[f"{word}_healthy_{ref}"] = bool(
                s[f"{word}_healthy_vs_{ref}"]["mean"] >= -2
            )
    ok["partner_adaptation"] = bool(s["adapt_S"]["mean"] >= 2)
    ok["specificity_over_spill"] = bool(s["adapt_S_minus_L"]["mean"] >= 1)
    return {
        "passed": all(ok.values()),
        "checks": ok,
        "effects_pp": s,
        "seeds": [r["seed"] for r in seed_rows],
    }


def csv_out(path, data):
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)


def analyze_replay(root, out):
    out.mkdir(parents=True, exist_ok=True)
    behavioral = []
    methods = []
    causal = []
    for seed in PILOT:
        grids = []
        for world in WORLDS:
            folder = root / "histories" / f"s{seed}" / world
            grids.append(np.load(folder / "final-panel.npy", allow_pickle=False))
        per = {"seed": seed}
        for role, name in enumerate(ROLES):
            stats = arm_stats(grids[0][role], seed, "stable")
            s, l = history_contrast(grids[0][role], grids[1][role], seed)
            methods.append(
                dict(
                    seed=seed, method=name, partner_adaptation=s, other_spill=l, **stats
                )
            )
        for ref in ("birth", "loreft_wrong", "native_lora", "residual_lora"):
            l = arm_stats(grids[0][ROLES.index("loreft")], seed, "stable")
            compare = arm_stats(grids[0][ROLES.index(ref)], seed, "stable")
            per.update({f"{k}_vs_{ref}": l[k] - compare[k] for k in l})
        s, l = history_contrast(grids[0][3], grids[1][3], seed)
        per.update(adapt_S=s, other_spill_L=l, adapt_S_minus_L=s - l)
        behavioral.append(per)
        # World, manipulation, wording, H, text, semantic action
        v = np.load(
            root / "steering" / f"s{seed}" / "probabilities.npy", allow_pickle=False
        ).astype(float)
        assert v.shape == (2, len(INTERVENTIONS), 2, 4, 32, 4)
        rec = {"seed": seed}
        selected = [
            j
            for j, i in enumerate(PANEL[1])
            if ITEMS[i]["partner"] == 2 and ITEMS[i]["task"] == 5
        ]

        def margin(control):
            z = v[:, INTERVENTIONS.index(control), 1]
            return (z[:, :, :, 2] - z[:, :, :, 3])[:, list(SOCIAL)][
                :, :, selected
            ].mean()

        external = 50 * (margin("plus") - margin("minus"))
        random_half = [
            abs(50 * (margin(x + "_plus") - margin(x + "_minus"))) for x in ("r1", "r2")
        ]
        s0 = history_contrast(v[0, 0], v[1, 0], seed, task=5)[0]

        def hs(control):
            j = INTERVENTIONS.index(control)
            return history_contrast(v[0, j], v[1, j], seed, task=5)[0]

        remove = s0 - hs("remove")
        fake = np.mean([abs(s0 - hs(x + "_remove")) for x in ("r1", "r2")])
        rec.update(
            positive_minus_negative_half=external,
            random_half=float(np.mean(random_half)),
            external_over_random=external - np.mean(random_half),
            native_S=s0,
            remove_drop=remove,
            random_drop_abs=float(fake),
            remove_over_random=remove - fake,
            swap_drop=s0 - hs("swap"),
            restore_probability_error=float(abs(v[:, 0] - v[:, 5]).max()),
        )
        causal.append(rec)
    g = behavioral_gate(behavioral)
    c = summarize(causal)
    cext = (
        c["positive_minus_negative_half"]["mean"] >= 1
        and c["external_over_random"]["mean"] > 0
    )
    cnat = (
        c["native_S"]["mean"] >= 2
        and c["remove_drop"]["mean"] >= 1
        and c["remove_drop"]["positive"] >= 6
        and c["remove_over_random"]["mean"] >= 0.5
        and c["swap_drop"]["mean"] > 0
        and max(v["restore_probability_error"] for v in causal) <= 1e-6
    )
    result = {
        "behavior": g,
        "steering": {
            "external_directional_control": bool(cext),
            "natural_history_mediation": bool(cnat),
            "effects": c,
            "primary_task": 5,
            "primary_wording": 4,
        },
    }
    csv_out(out / "behavior-by-seed.csv", behavioral)
    csv_out(out / "steering-by-seed.csv", causal)
    csv_out(out / "methods.csv", methods)
    write(out / "decisions.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    g = analyze_replay(a.root, a.out)
    print(
        json.dumps(
            {
                "behavior": g["behavior"]["passed"],
                "external": g["steering"]["external_directional_control"],
                "mediation": g["steering"]["natural_history_mediation"],
            }
        )
    )
