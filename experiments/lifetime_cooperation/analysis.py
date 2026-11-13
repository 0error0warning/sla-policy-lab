"""Recompute seed-level gates from exported action probabilities and actual exposures."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import describe
from experiments.individual_life.design import offers
from .design import ITEMS, PANEL, ROLES, WAVES, death_matrix

METRICS = ("cooperation", "death", "healthy")
SPLITS = ("familiar", "held")
SIGNS = {"cooperation": 1.0, "death": -1.0, "healthy": 1.0}


def scores(grid, bits, partner=None):
    assert grid.shape == (3, 2, 4, 32, 4) and np.allclose(
        grid.sum(axis=-1), 1.0, atol=3e-6, rtol=0.0
    )
    assert np.isfinite(grid).all() and grid.min() >= 0
    reference = death_matrix()
    for w, name in enumerate(SPLITS):
        ids = PANEL[w]
        keep = np.array(
            [partner is None or ITEMS[i]["partner"] == partner for i in ids]
        )
        offers_grid = np.stack([offers(ITEMS[i], bits)[0] for i in ids])[keep]
        # Each read-only panel includes every partner/frame/task and four cyclic layouts equally.
        for j, role in enumerate(ROLES):
            probs = grid[j, w][:, keep, :].astype(float)
            death = (1.0 - offers_grid) * reference[0, :, 0] + offers_grid * reference[
                0, :, 1
            ]
            yield dict(
                split=name,
                role=role,
                cooperation=float(
                    (probs[1, :, 2:] * offers_grid[:, 2:]).sum(axis=1).mean()
                ),
                death=float((probs[0] * death).sum(axis=1).mean()),
                healthy=float(probs[2:, :, 2:].sum(axis=-1).mean()),
            )


def csv_out(path, records):
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)


def collect(root):
    lives = []
    panels = []
    for file in sorted((root / "lives").glob("s*/*/*/life.json")):
        life = json.loads(file.read_bytes())
        meta = life.pop("checkpoints")
        lives.append(life)
        if not life["learning"]:
            continue
        with np.load(file.parent / "panels.npz") as z:
            for point in meta:
                for fam, key in [(life["kind"], point["key"])] + (
                    [("simple_on_qwen", point["key"] + "_on_tape")]
                    if life["kind"] == "qwen"
                    else []
                ):
                    for partner in (None, 2, 3):
                        for values in scores(z[key], point["bits"], partner):
                            for tag in point["tags"]:
                                panels.append(
                                    dict(
                                        seed=life["seed"],
                                        world=life["world"],
                                        family=fam,
                                        checkpoint=tag,
                                        completed=point["completed"],
                                        actual_health=point["health"],
                                        partner="all"
                                        if partner is None
                                        else ("P86", "P93")[partner - 2],
                                        **values,
                                    )
                                )
    return lives, panels


def gate(lives, panels, seeds, family):
    roles = (family, "simple_on_qwen") if family == "qwen" else (family,)
    chosen = [
        r
        for r in panels
        if r["world"] == "stable"
        and r["checkpoint"] == "last_living"
        and r["partner"] == "all"
        and r["family"] in roles
    ]

    def deltas(learner, split, metric, control):
        table = {
            (r["seed"], r["role"]): r[metric]
            for r in chosen
            if r["split"] == split and r["family"] == learner
        }
        return np.array(
            [100 * (table[s, "current"] - table[s, control]) for s in seeds]
        )

    checks = {}
    summaries = {}
    consistency = 3 * len(seeds) // 4
    for split in SPLITS:
        for key in METRICS:
            for ref in ("birth", "wrong"):
                v = deltas(family, split, key, ref)
                info = describe(v)
                summaries[f"{split}.{key}.{ref}"] = info
                good = SIGNS[key] * v
                if key == "healthy":
                    passed = good.mean() >= -2.0
                elif split == "familiar":
                    passed = (
                        good.mean() >= (5.0 if key == "cooperation" else 2.0)
                        and (good > 0).sum() >= consistency
                    )
                else:
                    passed = good.mean() > 0
                checks[f"{split}.{key}.{ref}"] = bool(passed)
    free = {
        (v["seed"], v["learning"]): v
        for v in lives
        if v["world"] == "stable" and v["kind"] == family
    }

    def healthy(s, training):
        v = free[s, training]
        return v["healthy_tasks"] / v["healthy_rounds"]

    rates = [100 * (healthy(s, True) - healthy(s, False)) for s in seeds]
    checks["actual_healthy_vs_frozen"] = mean(rates) >= -2.0
    summaries["actual_healthy_vs_frozen"] = describe(rates)
    low = sum(free[s, True]["low_then_acted"] > 0 for s in seeds)
    language = {}
    if family == "qwen":
        for ref in ("birth", "wrong"):
            for key in METRICS:
                gap = SIGNS[key] * (
                    deltas(family, "held", key, ref)
                    - deltas("simple_on_qwen", "held", key, ref)
                )
                language[f"{key}.{ref}"] = describe(gap)
    lang_success = (
        None
        if not language
        else bool(
            all(
                language[f"{k}.{r}"]["mean"] >= 2.0
                for k in ("death", "cooperation")
                for r in ("birth", "wrong")
            )
            and all(checks[f"held.healthy.{r}"] for r in ("birth", "wrong"))
        )
    )
    return dict(
        seeds=list(seeds),
        family=family,
        passed=all(checks.values()),
        checks=checks,
        paired_pp=summaries,
        low_successor_seeds=low,
        low_exposure_sparse=low < len(seeds) / 2,
        language_gap_benefit_pp=language,
        language_gate=lang_success,
    )


def analyze(root, output):
    receipt = json.loads((root / "receipt.json").read_bytes())
    seeds = WAVES[receipt["mode"]]
    lives, panels = collect(root)
    assert len(lives) == len(seeds) * 8 and sorted({v["seed"] for v in lives}) == list(
        seeds
    )
    output.mkdir(parents=True, exist_ok=True)
    csv_out(output / "lives.csv", lives)
    csv_out(output / "panels.csv", panels)
    primary = gate(lives, panels, seeds, "qwen")
    write(output / "gate.json", primary)
    write(output / "simple-gate.json", gate(lives, panels, seeds, "simple"))
    return primary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.root, a.out)
    print(
        json.dumps(
            {k: result[k] for k in ("seeds", "passed", "language_gate", "checks")},
            indent=2,
        )
    )
