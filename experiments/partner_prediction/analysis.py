"""Equal-life policy and forecast scores from frozen measurements and observed trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import settle, write
from experiments.individual_life.analysis import describe, records
from experiments.individual_life.design import offers
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT, rules

from .design import ARMS, MODES, SHADOWS

SPLITS = ("original", "held")
PARTNERS = (("both", None), ("P86", 2), ("P93", 3))


def evaluate_panel(seed, archive, phase):
    pol, pred = [], []
    bits = rules(seed, "stable", 0)
    scheme = "opposite" if bits[0] == bits[1] else "same"
    for w, split in enumerate(SPLITS):
        for label, person in PARTNERS:
            ix = np.array(
                [
                    j
                    for j, i in enumerate(PANEL[w])
                    if person is None or ITEMS[i]["partner"] == person
                ]
            )
            truth = np.stack([offers(ITEMS[i], bits)[0][1:] for i in PANEL[w, ix]])
            favored = truth[:, 1:].argmax(1)
            r = np.arange(len(ix))
            base = {
                "seed": seed,
                "rules": scheme,
                "phase": phase,
                "split": split,
                "partner": label,
            }
            for j, mode in enumerate(MODES):
                prob = archive["policy"][j, w][:, ix, :].astype(float)
                assert (
                    np.allclose(prob.sum(-1), 1, atol=3e-6, rtol=0) and prob.min() >= 0
                )
                a = prob[1, :, 2:]
                good, other = a[r, favored], a[r, 1 - favored]
                # Ratio of probability masses across a balanced panel, conditional on answering.
                conditional = good.sum() / (good.sum() + other.sum())
                pol.append(
                    dict(
                        **base,
                        mode=mode,
                        advantage=100 * (good - other).mean(),
                        conditional=100 * conditional,
                        favored=100 * good.mean(),
                        unfavored=100 * other.mean(),
                        cooperation=100 * (good * 0.9 + other * 0.1).mean(),
                        healthy=100 * prob[2:, :, 2:].sum(-1).mean(),
                        death=100 * (1 - 0.15 * prob[0, :, 1]).mean(),
                    )
                )
            for name in (*ARMS, *SHADOWS):
                forecast = archive[name][w, ix]
                order = forecast[r, 1 + favored] - forecast[r, 2 - favored]
                pred.append(
                    dict(
                        **base,
                        mode=name,
                        error=float(((forecast[:, 1:] - truth[:, 1:]) ** 2).mean()),
                        ask_error=float(((forecast[:, 0] - 0.15) ** 2).mean()),
                        rank=100 * ((order > 0).mean() + 0.5 * (order == 0).mean()),
                        ties=int((order == 0).sum()),
                        move_from_drop93=100
                        * abs(forecast[:, 1:] - archive["drop93"][w, ix, 1:]).mean(),
                        mean_favored=float(forecast[r, 1 + favored].mean()),
                        mean_unfavored=float(forecast[r, 2 - favored].mean()),
                    )
                )
    return pol, pred


def experience(seed, phase, arm, rows):
    bits = rules(seed, "stable", 0)
    summary = []
    for label, pid in PARTNERS:
        events = [e for e in rows if pid is None or ITEMS[e["row"]]["partner"] == pid]
        scored = [e for e in events if e["maintenance"] is not None]

        def brier(es):
            loss = [
                (
                    e["forecasts"][arm]["before"][0][e["action"] - 1]
                    - int(e["maintenance"])
                )
                ** 2
                for e in es
            ]
            return float(np.mean(loss)) if loss else None

        healthy = [e for e in events if e["health"] >= 90]
        direct = [e for e in scored if e["action"] >= 2]
        exposure = [e for e in events if e["after"] > 0 and e["after"] < e["health"]]
        recoverable = [
            e
            for e in events
            if all(
                settle(e["health"], a, True, e["noise"])[0] > 0
                and settle(e["health"], a, False, e["noise"])[0] == 0
                for a in (2, 3)
            )
        ]
        summary.append(
            {
                "seed": seed,
                "phase": phase,
                "arm": arm,
                "partner": label,
                "turns": len(events),
                "scorable": len(scored),
                "labeled_updates": sum(e["after"] > 0 for e in scored),
                "brier": brier(scored),
                "answer_brier": brier(direct),
                "maintained": sum(bool(e["maintenance"]) for e in scored),
                "healthy_rounds": len(healthy),
                "healthy_tasks": sum(
                    e["action"] >= 2 and e["after"] > 0 for e in healthy
                ),
                "injured_then_acted": len(exposure),
                "low_injured_then_acted": sum(e["health"] <= 10 for e in exposure),
                "recoverable": len(recoverable),
                "recoverable_answer": sum(e["action"] >= 2 for e in recoverable),
                "recoverable_survived_answer": sum(
                    e["action"] >= 2 and e["after"] > 0 for e in recoverable
                ),
                "total_tasks": sum(e["action"] >= 2 and e["after"] > 0 for e in events),
                "preferred_answer": sum(
                    e["action"] == offers(ITEMS[e["row"]], bits)[1] for e in direct
                ),
                "answers": len(direct),
            }
        )
    return summary


def immediate(seed, rows):
    bits = rules(seed, "stable", 0)
    bucket = {}
    for e in rows:
        a = e["action"]
        if not a or e["after"] == 0:
            continue
        row = ITEMS[e["row"]]
        source = row["partner"]
        label = "P93" if source == 3 else "P86"
        other = {**row, "partner": 5 - source}
        before, after = (
            np.array(e["forecasts"]["true"][k])[:, a - 1] for k in ("before", "after")
        )
        actual = np.array([offers(r, bits)[0][a] for r in (row, other)])
        improvement = (before - actual) ** 2 - (after - actual) ** 2
        change = 100 * (after - before)
        for acts in ("all", "answers" if a >= 2 else "ask"):
            bucket.setdefault((label, acts), []).append(
                [*change, *abs(change), *improvement]
            )
    fields = (
        "own_delta",
        "other_delta",
        "own_abs_pp",
        "other_abs_pp",
        "own_error_reduction",
        "other_error_reduction",
    )
    return [
        dict(
            seed=seed,
            source=who,
            expressions=acts,
            n=len(v),
            **dict(zip(fields, np.mean(v, axis=0), strict=True)),
        )
        for (who, acts), v in bucket.items()
    ]


def conclusions(policy, pred, life):
    def series(table, metric, mode, partner="both", phase="replay", split="held"):
        entries = {
            r["seed"]: r[metric]
            for r in table
            if r["phase"] == phase
            and r["partner"] == partner
            and r["split"] == split
            and r["mode"] == mode
        }
        return np.array([entries[s] for s in PILOT])

    def actual(arm):
        by = {
            r["seed"]: 100 * r["healthy_tasks"] / r["healthy_rounds"]
            for r in life
            if r["phase"] == "live" and r["partner"] == "both" and r["arm"] == arm
        }
        return np.array([by[s] for s in PILOT])

    criteria = {}

    def gate(name, delta, threshold, minimum=0):
        d = describe(delta)
        criteria[name] = dict(
            **d,
            paired=delta.tolist(),
            threshold=threshold,
            min_positive=minimum,
            passed=bool(d["mean"] >= threshold and d["positive"] >= minimum),
        )

    for comp in ("body", "wrong"):
        gate(
            "P93_error_gain_vs_" + comp,
            series(pred, "error", comp, "P93") - series(pred, "error", "true", "P93"),
            0.02,
        )
    gate(
        "P93_error_gain_from_P93",
        series(pred, "error", "drop93", "P93") - series(pred, "error", "true", "P93"),
        0.02,
        6,
    )
    gate(
        "P86_error_gain_from_P93",
        series(pred, "error", "drop93", "P86") - series(pred, "error", "true", "P86"),
        -0.01,
    )
    for comp in ("body", "wrong"):
        for metric, low in (("advantage", 3), ("cooperation", 0), ("healthy", -2)):
            gate(
                metric + "_vs_" + comp,
                series(policy, metric, "true") - series(policy, metric, comp),
                low,
            )
        gate("actual_healthy_vs_" + comp, actual("true") - actual(comp), -2)
    gate(
        "advantage_vs_flip",
        series(policy, "advantage", "true") - series(policy, "advantage", "flip"),
        1,
        6,
    )
    return {
        "gates": criteria,
        "prediction_pass": all(
            v["passed"] for k, v in criteria.items() if k.startswith("P")
        ),
        "policy_pass": all(
            v["passed"] for k, v in criteria.items() if not k.startswith("P")
        ),
    }


def analyze(root, out):
    out.mkdir(parents=True, exist_ok=True)
    policy, pred, life, writes = [], [], [], []
    exposures = []
    for seed in PILOT:
        for phase in ("replay", "live"):
            d = root / "lives" / f"s{seed}" / phase
            with np.load(d / "panel.npz") as z:
                p, q = evaluate_panel(seed, z, phase)
                policy.extend(p)
                pred.extend(q)
            for arm in ARMS:
                rows = records(
                    d / (("replay" if phase == "replay" else arm) + ".jsonl")
                )
                life.extend(experience(seed, phase, arm, rows))
                if phase == "replay" and arm == "true":
                    writes.extend(immediate(seed, rows))
                    for partner in (2, 3):
                        for frame in range(2):
                            for action in (1, 2, 3):
                                seen = [
                                    e
                                    for e in rows
                                    if e["after"] > 0
                                    and e["action"] == action
                                    and ITEMS[e["row"]]["partner"] == partner
                                    and ITEMS[e["row"]]["frame"] == frame
                                ]
                                exposures.append(
                                    {
                                        "seed": seed,
                                        "partner": ("P86", "P93")[partner - 2],
                                        "frame": frame,
                                        "action": action,
                                        "observed": len(seen),
                                        "accepted": sum(
                                            bool(e["maintenance"]) for e in seen
                                        ),
                                    }
                                )
    for name, rows in (
        ("policy", policy),
        ("prediction", pred),
        ("life", life),
        ("immediate", writes),
        ("source_exposures", exposures),
    ):
        csv_out(out / (name + ".csv"), rows)
    result = conclusions(policy, pred, life)
    write(out / "gates.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.root, a.out), indent=2))
