"""Paired estimates at seed level. The replay's logged next decision uses the original draw."""

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import discord, records, resamples
from experiments.first_fatal.design import BIRTHS
from experiments.health_baseline.design import ARMS, PILOT, REPLICATION
from experiments.individual_life.design import action_at

WAVES = (PILOT, REPLICATION)
PAIRS = {
    "single_global": (0, 1),
    "single_local_common": (0, 4),
    "single_local_own": (2, 3),
    "single_replace_global_by_local": (1, 4),
    "cumulative_replace_global_by_local": (1, 3),
}


def save_csv(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def ci(ids, values, waves=WAVES, adjusted_alpha=0.00625):
    ordered = sorted(zip(ids, values, strict=True))
    bins = [[v for seed, v in ordered if seed in wave] for wave in waves]
    assert len(set(ids)) == len(ids) == sum(map(len, bins))
    a = np.asarray(bins[0] + bins[1], float)
    if len(a) == 0:
        return None
    means = a[resamples(tuple(map(len, bins)))].mean(axis=1)
    qs = np.quantile(means, [adjusted_alpha, 0.025, 0.975, 1 - adjusted_alpha])
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "range": [float(a.min()), float(a.max())],
        "pilot": float(np.mean(bins[0])) if bins[0] else None,
        "replication": float(np.mean(bins[1])) if bins[1] else None,
        "lower_adjusted": float(qs[0]),
        "ci95": [float(qs[1]), float(qs[2])],
        "upper_adjusted": float(qs[3]),
    }


def read_replay(d):
    with gzip.open(d / "replay.jsonl.gz", "rt", encoding="utf8") as f:
        return [json.loads(s) for s in f]


def differences(p, q, u):
    # The first three CDF thresholds implement the actual semantic sampler. The
    # fourth interval absorbs the tiny roundoff in exported FP32 probabilities.
    edges = [
        np.r_[0.0, np.clip(np.asarray(x, dtype=float).cumsum()[:3], 0, 1), 1.0]
        for x in (p, q)
    ]
    l, r = edges
    overlap = np.maximum(
        0.0, np.minimum(l[1:], r[1:]) - np.maximum(l[:-1], r[:-1])
    ).sum()
    a, b = action_at(p, u), action_at(q, u)
    return (
        int(a != b),
        int((a >= 2) != (b >= 2)),
        float(np.abs(p - q).sum() / 2),
        max(0.0, float(1 - overlap)),
        float(q[2:].sum() - p[2:].sum()),
        int(np.array_equal(p, q)),
    )


def replay_summary(root, dest):
    from experiments.first_fatal.design import PILOT as OLD_PILOT
    from experiments.first_fatal.design import REPLICATION as OLD_REP

    focus = []
    raw = []
    diag = []
    for f in sorted(root.glob("s*/*/life.json")):
        m = json.loads(f.read_bytes())
        ev = read_replay(f.parent)
        seed, birth = m["seed"], m["birth"]
        for e in ev:
            diag.append((e["source_global_max_error"], e["source_global_hash_match"]))
        for prev, nxt in itertools.pairwise(ev):
            assert (
                prev["after"] == nxt["health"] > 0 and prev["turn"] + 1 == nxt["turn"]
            )
            p = np.asarray(nxt["probs"], dtype=float)

            def near(e):
                return e["practice"] and e["action"] >= 2

            row = {
                "seed": seed,
                "birth": birth,
                "turn": prev["turn"],
                "next_turn": nxt["turn"],
                "next_health": nxt["health"],
                "next_u": nxt["u"],
                "next_index": nxt["index"],
                "global_step": prev["updates"][0]["actual_step_norm"],
                "local_own_step": prev["updates"][1]["actual_step_norm"],
                "local_common_step": prev["updates"][2]["actual_step_norm"],
                "global_signal": prev["reward"] - prev["global_baseline"],
                "local_signal": prev["reward"] - prev["local_baseline"],
                "eligible": prev["eligible"],
                "practice": prev["practice"],
                "practice_answer": near(prev),
            }
            for key, (a, b) in PAIRS.items():
                (
                    row[key + "_flip"],
                    row[key + "_class"],
                    row[key + "_tv"],
                    row[key + "_uniform"],
                    row[key + "_answer_delta"],
                    row[key + "_identical"],
                ) = differences(p[a], p[b], nxt["u"])
            raw.append(row)
            if near(prev):
                focus.append(
                    {
                        **row,
                        **{
                            key: action_at(v, nxt["u"])
                            for key, v in zip(
                                (
                                    "pre_global_action",
                                    "post_global_action",
                                    "pre_local_action",
                                    "post_local_action",
                                    "one_local_common_action",
                                ),
                                p,
                                strict=True,
                            )
                        },
                    }
                )
    assert len(focus) == 680 and len({r["seed"] for r in raw}) == 24
    result = {}
    per_seed = []
    for exposure in ("all_successors", "practice", "practice_answer"):
        for origin in ("both", *BIRTHS):
            group = [
                r
                for r in raw
                if (exposure == "all_successors" or r[exposure])
                and (origin == "both" or r["birth"] == origin)
            ]
            seeds = sorted({r["seed"] for r in group})
            chunks = [[r for r in group if r["seed"] == s] for s in seeds]
            by = {}
            for name in PAIRS:
                flips = sum(r[name + "_flip"] for r in group)
                per = []
                for seed, chunk in zip(seeds, chunks, strict=True):
                    v = [r[name + "_flip"] for r in chunk]
                    per.append(float(np.mean(v)))
                    per_seed.append(
                        {
                            "exposure": exposure,
                            "birth": origin,
                            "comparison": name,
                            "seed": seed,
                            "n": len(v),
                            "flips": sum(v),
                            "class_flips": sum(r[name + "_class"] for r in chunk),
                            "mean_tv": float(np.mean([r[name + "_tv"] for r in chunk])),
                        }
                    )
                columns = {
                    k: [r[name + "_" + k] for r in group]
                    for k in ("class", "identical", "tv", "uniform", "answer_delta")
                }
                by[name] = {
                    "semantic_flips": flips,
                    "class_flips": sum(columns["class"]),
                    "identical_distributions": sum(columns["identical"]),
                    "mean_tv": float(np.mean(columns["tv"])),
                    "mean_exact_shared_uniform_disagreement": float(
                        np.mean(columns["uniform"])
                    ),
                    "mean_answer_probability_delta": float(
                        np.mean(columns["answer_delta"])
                    ),
                    "seed_flip_rate": ci(seeds, per, (OLD_PILOT, OLD_REP)),
                }
            steps = {
                k: float(np.median([r[k] for r in group]))
                for k in ("global_step", "local_own_step", "local_common_step")
            }
            by["steps"] = dict(
                **steps,
                ratio_of_common_to_global_medians=steps["local_common_step"]
                / steps["global_step"],
                local_negative=sum(r["local_signal"] < -1e-9 for r in group),
                local_positive=sum(r["local_signal"] > 1e-9 for r in group),
            )
            by["events"] = len(group)
            result[exposure + "_" + origin] = by
    save_csv(dest / "replay-practice-answers.csv", focus)
    save_csv(dest / "replay-seeds.csv", per_seed)
    result["previous_device"] = {
        "turns": len(diag),
        "exact_probabilities": sum(v[0] == 0 for v in diag),
        "matched_after_hash": sum(v[1] for v in diag),
        "max_probability_error": max(v[0] for v in diag),
    }
    write(dest / "replay.json", result)
    return result


def live_row(d):
    m = json.loads((d / "life.json").read_bytes())
    ev = records(d)
    first = next((e for e in ev if e["first_fatal"]), None)
    healthy = [e for e in ev if e["health"] >= 90]
    healthy_tasks = sum(e["action"] >= 2 and e["after"] > 0 for e in healthy)
    near = [e for e in ev if e["practice"] and e["action"] >= 2]
    r = {
        k: m[k]
        for k in ("seed", "birth", "rule", "turns", "practice_count", "first_fatal")
    }
    r.update(
        reached=int(first is not None),
        healthy_turns=len(healthy),
        healthy_completed=healthy_tasks,
        healthy_rate=healthy_tasks / len(healthy),
        near_answers=len(near),
        near_positive=sum(e["signal"] > 1e-9 for e in near),
        near_negative=sum(e["signal"] < -1e-9 for e in near),
        avoided=int(first is not None and first["after"] > 0),
    )
    if first is None:
        return r
    r.update(first_health=first["health"], chosen=first["word"], u=first["u"][0])
    with np.load(d / "predecision.npz") as z:
        same = z["same_request"].astype(float)[:, :, 2:].sum(-1)
        panel = z["panel"].astype(float)[:, :, :, 2:].sum(-1).mean(1)
    last_role = "extra" if "first_extra" in m["snapshot_hashes"] else "entry"
    for j, role in enumerate(("birth", "current", "wrong", "other", last_role)):
        if role in ("wrong", "other") and not m["rule"].endswith("_true"):
            continue
        for key, value in (
            ("fatal", same[0, j]),
            ("healthy_same", same[1:, j].mean()),
            ("fatal_panel", panel[0, j]),
            ("healthy_panel", panel[1:, j].mean()),
        ):
            r[key + "_" + role] = float(value)
    return r


def lives_summary(roots, dest):
    tab = [
        live_row(f.parent)
        for root in roots
        for f in sorted(root.glob("s*/*/*/life.json"))
    ]
    assert (
        len(tab) == 240
        and len({(r["seed"], r["birth"], r["rule"]) for r in tab}) == 240
    )
    save_csv(dest / "lives.csv", tab)
    result = {}
    for b in BIRTHS:
        groups = {
            arm: sorted(
                (r for r in tab if r["birth"] == b and r["rule"] == arm),
                key=lambda r: r["seed"],
            )
            for arm in ARMS
        }
        assert all(
            [r["seed"] for r in v] == list(PILOT + REPLICATION) for v in groups.values()
        )
        summary = {"arms": {}, "learners": {}, "free_comparisons": {}}
        for arm, group in groups.items():
            live = [r for r in group if r["reached"]]
            summary["arms"][arm] = {
                "lives": len(group),
                "reached": len(live),
                "survived": sum(r["avoided"] for r in group),
                "pilot_survived": sum(
                    r["avoided"] for r in group if r["seed"] in PILOT
                ),
                "replicate_survived": sum(
                    r["avoided"] for r in group if r["seed"] in REPLICATION
                ),
                "turns": sum(r["turns"] for r in group),
                "healthy_completed": sum(r["healthy_completed"] for r in group),
                "healthy_turns": sum(r["healthy_turns"] for r in group),
                "mean_healthy_rate": float(np.mean([r["healthy_rate"] for r in group])),
                "mean_first_answer": float(np.mean([r["fatal_current"] for r in live]))
                if live
                else None,
                "near_answers": sum(r["near_answers"] for r in group),
                "near_positive": sum(r["near_positive"] for r in group),
                "near_negative": sum(r["near_negative"] for r in group),
            }
            if not arm.endswith("_true"):
                continue
            contr = {}
            for role in ("birth", "wrong", "other", "entry"):
                for key in ("fatal", "fatal_panel", "healthy_same", "healthy_panel"):
                    contr[key + "_vs_" + role] = ci(
                        [r["seed"] for r in live],
                        [r[key + "_current"] - r[key + "_" + role] for r in live],
                    )
            success = len(live) == len(group)
            for role in ("birth", "wrong", "other"):
                v = contr["fatal_vs_" + role]
                w = contr["fatal_panel_vs_" + role]
                success = (
                    success
                    and v["upper_adjusted"] < 0
                    and max(v["pilot"], v["replication"], w["pilot"], w["replication"])
                    < 0
                )
            health = contr["healthy_panel_vs_birth"]
            success = success and health["lower_adjusted"] > -0.02
            summary["learners"][arm] = {
                "contrasts": contr,
                "passes_prelocked_adaptive_criteria": bool(success)
                if arm == "local_true"
                else None,
            }
        for arm in ("frozen", "global_true", "global_wrong", "local_wrong"):
            x, y = groups["local_true"], groups[arm]
            both = [
                (a, b)
                for a, b in zip(x, y, strict=True)
                if a["reached"] and b["reached"]
            ]
            summary["free_comparisons"][arm] = {
                "avoided": discord(
                    [(a["avoided"], b["avoided"]) for a, b in zip(x, y, strict=True)]
                ),
                "healthy_local_minus_other": ci(
                    [a["seed"] for a in x],
                    [
                        a["healthy_rate"] - b["healthy_rate"]
                        for a, b in zip(x, y, strict=True)
                    ],
                ),
                "fatal_local_minus_other": ci(
                    [a["seed"] for a, b in both],
                    [a["fatal_current"] - b["fatal_current"] for a, b in both],
                ),
            }
        result[b] = summary
    write(dest / "live.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path)
    p.add_argument("--lives", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    if a.replay:
        replay_summary(a.replay, a.output)
    if a.lives:
        lives_summary(a.lives, a.output)
