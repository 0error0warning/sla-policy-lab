"""Score held, task-matched stage interventions without treating repeated queries as independent lives."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import settle, write
from experiments.individual_life.analysis import describe
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT

from .design import HEALTH, STAGES, VIEWS, shared_draw, shared_joint

PAIRS = (("oracle", "neutral"), ("learned", "neutral"), ("oracle", "learned"))
SPLITS = ("original", "held")


def probability_scores(p, fav, truth):
    p = np.asarray(p, dtype=float)
    ix = np.arange(len(fav))
    other = 5 - fav  # action codes 2 and 3
    high, low = p[ix, fav], p[ix, other]
    total = high + low
    top_actions = p.argmax(-1)
    most_common = int(np.bincount(top_actions, minlength=4).argmax())
    return {
        "preference": float(100 * (high - low).mean()),
        "favorable": float(100 * high.mean()),
        "other": float(100 * low.mean()),
        "answer": float(100 * total.mean()),
        "conditional_favorable": float(100 * high.sum() / total.sum()),
        "cooperative_survival": float(
            100 * np.mean(p[:, 2] * truth[:, 1] + p[:, 3] * truth[:, 2])
        ),
        "concentration": float(100 * np.mean(p.max(-1) >= 0.9)),
        "top_action_mode": most_common,
        "same_top_action": float(100 * np.mean(top_actions == most_common)),
        "top_probability": float(100 * p.max(-1).mean()),
    }


def contrast(a, b, ua, fav):
    """Compare a against b with the same uniform and integrate the same coupling exactly."""
    a, b = a.astype(float), b.astype(float)
    sampled = shared_draw(np.stack((a, b), axis=1), ua[:, None])
    act, base = sampled.T
    cross = (act >= 2) & (base >= 2) & (act != base)
    greedy_a, greedy_b = a.argmax(-1), b.argmax(-1)
    joint = shared_joint(b, a)
    tv = 0.5 * abs(a - b).sum(-1)
    return {
        "sample_changed": int(np.sum(act != base)),
        "sample_exchanged": int(cross.sum()),
        "toward_good": int(np.sum(cross & (act == fav))),
        "away_from_good": int(np.sum(cross & (base == fav))),
        "enter_answer": int(np.sum((act >= 2) & (base < 2))),
        "leave_answer": int(np.sum((act < 2) & (base >= 2))),
        "greedy_changed": int(np.sum(greedy_a != greedy_b)),
        "greedy_exchanged": int(
            np.sum((greedy_a >= 2) & (greedy_b >= 2) & (greedy_a != greedy_b))
        ),
        "exact_sample_changed": float(
            100 * np.mean(1 - np.trace(joint, axis1=1, axis2=2))
        ),
        "exact_sample_exchanged": float(100 * np.mean(joint[:, 2, 3] + joint[:, 3, 2])),
        "mean_tv": float(100 * tv.mean()),
        "max_tv": float(100 * tv.max()),
        "different_rows": int((tv > 1e-10).sum()),
        "total": len(fav),
    }


def decisions(rows, s, k, w, h, ix, p, forecast, u, noise):
    truth = forecast[:, 1]
    fav = 2 + truth[:, 1:].argmax(-1)
    meta = {"seed": s, "stage": STAGES[k], "wording": SPLITS[w], "health": h}
    for j, i in enumerate(ix):
        row = ITEMS[i]
        for m, kind in enumerate(VIEWS):
            a = int(shared_draw(p[j, m], u[j, 0]))
            maintained = None if a == 0 else bool(u[j, 1] < truth[j, a - 1])
            after = settle(h, a, maintained, noise[j])[0]
            rows.append(
                dict(
                    **meta,
                    prompt_id=int(i),
                    partner=("P86", "P93")[row["partner"] - 2],
                    frame=row["frame"],
                    layout=row["layout"],
                    task=row["task"],
                    input=kind,
                    favored=int(fav[j]),
                    action=a,
                    greedy=int(p[j, m].argmax()),
                    matched_maintenance=maintained,
                    after=after,
                    completed=int(a >= 2 and after > 0),
                )
            )
    return fav


def seed_scores(src, seed, decision_rows):
    stage, comp, dense = [], [], []
    folder = src / "lives" / f"s{seed}"
    with np.load(folder / "stages.npz") as z, np.load(folder / "states.npz") as st:
        grid, q, small, dq = (z[n] for n in ("prob", "q", "dense_prob", "dense_q"))
        u, noise = st["uniforms"], st["noise"]
        params, sel, observation, source_h = (
            st["policy"],
            st["selected"],
            st["observed"],
            st["health"],
        )
        for k, t in enumerate(sel):
            for w, ix in enumerate(PANEL):
                pf = q[k, w]
                for hi, h in enumerate(HEALTH):
                    p = grid[k, w, hi]
                    fav = decisions(decision_rows, seed, k, w, h, ix, p, pf, u, noise)
                    meta = {
                        "seed": seed,
                        "stage": STAGES[k],
                        "completed": int(t),
                        "observed": int(observation[t]),
                        "historical_health": float(source_h[t]),
                        "wording": SPLITS[w],
                        "health": h,
                    }
                    metrics = {}
                    for m, kind in enumerate(VIEWS):
                        stats = probability_scores(p[:, m], fav, pf[:, 1])
                        if h != 6:
                            stats["cooperative_survival"] = stats["answer"]
                        forecast_mse = float(
                            np.mean((pf[:, m, 1:] - pf[:, 1, 1:]) ** 2)
                        )
                        act = shared_draw(p[:, m], u[:, 0])
                        answered = act >= 2
                        kept = (
                            (
                                u[:, 1]
                                < pf[np.arange(len(ix)), 1, np.maximum(act - 1, 0)]
                            )
                            if h == 6
                            else np.ones(len(ix), dtype=bool)
                        )
                        stage.append(
                            dict(
                                **meta,
                                input=kind,
                                **stats,
                                prediction_mse=forecast_mse,
                                sample_answers=int(answered.sum()),
                                sample_useful=int((act == fav).sum()),
                                sample_completed=int((answered & kept).sum()),
                                opportunities=len(ix),
                            )
                        )
                        metrics[kind] = stats
                    for target, ref in PAIRS:
                        ai, bi = VIEWS.index(target), VIEWS.index(ref)
                        dif = {
                            m: metrics[target][m] - metrics[ref][m]
                            for m in (
                                "preference",
                                "answer",
                                "cooperative_survival",
                                "conditional_favorable",
                            )
                        }
                        comp.append(
                            dict(
                                **meta,
                                target=target,
                                reference=ref,
                                **dif,
                                **contrast(p[:, ai], p[:, bi], u[:, 0], fav),
                            )
                        )
        for t, p in enumerate(small):
            f = dq[t]
            fav = 2 + f[:, 1, 1:].argmax(-1)
            by = [probability_scores(p[:, i], fav, f[:, 1]) for i in range(3)]
            d = {
                k: float(by[i]["preference"] - by[2]["preference"])
                for i, k in enumerate(("learned_preference", "oracle_preference"))
            }
            tv = 0.5 * abs(p[:, 1].astype(float) - p[:, 2]).sum(-1)
            dense.append(
                dict(
                    seed=seed,
                    completed=t,
                    total=len(small) - 1,
                    historical_health=float(source_h[t]),
                    observed=int(observation[t]),
                    **d,
                    neutral_concentration=by[2]["concentration"],
                    oracle_tv=100 * float(tv.mean()),
                    oracle_max_tv=100 * float(tv.max()),
                    readin_norm=float(np.linalg.norm(params[t, -6:].astype(float))),
                    learned_mse=float(((f[:, 0, 1:] - f[:, 1, 1:]) ** 2).mean()),
                )
            )
    return stage, comp, dense


def distribution(values):
    v = np.asarray(values, dtype=float)
    assert len(v) == len(PILOT)
    return {**describe(v), "paired": v.tolist()}


def summarize(stage, comp):
    metrics = {}
    criteria = {}

    def pick(table, stage_name, key, health, **conditions):
        data = [
            v
            for v in table
            if v["stage"] == stage_name
            and v["wording"] == "held"
            and v["health"] in health
            and all(v[k] == t for k, t in conditions.items())
        ]
        return np.array(
            [np.mean([v[key] for v in data if v["seed"] == s]) for s in PILOT]
        )

    effects = {}
    for name in STAGES:
        for target, ref in PAIRS:
            pair = target + "_" + ref
            d = {
                key: distribution(
                    pick(comp, name, key, (6.0,), target=target, reference=ref)
                )
                for key in (
                    "preference",
                    "answer",
                    "cooperative_survival",
                    "conditional_favorable",
                    "mean_tv",
                    "exact_sample_exchanged",
                    "exact_sample_changed",
                )
            }
            d["healthy_difference"] = distribution(
                pick(comp, name, "answer", (90.0, 300.0), target=target, reference=ref)
            )
            counts = [
                r
                for r in comp
                if r["stage"] == name
                and r["wording"] == "held"
                and r["health"] == 6
                and r["target"] == target
                and r["reference"] == ref
            ]
            d["counts"] = {
                k: int(sum(v[k] for v in counts))
                for k in (
                    "sample_changed",
                    "sample_exchanged",
                    "toward_good",
                    "away_from_good",
                    "enter_answer",
                    "leave_answer",
                    "greedy_changed",
                    "greedy_exchanged",
                    "different_rows",
                    "total",
                )
            }
            metrics[name + "." + pair] = d
            if ref == "neutral":
                effects[name, target] = np.array(d["preference"]["paired"])
                use = d["preference"]["mean"] >= 1 and d["preference"]["positive"] >= 6
                keep = d["healthy_difference"]["mean"] >= -2
                criteria[name + "." + target] = {
                    "useful": bool(use),
                    "healthy_retained": bool(keep),
                    "joint": bool(use and keep),
                }
        metrics[name + ".concentration"] = distribution(
            pick(stage, name, "concentration", (6.0,), input="neutral")
        )
    loss = distribution(effects["early", "oracle"] - effects["late", "oracle"])
    increase = distribution(
        np.array(metrics["late.concentration"]["paired"])
        - metrics["early.concentration"]["paired"]
    )
    late_lock = (
        criteria["early.oracle"]["useful"]
        and not criteria["late.oracle"]["useful"]
        and loss["mean"] >= 1
        and loss["positive"] >= 6
        and increase["mean"] >= 10
        and increase["positive"] >= 6
    )
    return {
        "by_stage": metrics,
        "criteria": criteria,
        "paired_early_minus_late_oracle": loss,
        "paired_late_minus_early_concentration": increase,
        "early_use_then_lock": bool(late_lock),
    }


def save_choices(path, rows):
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as pack,
        io.TextIOWrapper(pack, encoding="utf8", newline="") as f,
    ):
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def analyze(src, out):
    out.mkdir(parents=True, exist_ok=True)
    stage, comp, dense, choice = [], [], [], []
    for seed in PILOT:
        ss, cc, dd = seed_scores(src, seed, choice)
        stage.extend(ss)
        comp.extend(cc)
        dense.extend(dd)
    for name, rows in (("stages", stage), ("contrasts", comp), ("dense", dense)):
        csv_out(out / (name + ".csv"), rows)
    save_choices(out / "choices.csv.gz", choice)
    v = summarize(stage, comp)
    write(out / "summary.json", v)
    return v


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("src", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.src, a.out)["criteria"]))
