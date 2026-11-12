"""Paired next-alive decisions and effects attributable to the individual's own updates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import rng, write
from experiments.individual_life.design import BIRTHS, PILOT, PROBE_H, offers, rows
from experiments.lethal_boundary.physics import integration
from experiments.social_closed_loop.core import BodyValue


def records(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]


def describe(a):
    x = np.asarray(a, dtype=float)
    if not len(x):
        return None
    boot = rng(68901, "whole-life-bootstrap", len(x)).integers(
        len(x), size=(12000, len(x))
    )
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "range": [float(x.min()), float(x.max())],
        "positive": int(np.sum(x > 1e-9)),
        "negative": int(np.sum(x < -1e-9)),
        "ci95": np.quantile(x[boot].mean(1), [0.025, 0.975]).tolist(),
    }


class Evaluator:
    def __init__(self, all_h, items):
        self.axis = np.unique(np.r_[list(all_h), PROBE_H])
        self.rows = items
        self.death, self.value = integration(self.axis, BodyValue())
        self.probe_rows = [r for r in items if (r["task"], r["wording"]) == (5, 4)]

    def criteria(self, h, row, bits):
        j = np.searchsorted(self.axis, h)
        assert self.axis[j] == h
        p, fav = offers(row, bits)
        q = (1 - p) * self.value[j, :, 0] + p * self.value[j, :, 1]
        death = (1 - p) * self.death[j, :, 0] + p * self.death[j, :, 1]
        utility = (q - q.min()) / np.ptp(q) if np.ptp(q) > 1e-9 else np.zeros(4)
        return q, utility, death, fav

    def probe(self, prob, bits):
        by = []
        assert prob.shape == (4, len(self.probe_rows), 4)
        for hi, h in enumerate(PROBE_H):
            m = []
            for p, r in zip(prob[hi], self.probe_rows, strict=True):
                q, util, d, f = self.criteria(h, r, bits)
                ans = p[2:].sum()
                m.append([ans, p @ d, p[2:] @ (1 - d[2:]), p[f], p @ q, p @ util])
            by.append(np.mean(m, axis=0))
        a = np.asarray(by)
        return {
            "danger_answer": float(a[0, 0]),
            "danger_total_death": float(a[0, 1]),
            "rescue_safe_task": float(a[1, 2]),
            "healthy_answer": float(a[2:, 0].mean()),
            "healthy_preferred_expression": float(a[2:, 3].mean()),
            "healthy_minus_doomed_answer": float(a[2:, 0].mean() - a[0, 0]),
            "mean_reference_quality": float(a[:, 5].mean()),
        }


def assess(d, ev):
    meta = json.loads((d / "life.json").read_bytes())
    turns = records(d / "life.jsonl")
    updates = records(d / "updates.jsonl")
    out = {
        "turns": len(turns),
        "tasks": sum(t["action"] >= 2 for t in turns),
        "safe_tasks": sum(t["action"] >= 2 and not t["terminal"] for t in turns),
        "received_maintenance": sum(t["maintenance"] is True for t in turns),
        "main_steps": meta["main_steps"],
        "steps_with_a_living_successor": sum(
            t["updates_after_settlement"] for t in turns[:-1]
        ),
        "terminal_steps": turns[-1]["updates_after_settlement"],
        "actual_doomed_encounters": sum(
            t["health_before"] <= 2.4414549078 for t in turns
        ),
        "actual_doomed_answers": sum(
            t["health_before"] <= 2.4414549078 and t["action"] >= 2 for t in turns
        ),
        "actual_rescuable_encounters": sum(
            4.520616 < t["health_before"] < 7.994132 for t in turns
        ),
        "already_healthy_encounters": sum(t["health_before"] >= 90 for t in turns),
    }
    points = []
    examples = {}
    for k in range(1, len(turns)):
        t = turns[k]
        prior = turns[k - 1]
        row = ev.rows[t["context"]]
        q, u, death, fav = ev.criteria(t["health_before"], row, meta["world"])
        cur, pre, birth = [
            np.array(t["prob_" + name]) for name in ("current", "previous", "birth")
        ]
        shadow = t["prob_wrong_shadow"]
        deferred = t["prob_deferred_same_tape"]
        delta = cur - pre
        since_birth = cur - birth
        pt = {
            "turn": t["turn"],
            "h": t["health_before"],
            "L": float(delta @ u),
            "L_raw": float(delta @ q),
            "G": float(since_birth @ u),
            "G_raw": float(since_birth @ q),
            "TV": float(np.abs(delta).sum() / 2),
            "TV_birth": float(np.abs(since_birth).sum() / 2),
            "immediate_flip": float(t["action"] != t["counterfactual_previous_choice"]),
            "vs_birth_flip": float(t["action"] != t["counterfactual_birth_choice"]),
            "immediate_death_change": float(delta @ death),
            "birth_death_change": float(since_birth @ death),
            "birth_safe_task_change": float(since_birth[2:] @ (1 - death[2:])),
            "birth_answer_change": float(since_birth[2:].sum()),
            "birth_preferred_style_change": float(since_birth[fav]),
            "cur_quality": float(cur @ u),
            "current_total_death": float(cur @ death),
            "current_safe_task": float(cur[2:] @ (1 - death[2:])),
            "released": int(prior["updates_after_settlement"] > 0),
            "same_partner": int(t["partner"] == prior["partner"]),
            "same_frame": int(t["frame"] == prior["frame"]),
            "last_action_probability_change": float(delta[prior["action"]]),
            "preceding_response": "silent"
            if prior["maintenance"] is None
            else ("maintained" if prior["maintenance"] else "skipped"),
        }
        if shadow is not None:
            diff = cur - np.asarray(shadow)
            pt["C"] = float(diff @ u)
            pt["C_raw"] = float(diff @ q)
            pt["TV_shadow"] = float(np.abs(diff).sum() / 2)
        if deferred is not None:
            pt["same_tape_immediate_over_delayed"] = float(
                (cur - np.asarray(deferred)) @ u
            )
        points.append(pt)
        key = pt["preceding_response"]
        if key not in examples:
            examples[key] = {
                "turn_that_supplied_feedback": prior["turn"],
                "previous_partner": prior["partner"],
                "previous_action": prior["word"],
                "previous_health": prior["health_before"],
                "after_health": prior["health_after"],
                "maintenance": prior["maintenance"],
                "own_feedback": prior["reference_change"],
                "advantage": prior["centered_signal"],
                "next_turn": t["turn"],
                "next_partner": t["partner"],
                "next_request": t["request"],
                "next_layout": t["layout"],
                "before_update_prob": pre.tolist(),
                "after_update_prob": cur.tolist(),
                "birth_prob": birth.tolist(),
                "previous_parameter_choice": t["counterfactual_previous_choice"],
                "actual_choice": t["action"],
                "next_value_change": pt["L"],
                "next_tv": pt["TV"],
            }
    assert len(points) == len(turns) - 1 and points
    for key in points[0]:
        if key not in ("turn", "h", "preceding_response"):
            out[key] = float(np.mean([v[key] for v in points]))
    out["next_living_decisions"] = len(points)
    out["changed_next_choices"] = int(sum(v["immediate_flip"] for v in points))
    movement = [v for v in points if v["released"]]
    out["released_next_decisions"] = len(movement)
    out["L_when_released"] = (
        float(np.mean([v["L"] for v in movement])) if movement else 0.0
    )
    out["TV_when_released"] = (
        float(np.mean([v["TV"] for v in movement])) if movement else 0.0
    )
    own = [v for v in updates if v["role"] == "main"]
    out["clipped_ratios"] = sum(
        x["importance_ratio"] != x["applied_ratio"] for x in own
    )
    out["mean_parameter_change"] = (
        float(np.mean([np.hypot(x["A_change_l2"], x["B_change_l2"]) for x in own]))
        if own
        else 0.0
    )
    subgroup = {}
    for name, predicate in (
        ("healthy", lambda x: x["h"] >= 90),
        ("rescuable", lambda x: 4.520616 < x["h"] < 7.994132),
        ("doomed", lambda x: x["h"] <= 2.4414549078),
        ("early", lambda x: x["turn"] <= 16),
        ("mid", lambda x: 16 < x["turn"] <= 32),
        ("late", lambda x: x["turn"] > 32),
        ("same_partner_frame", lambda x: x["same_partner"] and x["same_frame"]),
        (
            "other_partner_or_frame",
            lambda x: not (x["same_partner"] and x["same_frame"]),
        ),
    ):
        a = [v for v in points if predicate(v)]
        subgroup[name] = {"n": len(a)}
        if a:
            for key in (
                "L",
                "G",
                "TV",
                "birth_answer_change",
                "birth_death_change",
                "birth_safe_task_change",
                "birth_preferred_style_change",
            ):
                subgroup[name][key] = float(np.mean([v[key] for v in a]))
    held = {}
    with np.load(d / "living-probes.npz") as z:
        for marker, details in meta["living_probes"].items():
            a = ev.probe(z[marker + "-main"], meta["world"])
            held[marker] = {
                "turn": details["after_turn"],
                "real_health": details["actual_alive_health"],
                "value": a,
            }
            if marker + "-shadow" in z:
                held[marker]["shadow"] = ev.probe(z[marker + "-shadow"], meta["world"])
    for marker, entry in held.items():
        entry["change_since_own_birth"] = {
            k: v - held["birth"]["value"][k] for k, v in entry["value"].items()
        }
    return {
        "metrics": out,
        "living_subsets": subgroup,
        "probes": held,
        "first_examples": examples,
        "pointwise": points,
    }


def screen(cells, birth, seeds):
    r = [cells[f"visible/s{s}/{birth}/true1"]["metrics"] for s in seeds]
    g = np.array([v["G"] for v in r])
    c = np.array([v["C"] for v in r])
    l = np.array([v["L"] for v in r])
    for sign in (1, -1):
        joint = int(((sign * g > 0) & (sign * c > 0)).sum())
        ls = int((sign * l > 0).sum())
        if min(sign * g.mean(), sign * c.mean()) >= 0.02 and joint >= 3 and ls >= 3:
            return {
                "direction": "benefit" if sign == 1 else "harm",
                "G": describe(g),
                "C": describe(c),
                "L": describe(l),
                "joint_direction": joint,
                "immediate_direction": ls,
            }
    return {
        "direction": None,
        "G": describe(g),
        "C": describe(c),
        "L": describe(l),
        "joint_benefit": int(((g > 0) & (c > 0)).sum()),
        "joint_harm": int(((g < 0) & (c < 0)).sum()),
    }


def run(root, out):
    paths = sorted(root.glob("*/s*/*/*/life.json"))
    assert paths
    all_h = set()
    for f in paths:
        all_h.update(t["health_before"] for t in records(f.parent / "life.jsonl"))
    ev = Evaluator(all_h, rows())
    cells = {}
    groups = {}
    pairs = {}
    examples = {}
    out.mkdir(exist_ok=True, parents=True)
    with (out / "next-living.csv").open("w", encoding="utf8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "life",
                "turn",
                "health",
                "release",
                "individual_vs_previous_TV",
                "individual_vs_previous_quality",
                "individual_vs_birth_quality",
                "individual_vs_wrong_shadow_quality",
                "actual_choice_would_change",
            ]
        )
        for f in paths:
            name = f.parent.relative_to(root).as_posix()
            sense, seed, birth, rule = name.split("/")
            obj = assess(f.parent, ev)
            for t in obj.pop("pointwise"):
                writer.writerow(
                    [
                        name,
                        t["turn"],
                        t["h"],
                        t["released"],
                        t["TV"],
                        t["L"],
                        t["G"],
                        t.get("C", ""),
                        t["immediate_flip"],
                    ]
                )
            sample = obj.pop("first_examples")
            if seed in ("s68011", "s68101") and rule == "true1":
                examples[name] = sample
            cells[name] = obj
            groups.setdefault(f"{sense}/{birth}/{rule}", []).append(obj["metrics"])
    aggregate = {
        name: {k: describe([v[k] for v in vv]) for k in vv[0]}
        for name, vv in groups.items()
    }
    for name in groups:
        sense, birth, rule = name.split("/")
        if rule == "frozen":
            continue
        comps = {}
        for other in ("frozen", "wrong1", "true8", "shuffle8"):
            ab = []
            if other == rule:
                continue
            for path, entry in cells.items():
                s, seed, b, rr = path.split("/")
                if (s, b, rr) != (sense, birth, rule):
                    continue
                key = f"{s}/{seed}/{b}/{other}"
                if key in cells:
                    ab.append((entry["metrics"], cells[key]["metrics"]))
            if ab:
                comps[other] = {
                    k: describe([a[k] - b[k] for a, b in ab])
                    for k in (
                        "tasks",
                        "safe_tasks",
                        "turns",
                        "cur_quality",
                        "current_total_death",
                    )
                }
        pairs[name] = comps
    probed = {}
    for name in groups:
        ab = [
            q["probes"]
            for tag, q in cells.items()
            if "/".join([tag.split("/")[i] for i in (0, 2, 3)]) == name
        ]
        probed[name] = {}
        for marker in ("birth", "t8", "t16", "t32", "t48", "first_low"):
            obs = [v[marker] for v in ab if marker in v]
            probed[name][marker] = {"alive_observed": len(obs), "assigned": len(ab)}
            if obs:
                probed[name][marker].update(
                    {
                        k: describe([v["change_since_own_birth"][k] for v in obs])
                        for k in obs[0]["value"]
                    }
                )
                probed[name][marker]["levels"] = {
                    k: describe([v["value"][k] for v in obs]) for k in obs[0]["value"]
                }
    res = {
        "groups": aggregate,
        "closed_loop_paired": pairs,
        "read_only_probes": probed,
        "lives": len(cells),
        "encounters": sum(q["metrics"]["turns"] for q in cells.values()),
        "living_next_choices": sum(
            q["metrics"]["next_living_decisions"] for q in cells.values()
        ),
    }
    if all(f"visible/s{s}/{b}/true1" in cells for s in PILOT for b in BIRTHS):
        res["screen"] = {b: screen(cells, b, PILOT) for b in BIRTHS}
    write(out / "lives.json", cells)
    write(out / "examples.json", examples)
    write(out / "summary.json", res)
    print(
        json.dumps(
            {
                "lives": res["lives"],
                "encounters": res["encounters"],
                "screen": {
                    k: (v["direction"], v["G"]["mean"], v["C"]["mean"], v["L"]["mean"])
                    for k, v in res.get("screen", {}).items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
