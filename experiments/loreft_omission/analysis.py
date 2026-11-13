"""Read full fixed-history forks; positive effects mean the omitted update supported the target."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.individual_life.design import offers
from experiments.lifetime_cooperation.design import rules
from .design import (
    ACTIONS,
    ARMS,
    GRID,
    ITEMS,
    PARTNER,
    actual_labels,
    branch_key,
    event_plan,
    final_run,
    histories,
    majority,
    panel_labels,
    reference_age,
    starts,
)


def dump_table(path, rows):
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def load_trace(folder, normal, actual, omitted):
    first = min(omitted)
    with np.load(folder / "forks" / branch_key(omitted) / "trace.npz") as z:
        assert int(z["first"]) == first and np.array_equal(z["omitted"], omitted)
        g = np.concatenate([normal[: first - 1], z["panels"]])
        y = np.concatenate([actual[: first - 1], z["actual"]])
        assert len(g) == len(normal) and np.array_equal(g[first - 1], normal[first - 1])
    return g, y


def measures(x, seed, a):
    """x = health × context × four semantic choices. Return percentages."""
    x = x.astype(float)
    ans = x[:, :, 2:].sum(-1)
    h = x[2:].mean(axis=0)
    result = {
        k: 100 * float(h[PARTNER == p, a].mean()) for k, p in (("p86", 2), ("p93", 3))
    }
    for j, health in enumerate((2, 6, 90, 300)):
        for p, name in ((2, "p86"), (3, "p93")):
            result[f"H{health}_{name}"] = 100 * float(x[j, PARTNER == p, a].mean())
    chance = np.array([offers(ITEMS[i], rules(seed, "stable", 0))[0][2:] for i in GRID])
    result.update(
        healthy_task=100 * ans[2:].mean(),
        rescued_task=100 * (x[1, :, 2:] * chance).sum(-1).mean(),
        total_death=100 * (x[0, :, 0] + 0.85 * x[0, :, 1] + ans[0]).mean(),
        healthy_top_share=100 * (x[2:].argmax(-1) == a).mean(),
    )
    return result


def label_sequences(g, y, birth):
    return {
        "actual": starts(actual_labels(y[:-1], birth[:-1])),
        "panel": starts(panel_labels(g)),
        "final": final_run(majority(g)),
    }


def qualifies(seq, family, t, a):
    if family == "no_onset":
        return None
    if family == "final":
        final = seq["final"]
        return bool(final and final[1] == a and t >= final[0])
    return seq[family].get(t) == a


def next_target(seq, family, t, a):
    if family == "final":
        v = seq["final"]
        return v[0] if v and v[1] == a else None
    if family == "no_onset":
        return None
    return next((k for k, v in seq[family].items() if k >= t and v == a), None)


def desc(rows, key):
    values = np.asarray([r[key] for r in rows], float)
    return (
        dict(
            n=len(values),
            mean=float(values.mean()),
            median=float(np.median(values)),
            range=[float(values.min()), float(values.max())],
        )
        if len(values)
        else None
    )


def reduced(rows):
    keys = (
        "delta_p86",
        "delta_p93",
        "delta_min",
        "delta_healthy_task",
        "delta_rescued_task",
        "delta_total_death",
    )
    return dict(
        n=len(rows),
        both_5=sum(r["both_five"] for r in rows),
        breaks=sum(r["broken"] is True for r in rows),
        material=sum(r["material"] is True for r in rows),
        **{k: desc(rows, k) for k in keys},
    )


def analyze(root, inputs, out):
    histories_ = histories(inputs)
    frac = reference_age(histories_)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    effect = []
    experiences = []
    daily = []
    branch_pairs = []
    per_seed = []
    evidence = []
    for seed, (src, ref, _, _) in histories_.items():
        folder = root / "lives" / f"s{seed}"
        plan = json.loads((folder / "plan.json").read_bytes())
        with np.load(folder / "normal.npz") as z:
            normal = z["panels"]
            actual = z["actual"]
            birth = z["birth_actual"]
        events = event_plan(src, actual, birth, normal, frac)
        assert events == plan["events"]
        n = len(src)
        seq0 = label_sequences(normal, actual, birth)
        ylab = actual_labels(actual, birth)
        plab = panel_labels(normal)
        maj = majority(normal)
        for t, e in enumerate(src, 1):
            row = ITEMS[e["row"]]
            for a in range(4):
                m = measures(normal[t - 1], seed, a)
                daily.append(
                    dict(
                        seed=seed,
                        decision=t,
                        physical_health=e["health"],
                        actual_partner=row["partner"],
                        action=ACTIONS[a],
                        actual_prob=float(actual[t - 1, a]),
                        actual_argmax=int(actual[t - 1].argmax()),
                        high_actual=int(ylab[t - 1]),
                        high_panel=int(plab[t - 1]),
                        healthy_majority=int(maj[t - 1]),
                        **m,
                    )
                )
        cache = {}
        for entry in events:
            event = entry["id"]
            family = entry["family"]
            target = entry["action"]
            t = entry["turn"]
            control = entry["control"]
            meta.append(
                dict(
                    seed=seed,
                    event=event,
                    previous=ACTIONS[entry["previous"]]
                    if entry["previous"] is not None
                    else "",
                    target=ACTIONS[target],
                    anchor=t,
                    health=src[t - 1]["health"],
                    ordinary=control or "",
                    ordinary_health=src[control - 1]["health"] if control else "",
                    match_distance=entry["match_distance"],
                )
            )
            for purpose, anchor in (("event", t), ("ordinary", control)):
                if anchor is None or anchor < 4:
                    continue
                for offset in (-3, -2, -1):
                    u = anchor + offset
                    e = src[u - 1]
                    r = ITEMS[e["row"]]
                    m0 = measures(normal[u - 1], seed, target)
                    m1 = measures(normal[u], seed, target)
                    experiences.append(
                        dict(
                            seed=seed,
                            event=event,
                            window=purpose,
                            anchor=anchor,
                            update=u,
                            offset=offset,
                            partner=("P86", "P93")[r["partner"] - 2],
                            frame=r["frame"],
                            task=r["task"],
                            source_action=ACTIONS[e["action"]],
                            maintenance=e["maintenance"],
                            pre_health=e["health"],
                            post_health=e["after"],
                            body_loss=e["health"] - e["after"],
                            feedback=e["reward"],
                            signal=e["signal"],
                            actual_parameter_step=ref[u - 1]["updates"]["loreft"][
                                "actual_l2"
                            ],
                            target=ACTIONS[target],
                            immediate_p86=m1["p86"] - m0["p86"],
                            immediate_p93=m1["p93"] - m0["p93"],
                        )
                    )
                for arm, offsets in ARMS.items():
                    om = tuple(anchor + i for i in offsets)
                    if om not in cache:
                        g, y = load_trace(folder, normal, actual, om)
                        cache[om] = (g, y, label_sequences(g, y, birth))
                    g, y, seq = cache[om]
                    matched = qualifies(seq0, family, anchor, target)
                    kept = qualifies(seq, family, anchor, target)
                    broken = (
                        (not kept) if matched else (False if matched is False else None)
                    )
                    first = next_target(seq, family, anchor, target)
                    endmode = int(
                        np.bincount(g[-1, 2:].argmax(-1).ravel(), minlength=4).argmax()
                    )
                    for tag, at in (
                        ("at", anchor),
                        ("plus2", anchor + 2),
                        ("plus8", anchor + 8),
                        ("last", n),
                    ):
                        if at > n:
                            continue
                        std = measures(normal[at - 1], seed, target)
                        alt = measures(g[at - 1], seed, target)
                        delta = {
                            f"delta_{k}": float(std[k] - v) for k, v in alt.items()
                        }
                        low = min(delta["delta_p86"], delta["delta_p93"])
                        both = bool(low >= 5)
                        effect.append(
                            dict(
                                seed=seed,
                                event=event,
                                family=family,
                                kind=entry["kind"],
                                window=purpose,
                                arm=arm,
                                anchor=anchor,
                                target=ACTIONS[target],
                                time=tag,
                                decision=at,
                                omitted=",".join(map(str, om)),
                                normal_qualifies=matched,
                                omitted_qualifies=kept,
                                broken=broken,
                                next_target_start=first if first is not None else "",
                                last_mode=ACTIONS[endmode],
                                normal_actual_prob=100 * float(actual[at - 1, target]),
                                omitted_actual_prob=100 * float(y[at - 1, target]),
                                actual_difference=100
                                * float(actual[at - 1, target] - y[at - 1, target]),
                                both_five=both,
                                material=(bool(broken and both) if matched else None),
                                delta_min=low,
                                normal_p86=std["p86"],
                                normal_p93=std["p93"],
                                omitted_p86=alt["p86"],
                                omitted_p93=alt["p93"],
                                **delta,
                            )
                        )
        seed_records = [r for r in effect if r["seed"] == seed and r["time"] == "at"]
        by = {(r["event"], r["window"], r["arm"]): r for r in seed_records}
        for e in events:
            if e["turn"] < 4:
                continue
            for arm in ARMS:
                one = by[e["id"], "event", arm]
                other = by.get((e["id"], "ordinary", arm))
                if other:
                    branch_pairs.append(
                        dict(
                            seed=seed,
                            event=e["id"],
                            family=e["family"],
                            kind=e["kind"],
                            arm=arm,
                            target=one["target"],
                            event_turn=e["turn"],
                            ordinary_turn=e["control"],
                            match_distance=e["match_distance"],
                            event_min=one["delta_min"],
                            ordinary_min=other["delta_min"],
                            paired_min=one["delta_min"] - other["delta_min"],
                            paired_p86=one["delta_p86"] - other["delta_p86"],
                            paired_p93=one["delta_p93"] - other["delta_p93"],
                            event_both=one["both_five"],
                            ordinary_both=other["both_five"],
                            event_material=one["material"],
                        )
                    )
            b = by[e["id"], "event", "skip3"]
            single = [
                by[e["id"], "event", a] for a in ("only_minus3", "only_minus2", "skip1")
            ]
            evidence.append(
                dict(
                    seed=seed,
                    event=e["id"],
                    target=b["target"],
                    turn=e["turn"],
                    window_material=b["material"],
                    individual_material=",".join(
                        str(v["omitted"]) for v in single if v["material"]
                    ),
                    joint_min=b["delta_min"],
                    single_minimums=",".join(f"{v['delta_min']:.4f}" for v in single),
                )
            )
        actualfirst = [
            e for e in events if e["family"] == "actual" and e["kind"] == "onset"
        ]
        firstpanel = [
            e for e in events if e["family"] == "panel" and e["kind"] == "onset"
        ]
        final = final_run(maj)
        mode = int(np.bincount(normal[-1, 2:].argmax(-1).ravel(), minlength=4).argmax())
        per_seed.append(
            dict(
                seed=seed,
                states=n,
                first_actual=actualfirst[0]["turn"] if actualfirst else "",
                actual_target=ACTIONS[actualfirst[0]["action"]] if actualfirst else "",
                first_panel=firstpanel[0]["turn"] if firstpanel else "",
                panel_target=ACTIONS[firstpanel[0]["action"]] if firstpanel else "",
                last_mode=ACTIONS[mode],
                final_established=final[0] if final else "",
                actual_rewrites=",".join(
                    f"{e['turn']}:{ACTIONS[e['action']]}"
                    for e in events
                    if e["family"] == "actual" and e["kind"] == "rewrite"
                ),
                panel_rewrites=",".join(
                    f"{e['turn']}:{ACTIONS[e['action']]}"
                    for e in events
                    if e["family"] == "panel" and e["kind"] == "rewrite"
                ),
                unique_forks=len(cache),
            )
        )
    for filename, values in (
        ("events", meta),
        ("event-effects", effect),
        ("prior-experiences", experiences),
        ("normal-stepwise", daily),
        ("matched-ordinary", branch_pairs),
        ("attribution", evidence),
        ("lives", per_seed),
    ):
        dump_table(out / (filename + ".csv"), values)
    groups = {}
    for family, kind in (
        ("actual", "onset"),
        ("actual", "rewrite"),
        ("panel", "onset"),
        ("panel", "rewrite"),
        ("final", "established"),
        ("no_onset", "reference"),
    ):
        for arm in ARMS:
            s = [
                r
                for r in effect
                if (r["family"], r["kind"], r["window"], r["arm"], r["time"])
                == (family, kind, "event", arm, "at")
            ]
            ords = [
                r
                for r in effect
                if (r["family"], r["kind"], r["window"], r["arm"], r["time"])
                == (family, kind, "ordinary", arm, "at")
            ]
            paired = [
                r
                for r in branch_pairs
                if (r["family"], r["kind"], r["arm"]) == (family, kind, arm)
            ]
            # For repeated rewrites, average within a life before interpreting the sample average.
            byseed = defaultdict(list)
            for row in paired:
                byseed[row["seed"]].append(row["paired_min"])
            signed = [np.mean(v) for v in byseed.values()]
            groups[f"{family}_{kind}_{arm}"] = dict(
                event=reduced(s),
                ordinary=reduced(ords),
                matched_windows=len(paired),
                seeds_matched=len(byseed),
                life_mean_matched_minimum=float(np.mean(signed)) if signed else None,
            )
    result = dict(
        independent_lives=len(per_seed),
        events=len(meta),
        unique_forks=sum(v["unique_forks"] for v in per_seed),
        ordinary_matched=sum(r["ordinary"] != "" for r in meta),
        all_groups=groups,
    )
    write(out / "summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = analyze(a.root, a.inputs, a.out)
    print(
        {
            k: data[k]
            for k in ("independent_lives", "events", "unique_forks", "ordinary_matched")
        }
    )
