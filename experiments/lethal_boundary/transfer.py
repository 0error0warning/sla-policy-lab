"""Check individual record counts, two-dimensional lookup, and continuous-body refits."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, spearmanr

from experiments.direct_policy.protocol import write
from experiments.lethal_boundary.design import CONFIRM, HEALTH, INTERIOR, QS, levels
from experiments.lethal_boundary.evaluate import (
    BANDS,
    MI,
    crossings,
    index,
    interpolation,
    physics_matrix,
    policies,
    scores,
    summaries,
    summarize,
)

JITTER_CELLS = ("fixed", "jitter", "jitter_gets_fixed", "fixed_gets_jitter")


def closed_form_record(means, mode):
    counts = np.arange(21)
    if mode == "unseen_only":
        w = binom.pmf(counts[None], 20, QS[:, None])
        w[:, [5, 17]] = 0
        w /= w.sum(1, keepdims=True)
        return np.einsum("qk,hkga->qhga", w, means, optimize=True)
    if mode == "nearest":
        s = (counts > 11).astype(float)
    else:
        s = np.clip((counts - 5) / 12, 0, 1)
    approx = (
        means[:, 5:6] * (1 - s[None, :, None, None])
        + means[:, 17:18] * s[None, :, None, None]
    )
    return policies(approx, "iid")


def one(file, axis, anchors, phys, intrinsic):
    with np.load(file) as z:
        mean = z["mean"]
    ids = index(axis, HEALTH)
    full = policies(mean, "iid")[:, ids]
    observed = scores(full, phys, intrinsic)
    calc = {
        "iid": summaries(observed),
        "unseen_only": summaries(
            scores(closed_form_record(mean, "unseen_only")[:, ids], phys, intrinsic)
        ),
        "clean": summaries(scores(policies(mean, "clean")[:, ids], phys, intrinsic)),
    }
    out = {}
    b = BANDS["transition"]
    for mode in ("nearest", "loglinear"):
        combo = interpolation(closed_form_record(mean, mode), axis, anchors, mode)
        estimate = scores(combo, phys, intrinsic)
        err = (estimate - observed)[INTERIOR][:, b]
        out[mode + "_regret_minus_model"] = float(
            err[..., MI["reference_regret"]].mean()
        )
        out[mode + "_death_minus_model"] = float(err[..., MI["total_death"]].mean())
        out[mode + "_safe_minus_model"] = float(err[..., MI["safe_task"]].mean())
        # The joint lookup uses exactly the same two training record-count anchors.
        out[mode + "_mean_absolute_answer_gap"] = float(
            np.abs(err[..., MI["answer"]]).mean()
        )
        out[mode + "_mean_abs_action_gap"] = float(
            np.abs((combo - full)[INTERIOR][:, b]).mean()
        )
    by_count = mean[ids][:, :, :, 2:].sum(-1).mean(-1)
    r = by_count[BANDS["rescue"]].mean(0)
    out["count_monotone_adjacent_share"] = float((np.diff(r) >= 0).mean())
    out["count_rank_correlation"] = (
        float(spearmanr(np.arange(21), r).statistic) if np.ptp(r) > 0 else 0.0
    )
    return observed, calc, out, crossings(observed), r


def run(root, study, analysis, out, jitter):
    out.mkdir(exist_ok=True, parents=True)
    axis = np.asarray(json.loads((root / "axes.json").read_bytes())["health"])
    with np.load(analysis / "physical.npz") as z:
        d = z["conditional_death"]
        v = z["conditional_rul_change"]
    phi = [physics_matrix(d, v, m) for m in (0, 1)]
    seeds = list(range(67201, 67213)) if jitter else list(CONFIRM)
    names = JITTER_CELLS if jitter else ("R", "D", "M", "Mshift")
    rows = []
    interps = []
    all_cross = {}
    count_lines = []
    surfaces = []
    for seed in seeds:
        a = []
        b = []
        c = []
        lines = []
        surface = []
        for m in (0, 1):
            metrics = {}
            lookup = {}
            cross = {}
            bycount = {}
            curves = {}
            for name in ("frozen", *names):
                folder = study / f"s{seed}/m{m}"
                f = (
                    root / "frozen-record.npz"
                    if name == "frozen"
                    else folder / (("" if jitter else "record-") + name + "-probe.npz")
                )
                anchors = levels("M" if jitter or name == "frozen" else name, seed)
                obs, calc, diag, where, line = one(f, axis, anchors, phi[m], d[:, 2, 1])
                for mode, value in calc.items():
                    metrics[name + "-" + mode] = value
                lookup[name] = diag
                cross[name] = where
                bycount[name] = line
                curves[name] = obs
            a.append(metrics)
            b.append(lookup)
            c.append(cross)
            lines.append(bycount)
            surface.append(curves)

        def avg(items):
            return {
                k: {j: float(np.mean([v[k][j] for v in items])) for j in items[0][k]}
                for k in items[0]
            }

        rows.append(avg(a))
        interps.append(avg(b))
        all_cross[str(seed)] = c
        count_lines.append({k: (lines[0][k] + lines[1][k]) / 2 for k in lines[0]})
        surfaces.append({k: (surface[0][k] + surface[1][k]) / 2 for k in surface[0]})
    absolute = {
        k: {j: summarize([r[k][j] for r in rows]) for j in rows[0][k]} for k in rows[0]
    }
    lookup = {
        k: {j: summarize([r[k][j] for r in interps]) for j in interps[0][k]}
        for k in interps[0]
    }
    ctr = {}
    comparisons = (
        [
            ("jitter", "fixed"),
            ("jitter_gets_fixed", "fixed"),
            ("fixed_gets_jitter", "fixed"),
            ("jitter", "jitter_gets_fixed"),
            ("jitter", "fixed_gets_jitter"),
            ("jitter", "frozen"),
        ]
        if jitter
        else [("M", "R"), ("M", "frozen"), ("Mshift", "M")]
    )
    for l, r in comparisons:
        for mode in ("iid", "unseen_only", "clean"):
            lab = l + "_minus_" + r + "_" + mode
            ctr[lab] = {
                k: summarize(
                    [z[l + "-" + mode][k] - z[r + "-" + mode][k] for z in rows]
                )
                for k in rows[0][l + "-" + mode]
            }
    for name in names:
        for mode in ("iid", "unseen_only", "clean"):
            ctr[name + "_interaction_over_frozen_" + mode] = summarize(
                [
                    z[name + "-" + mode]["q_rescue_minus_doom_interaction"]
                    - z["frozen-" + mode]["q_rescue_minus_doom_interaction"]
                    for z in rows
                ]
            )
    movement = {}
    first, second = ("fixed", "jitter") if jitter else ("M", "Mshift")
    for j, q in enumerate(QS):
        paired = []
        comparable = 0
        valid_seeds = []
        left = []
        right = []
        for s in seeds:
            pair = []
            for world in all_cross[str(s)]:
                u, vv = [world[k][j] for k in (first, second)]

                def single(row):
                    return (
                        not row["starts_above"]
                        and len(row["up"]) == 1
                        and not row["down"]
                    )

                if single(u):
                    left += u["up"]
                if single(vv):
                    right += vv["up"]
                if single(u) and single(vv):
                    comparable += 1
                    pair.append(vv["up"][0] - u["up"][0])
            if pair:
                paired.append(np.mean(pair))
                valid_seeds.append(s)
        movement[str(q)] = {
            "left_usable": len(left),
            "right_usable": len(right),
            "eligible_mirror_pairs": comparable,
            "eligible_seeds": valid_seeds,
            "left_median": float(np.median(left)) if left else None,
            "right_median": float(np.median(right)) if right else None,
            "paired_shift": summarize(paired) if paired else None,
            "n_worlds": 2 * len(seeds),
        }
    write(
        out / "summary.json",
        {
            "seeds": seeds,
            "absolute": absolute,
            "contrasts": ctr,
            "lookup": lookup,
            "crossing_movement": movement,
        },
    )
    write(
        out / "per-seed.json",
        {"seeds": seeds, "values": rows, "lookup": interps, "crossings": all_cross},
    )
    np.savez_compressed(
        out / "responses.npz",
        health=HEALTH,
        q=QS,
        seeds=seeds,
        **{k: np.stack([r[k] for r in surfaces]) for k in surfaces[0]},
        **{"count-" + k: np.stack([r[k] for r in count_lines]) for k in count_lines[0]},
    )
    print(
        "Decoded held-out probabilities and joint lookup for",
        len(seeds),
        "seeds",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "study", "analysis", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--jitter", action="store_true")
    a = p.parse_args()
    run(a.root, a.study, a.analysis, a.output, a.jitter)
