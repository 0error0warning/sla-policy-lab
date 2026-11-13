"""Analyse genuine vocabulary-head scores and the paired categorical sampler on CPU."""

import argparse
import csv
import gzip
import io
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.special import softmax

from experiments.consequence_policy.study import EXPLORATION, TEMP, WORDS
from experiments.direct_policy.protocol import settle, write
from experiments.forecast_stages.design import DENSE, VIEWS, shared_joint
from experiments.individual_life.analysis import describe
from experiments.individual_life.design import action_at
from experiments.lifetime_cooperation.analysis import csv_out
from experiments.lifetime_cooperation.design import ITEMS, PILOT

from .design import CASE_ID, CASE_SEED, FOCUS, TWIN_ID

PAIRS = (("oracle", "neutral"), ("learned", "neutral"), ("oracle", "learned"))
CLASSES = ("neither", "equal_shift", "same_sign_unequal", "one_only", "opposite")


def policy(z):
    return (1.0 - EXPLORATION) * softmax(
        np.asarray(z, dtype=float) / TEMP, axis=-1
    ) + EXPLORATION / 4


def boundary(z):
    return float(policy(z)[:3].sum())


def score_terms(delta):
    h, t = map(float, delta[2:])
    if h == 0 and t == 0:
        category = "neither"
    elif h == t:
        category = "equal_shift"
    elif h == 0 or t == 0:
        category = "one_only"
    elif np.sign(h) == np.sign(t):
        category = "same_sign_unequal"
    else:
        category = "opposite"
    return (h + t) / 2, (t - h) / 2, category


def contribution(base, changed):
    """Shapley effects on the help/trade CDF boundary using reconstructed score-level hybrids."""
    base, changed = np.asarray(base, dtype=float), np.asarray(changed, dtype=float)
    dz = changed - base
    c, d, _ = score_terms(dz)
    pieces = np.array([[0, 0, c, c], [0, 0, -d, d], [dz[0], dz[1], 0, 0]])

    def all_permutations(parts):
        n = len(parts)
        hybrids = np.array(
            [
                base
                + sum(
                    (v for j, v in enumerate(parts) if mask & 1 << j), start=np.zeros(4)
                )
                for mask in range(1 << n)
            ]
        )
        bs = policy(hybrids)[:, :3].sum(1)
        result = []
        for order in permutations(range(n)):
            row = np.zeros(n)
            mask = 0
            for j in order:
                nxt = mask | 1 << j
                row[j] = bs[nxt] - bs[mask]
                mask = nxt
            result.append(row)
        values = np.mean(result, axis=0)
        np.testing.assert_allclose(
            values.sum(), boundary(changed) - boundary(base), atol=8e-15, rtol=0
        )
        return 100 * values

    two = all_permutations([pieces[0] + pieces[1], pieces[2]])
    three = all_permutations(pieces)
    return {
        "boundary_from_answers": two[0],
        "boundary_from_refusals": two[1],
        "boundary_from_common": three[0],
        "boundary_from_differential": three[1],
        "boundary_from_two_refusals": three[2],
    }


def group_metrics(rows):
    def vec(k):
        return np.asarray([r[k] for r in rows], dtype=float)

    def mean(k):
        return float(vec(k).mean())

    def mean_abs(k):
        return float(abs(vec(k)).mean())

    def quant(k, level):
        return float(np.quantile(abs(vec(k)), level))

    c = vec("common")
    d = vec("differential")
    ce = float(np.mean(c * c))
    de = float(np.mean(d * d))
    cats = [r["category"] for r in rows]
    result = {
        "n": len(rows),
        "avg_d_help": mean("d_help"),
        "avg_d_trade": mean("d_trade"),
        "mean_abs_d_help": mean_abs("d_help"),
        "mean_abs_d_trade": mean_abs("d_trade"),
        "mean_abs_gap": mean_abs("d_gap"),
        "median_abs_gap": quant("d_gap", 0.5),
        "p90_abs_gap": quant("d_gap", 0.90),
        "max_abs_gap": float(abs(vec("d_gap")).max()),
        "favored_gap": mean("favored_gap"),
        "mean_abs_common": mean_abs("common"),
        "mean_abs_differential": mean_abs("differential"),
        "common_energy": ce,
        "differential_energy": de,
        "common_energy_share": ce / (ce + de) if ce + de else None,
        "gap_unchanged": int(np.sum(vec("d_gap") == 0)),
        "all_four_unchanged": sum(r["all_four_unchanged"] for r in rows),
        "sampled_changed": sum(r["sample_changed"] for r in rows),
        "sampled_exchanged": sum(r["expression_swap"] for r in rows),
        "greedy_changed": sum(r["greedy_changed"] for r in rows),
        "greedy_exchanged": sum(r["greedy_exchanged"] for r in rows),
        "toward_better": sum(r["toward_better"] for r in rows),
        "away_from_better": sum(r["away_from_better"] for r in rows),
        "mean_exact_swap_pp": mean("exact_expression_swap_pp"),
        "mean_hp_abs_gap": mean_abs("hp_d_gap"),
        "hp_gap_on_native_ties": 0.0,
        "mean_abs_d_answer_vs_refusal": mean_abs("d_answer_vs_refusal"),
        "tv_pp": mean("tv_pp"),
        "aggregate_numerically_changed": sum(r["category"] != "neither" for r in rows),
    }
    ties = vec("d_gap") == 0
    moved = np.array([k != "neither" for k in cats])
    result["common_share_among_moved"] = (
        float(np.sum(c[moved] ** 2) / np.sum(c[moved] ** 2 + d[moved] ** 2))
        if moved.any()
        else None
    )
    result["mean_abs_help_if_moved"] = (
        float(abs(vec("d_help")[moved]).mean()) if moved.any() else None
    )
    result["mean_abs_trade_if_moved"] = (
        float(abs(vec("d_trade")[moved]).mean()) if moved.any() else None
    )
    result["mean_abs_gap_if_moved"] = (
        float(abs(vec("d_gap")[moved]).mean()) if moved.any() else None
    )
    if ties.any():
        result["hp_gap_on_native_ties"] = float(abs(vec("hp_d_gap")[ties]).mean())
    else:
        result["hp_gap_on_native_ties"] = None
    result.update({cat: cats.count(cat) for cat in CLASSES})
    return result


def gzip_table(path, items):
    with (
        path.open("wb") as f,
        gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as pack,
        io.TextIOWrapper(pack, encoding="utf8", newline="") as text,
    ):
        w = csv.DictWriter(text, fieldnames=items[0].keys())
        w.writeheader()
        w.writerows(items)


def one_seed(seed, src, source, weight, bias):
    zfile = src / "lives" / f"s{seed}" / "scores.npz"
    data, changes, prior_case = [], [], []
    max_residual = 0.0
    max_round = 0.0
    with np.load(zfile) as z:
        keys = z["keys"]
        idx = z["rows"]
        native = z["scores"].astype(float)
        p = z["prob"].astype(float)
        bits = z["hidden_bits"]
        hidden = (bits.astype(np.uint32) << 16).view(np.float32)
    extra = hidden.astype(float) @ weight.T + bias
    max_residual = float(np.max(abs(native - extra)))
    # BF16 ties-to-even rounding of the CPU matrix projection; accumulated before rounding on a different device.
    fl = extra.astype(np.float32).view(np.uint32)
    rounded = (
        (fl + np.uint32(0x7FFF) + ((fl >> 16) & 1)) & np.uint32(0xFFFF0000)
    ).view(np.float32)
    max_round = float(np.max(abs(rounded.astype(float) - native)))
    uq, noise, reference, q, slots, hashes = (
        source[k]
        for k in ("uniforms", "noise", "previous_prob", "q", "slots", "policy_sha")
    )
    unique = idx[(keys[:, 4] == CASE_ID) & (keys[:, 1] == 31)]
    focus_u = float(uq[unique[0], 0]) if seed == CASE_SEED else None
    for pos, (n, key) in enumerate(zip(idx, keys, strict=True)):
        s, t, w, j, i = map(int, key)
        assert s == seed
        r = ITEMS[i]
        meta = {
            "seed": s,
            "completed": t,
            "wording": ("original", "held")[w],
            "column": j,
            "prompt": i,
            "partner": ("P86", "P93")[r["partner"] - 2],
            "frame": r["frame"],
            "task": r["task"],
            "layout": r["layout"],
        }
        real = q[n, 1]
        fav = int(2 + real[1:].argmax())
        u, v = map(float, uq[n])
        actions = []
        top = []
        for m, view in enumerate(VIEWS):
            logits = native[pos, m]
            hi = extra[pos, m]
            pp = p[pos, m]
            act = action_at(pp, u)
            greedy = int(np.argmax(logits))
            actions.append(act)
            top.append(greedy)
            assert greedy == int(pp.argmax()) or logits[greedy] == logits[pp.argmax()]
            mflag = None if not act else bool(v < real[act - 1])
            after = settle(6.0, act, mflag, noise[n])[0]
            old = reference[n, m]
            if seed == CASE_SEED and t == 31 and i in FOCUS:
                components = old.astype(float) - EXPLORATION / 4
                assert np.all(components > 0)
                prior_case.append(
                    {
                        **meta,
                        "view": view,
                        "u": u,
                        "partner_u": v,
                        "action": WORDS[action_at(old, u)],
                        "top": WORDS[int(old.argmax())],
                        **{
                            "p_" + k: float(val)
                            for k, val in zip(WORDS, old, strict=True)
                        },
                        "lower_help": float(old[:2].astype(float).sum()),
                        "upper_help": float(old[:3].astype(float).sum()),
                        "gap_inferred_from_archived_prob": float(
                            TEMP * np.log(components[3] / components[2])
                        ),
                    }
                )
            A = float(pp[:2].sum())
            B = float(pp[:3].sum())
            row = {
                **meta,
                "view": view,
                "policy_sha": hashes[slots[n]].decode(),
                "health": 6.0,
                "q_ask": q[n, m, 0],
                "q_help": q[n, m, 1],
                "q_trade": q[n, m, 2],
                **{"z_" + k: float(val) for k, val in zip(WORDS, logits, strict=True)},
                "gap_trade_help": logits[3] - logits[2],
                **{"p_" + k: float(val) for k, val in zip(WORDS, pp, strict=True)},
                "u": u,
                "partner_u": v,
                "lower_help": A,
                "upper_help": B,
                "B_minus_u": B - u,
                "action": WORDS[act],
                "top": WORDS[greedy],
                "maintained": mflag,
                "after": after,
                "completed_task": int(act >= 2 and after > 0),
                "choice_at_case_u": WORDS[action_at(pp, focus_u)]
                if seed == CASE_SEED and t == 31
                else "",
                "archive_action": WORDS[action_at(old, u)],
                "probability_max_drift_pp": float(100 * abs(pp - old).max()),
                "high_precision_help": hi[2],
                "high_precision_trade": hi[3],
                "high_precision_gap": hi[3] - hi[2],
                "favored": WORDS[fav],
            }
            data.append(row)
        for target, ref in PAIRS:
            ia, ib = VIEWS.index(target), VIEWS.index(ref)
            za, zb = native[pos, ia], native[pos, ib]
            pa, pb = p[pos, ia], p[pos, ib]
            aa, ab = actions[ia], actions[ib]
            ta, tb = top[ia], top[ib]
            dz = za - zb
            dhp = extra[pos, ia] - extra[pos, ib]
            c, d, kind = score_terms(dz)
            joint = shared_joint(pb[None, :], pa[None, :])[0]
            swapped = aa >= 2 and ab >= 2 and aa != ab
            record = dict(
                **meta,
                target=target,
                reference=ref,
                d_wait=dz[0],
                d_ask=dz[1],
                d_help=dz[2],
                d_trade=dz[3],
                d_gap=dz[3] - dz[2],
                log_odds_change=(dz[3] - dz[2]) / TEMP,
                common=c,
                differential=d,
                category=kind,
                favored_gap=(1 if fav == 3 else -1) * (dz[3] - dz[2]),
                d_answer_vs_refusal=c - dz[:2].mean(),
                tv_pp=50 * abs(pa - pb).sum(),
                d_boundary_pp=100 * (pa[:3].sum() - pb[:3].sum()),
                sample_changed=int(aa != ab),
                expression_swap=int(swapped),
                toward_better=int(swapped and aa == fav),
                away_from_better=int(swapped and ab == fav),
                greedy_changed=int(ta != tb),
                greedy_exchanged=int(ta >= 2 and tb >= 2 and ta != tb),
                exact_expression_swap_pp=100 * (joint[2, 3] + joint[3, 2]),
                all_four_unchanged=int(np.all(dz == 0)),
                hp_d_help=dhp[2],
                hp_d_trade=dhp[3],
                hp_d_gap=dhp[3] - dhp[2],
                u=u,
                base_boundary=pb[:3].sum(),
                target_boundary=pa[:3].sum(),
                base_action=WORDS[ab],
                target_action=WORDS[aa],
            )
            changes.append(record)
    return (
        data,
        changes,
        prior_case,
        {
            "max_native_minus_projection": max_residual,
            "max_native_minus_rounded_projection": max_round,
            "rows": len(idx),
            "native_nonzero": int((native != 0).sum()),
        },
    )


def analyze(src, inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    data, diffs, prior_cases, runs, life, early_time = [], [], [], [], [], []
    with (
        np.load(src / "readout.npz") as w,
        np.load(inputs / "frozen-snapshots.npz") as pack,
    ):
        weight = w["weight"].astype(float)
        bias = w["bias"].astype(float)
        for seed in PILOT:
            r, d, previous_case, precision = one_seed(seed, src, pack, weight, bias)
            data.extend(r)
            diffs.extend(d)
            prior_cases.extend(previous_case)
            runs.append(dict(seed=seed, **precision))
            for word in ("original", "held"):
                for target, ref in PAIRS:
                    sample = [
                        v
                        for v in d
                        if v["completed"] == 8
                        and v["wording"] == word
                        and v["target"] == target
                        and v["reference"] == ref
                    ]
                    life.append(
                        dict(
                            seed=seed,
                            wording=word,
                            target=target,
                            reference=ref,
                            **group_metrics(sample),
                        )
                    )
            for t in range(9):
                for target, ref in PAIRS:
                    sample = [
                        v
                        for v in d
                        if v["completed"] == t
                        and v["wording"] == "held"
                        and v["column"] in DENSE
                        and v["target"] == target
                        and v["reference"] == ref
                    ]
                    early_time.append(
                        dict(
                            seed=seed,
                            completed=t,
                            target=target,
                            reference=ref,
                            **group_metrics(sample),
                        )
                    )
    gzip_table(out / "native_views.csv.gz", data)
    gzip_table(out / "native_contrasts.csv.gz", diffs)
    csv_out(out / "early_lives.csv", life)
    csv_out(out / "early_time.csv", early_time)
    csv_out(out / "readout_precision.csv", runs)
    fc = [
        v
        for v in data
        if v["seed"] == CASE_SEED and v["completed"] == 31 and v["prompt"] in FOCUS
    ]
    cd = [
        v
        for v in diffs
        if v["seed"] == CASE_SEED and v["completed"] == 31 and v["prompt"] in FOCUS
    ]
    group = {
        (v["prompt"], v["view"]): np.array([v["z_" + k] for k in WORDS]) for v in fc
    }
    for v in cd:
        v.update(
            contribution(
                group[v["prompt"], v["reference"]], group[v["prompt"], v["target"]]
            )
        )
    csv_out(out / "case_views.csv", fc)
    csv_out(out / "case_contrasts.csv", cd)
    csv_out(out / "archived_case_boundaries.csv", prior_cases)
    stats = {}
    for word in ("held", "original"):
        for target, ref in PAIRS:
            key = f"{word}.{target}.{ref}"
            sample = [
                v
                for v in diffs
                if v["completed"] == 8
                and v["wording"] == word
                and v["target"] == target
                and v["reference"] == ref
            ]
            seed_rows = [
                v
                for v in life
                if v["wording"] == word
                and v["target"] == target
                and v["reference"] == ref
            ]
            paired = {
                k: describe([z[k] for z in seed_rows])
                for k in (
                    "mean_abs_d_help",
                    "mean_abs_d_trade",
                    "mean_abs_gap",
                    "favored_gap",
                    "mean_abs_common",
                    "mean_abs_differential",
                    "tv_pp",
                )
            }
            stats[key] = {"overall": group_metrics(sample), "seed_uncertainty": paired}
    # Strict equivalence to the already saved choices is reported separately from the linked case's transition.
    main = {
        i: {v["view"]: v["action"] for v in fc if v["prompt"] == i}
        for i in (CASE_ID, TWIN_ID)
    }
    result = {
        "early": stats,
        "cases": main,
        "n_rows": len(data),
        "max_drift_pp": float(max(v["probability_max_drift_pp"] for v in data)),
        "same_archived_choices": sum(v["archive_action"] == v["action"] for v in data),
    }
    write(out / "summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("src", "inputs", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    ans = analyze(**vars(p.parse_args()))
    print(json.dumps(ans["cases"]))
