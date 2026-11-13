"""Independent CPU verification of archived histories, probabilities, parameter chains and causal contrasts."""

import argparse
import json
from pathlib import Path
from hashlib import sha256
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.individual_life.design import offers
from .analysis import analyze_replay
from .design import (
    ITEMS,
    LEVELS,
    PANEL,
    PILOT,
    ROLES,
    WORLDS,
    leave_one_out,
    tape_inputs,
)


def digest(x):
    return sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def eq(a, b, atol=1e-7):
    np.testing.assert_allclose(a, b, rtol=0.0, atol=atol)


def calc(panel, seed, world, split, partner=None, held_task=None):
    """Explicit semantic sums over cells independent of the vectorized scoring path."""
    from experiments.lifetime_cooperation.design import rules

    bits = rules(seed, world, 100)
    results = {x: [] for x in ("social", "death", "cooperation", "healthy")}
    for j, idx in enumerate(PANEL[split]):
        r = ITEMS[idx]
        if (partner and r["partner"] != partner) or (
            held_task and r["task"] != held_task
        ):
            continue
        chance, fav = offers(r, bits)
        for k, h in enumerate(LEVELS):
            q = panel[split, k, j].astype(float)
            if h in (6, 90):
                results["social"].append(q[fav] - q[5 - fav])
            if h == 2:
                results["death"].append(q[0] + 0.85 * q[1] + q[2] + q[3])
            if h == 6:
                results["cooperation"].append(q[2] * chance[2] + q[3] * chance[3])
            if h >= 90:
                results["healthy"].append(q[2] + q[3])
    return {k: 100 * float(np.mean(v)) for k, v in results.items()}


def hs(st, ch, seed, task=None):
    # P86 changes rules; using the stable sign for both histories is intentional.
    from experiments.lifetime_cooperation.design import rules

    s = []
    l = []
    for k in (1, 2):
        for j, i in enumerate(PANEL[1]):
            r = ITEMS[i]
            if task is not None and r["task"] != task:
                continue
            x, y = st[1, k, j].astype(float), ch[1, k, j].astype(float)
            diff = 50 * ((x[2] - x[3]) - (y[2] - y[3]))
            if r["partner"] == 2:
                _, f = offers(r, rules(seed, "stable", 100))
                s.append(diff if f == 2 else -diff)
            else:
                l.append(abs(diff))
    return np.mean(s), np.mean(l)


def check_replay(root, source, dest):
    gate = analyze_replay(root, dest)
    receipt = json.loads((root / "receipt.json").read_bytes())
    assert receipt["frozen_backbone"] and receipt["mode"] == "sameresid"
    orig = tape_inputs(source)
    checked = 0
    chain_count = 0
    max_restore = 0.0
    all_drift = []
    all_native = []
    final_metrics = {}
    causal_rows = []
    params = {}
    with np.load(root / "steering" / "donors.npz") as z:
        donor = {s: z[f"s{s}"] for s in PILOT}
    for seed in PILOT:
        two = []
        prefix = None
        originals = []
        hashes24 = []
        for world in WORLDS:
            d = root / "histories" / f"s{seed}" / world
            data = np.load(d / "final-panel.npy")
            b0 = np.load(d / "birth-panel.npy")
            assert data.shape == (6, 2, 4, 32, 4) and np.array_equal(data[0], b0)
            assert np.isfinite(data).all() and data.min() > 0
            eq(data.sum(-1), np.ones(data.shape[:-1]), 3e-6)
            two.append(data)
            log = list(map(json.loads, (d / "updates.jsonl").read_bytes().splitlines()))
            src = orig[seed, world]
            checked += len(log)
            meta = json.loads((d / "receipt.json").read_bytes())
            all_drift.append(meta["birth_max_wrt_archive"])
            all_native.append(meta["native_max_wrt_archive"])
            assert len(log) == len(src) == meta["turns"] and meta["updates"] == 5 * len(
                src
            )
            with (
                np.load(d / "birth.npz") as b,
                np.load(d / "at24.npz") as a,
                np.load(d / "last-living.npz") as end,
            ):
                chain = {role: digest(b[role]) for role in ROLES[1:]}
                params = {role: int(b[role].size) for role in ROLES[1:]}
                eq(b["loreft"], b["loreft_wrong"], 0)
                eq(b["native_lora"], b["native_wrong"], 0)
                eq(b["loreft"][:5120], b["loreft"][5120:10240], 0)
                eq(b["loreft"][10240:], np.zeros(2), 0)
                eq(b["residual_lora"][5120:], np.zeros(5120), 0)
                for obj in (b, a, end):
                    for role in ("loreft", "loreft_wrong"):
                        r = obj[role][:5120].reshape(2, 2560).astype(float)
                        eq(r @ r.T, np.eye(2), 2e-5)
                for n, (event, e) in enumerate(zip(log, src, strict=True)):
                    assert all(
                        event[k] == e[k]
                        for k in (
                            "turn",
                            "row",
                            "health",
                            "after",
                            "action",
                            "wrong_action",
                            "signal",
                            "reward",
                            "maintenance",
                        )
                    )
                    assert event["origin_behavior"] == e["prob"][0] and e["after"] > 0
                    probs = np.array(event["probabilities"])
                    eq(probs.sum(-1), np.ones(6), 3e-6)
                    if n == 0:
                        eq(probs, np.broadcast_to(probs[0], probs.shape), 0)
                    for slot, role in enumerate(ROLES[1:], 1):
                        step = event["updates"][role]
                        label = (
                            e["wrong_action"] if role.endswith("wrong") else e["action"]
                        )
                        p = probs[slot, label]
                        origp = e["prob"][0][label]
                        eq(
                            [step["importance"], step["current"], step["behavior"]],
                            [np.clip(p / origp, 0.5, 2.0), p, origp],
                            1e-9,
                        )
                        grad = step["gradient_norm"]
                        actual = min(
                            0.4,
                            abs(
                                0.18
                                * e["signal"]
                                * np.clip(p / origp, 0.5, 2.0)
                                / (0.01 + grad * grad)
                            )
                            * grad,
                        )
                        eq(actual, step["nominal_l2"], 2e-8)
                        assert (
                            step["before"] == chain[role]
                            and step["actual_l2"] >= 0
                            and step["orthogonality_error"] <= 2e-5
                        )
                        chain[role] = step["after"]
                        chain_count += 1
                        if n == 23:
                            assert digest(a[role]) == chain[role]
                assert all(chain[role] == digest(end[role]) for role in ROLES[1:])
                hashes24.append({role: digest(a[role]) for role in ROLES[1:]})
            if prefix is None:
                prefix = log[:24]
            else:
                assert prefix == log[:24] and hashes24[0] == hashes24[1]
            originals.append(b0)
        assert np.array_equal(originals[0], originals[1])
        for ref in ("birth", "loreft_wrong", "native_lora", "residual_lora"):
            a = ROLES.index(ref)
            for i, name in enumerate(("source", "held")):
                left = calc(two[0][3], seed, "stable", i)
                right = calc(two[0][a], seed, "stable", i)
                for k in left:
                    final_metrics.setdefault(name + "_" + k + "_vs_" + ref, []).append(
                        left[k] - right[k]
                    )
        ss, ll = hs(two[0][3], two[1][3], seed)
        for k, v in (
            ("adapt_S", ss),
            ("other_spill_L", ll),
            ("adapt_S_minus_L", ss - ll),
        ):
            final_metrics.setdefault(k, []).append(v)
        d = root / "steering" / f"s{seed}"
        with np.load(d / "directions.npz") as z:
            v, alpha, r = leave_one_out(donor, seed)
            eq(v, z["unit"], 3e-13)
            eq(r, z["controls"], 3e-13)
            eq(alpha, z["alpha"], 3e-13)
            assert seed not in z["seeds"] and set(z["seeds"]) == set(PILOT) - {seed}
            if alpha > 0:
                eq(np.linalg.norm(v), 1.0, 1e-10)
                eq(r @ v, [0.0, 0.0], 1e-10)
            eq(np.linalg.norm(r, axis=1), [1.0, 1.0], 1e-10)
        mat = np.load(d / "probabilities.npy").astype(float)
        assert (
            mat.shape == (2, 12, 2, 4, 32, 4)
            and np.isfinite(mat).all()
            and mat.min() > 0
        )
        eq(mat.sum(axis=-1), np.ones(mat.shape[:-1]), 3e-6)
        for wi in range(2):
            eq(mat[wi, 0], two[wi][3], 0)
        max_restore = max(max_restore, float(abs(mat[:, 0] - mat[:, 5]).max()))
        svals = [hs(mat[0, j], mat[1, j], seed, task=5)[0] for j in range(12)]
        ext = []
        for jplus, jminus in ((1, 2), (6, 7), (9, 10)):
            dif = []
            for wi in range(2):
                for hh in (1, 2):
                    for j, idx in enumerate(PANEL[1]):
                        if ITEMS[idx]["partner"] == 2 and ITEMS[idx]["task"] == 5:
                            x = mat[wi, jplus, 1, hh, j] - mat[wi, jminus, 1, hh, j]
                            dif.append(50 * (x[2] - x[3]))
            ext.append(float(np.mean(dif)))
        causal_rows.append(
            [
                ext[0],
                ext[0] - (abs(ext[1]) + abs(ext[2])) / 2,
                svals[0],
                svals[0] - svals[3],
                svals[0]
                - svals[3]
                - (abs(svals[0] - svals[8]) + abs(svals[0] - svals[11])) / 2,
                svals[0] - svals[4],
            ]
        )
    summary = gate["behavior"]["effects_pp"]
    for k, v in final_metrics.items():
        eq(np.mean(v), summary[k]["mean"], 1e-6)
    for k, vals in zip(
        (
            "positive_minus_negative_half",
            "external_over_random",
            "native_S",
            "remove_drop",
            "remove_over_random",
            "swap_drop",
        ),
        np.array(causal_rows).T,
        strict=True,
    ):
        eq(np.mean(vals), gate["steering"]["effects"][k]["mean"], 1e-6)
    a = {
        "independent_seeds": len(PILOT),
        "matched_histories": len(orig),
        "nonterminal_replay_events": checked,
        "private_updates_with_verified_chain": chain_count,
        "branch_prefix_events": 24,
        "learned_parameters_per_arm": params,
        "max_restore_probability_error": max_restore,
        "max_archived_birth_drift": max(all_drift),
        "max_archived_native_trajectory_drift": max(all_native),
        "semantic_panel_metrics_recomputed": len(final_metrics),
        "target_seeds_excluded_from_directions": len(PILOT),
        "max_on_device_probability_error": receipt["max_probability_error"],
        "max_on_device_gradient_error": receipt["max_gradient_error"],
    }
    write(dest / "audit.json", a)
    return a


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(check_replay(a.root, a.source, a.out), indent=2))
