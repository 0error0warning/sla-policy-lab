"""Check true post-update weights and summarize transfer by partner, wording and healthy task."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.health_baseline.analysis import save_csv
from experiments.individual_life.design import action_at
from experiments.order_transfer.analysis import (
    load,
    paired_interval,
    verify_healthy,
    write_json,
)
from experiments.order_transfer.design import HEALTH, rotations
from experiments.paraphrase_transfer.design import (
    MEASURES,
    ROLES,
    UNSEEN,
    WORDS,
    bank,
    grid,
    training_views,
)

ALPHA = 0.05 / 12
DANGER = (
    "held_new_wording",
    "actual_next",
    "actual_next_same_partner",
    "actual_next_changed_partner",
)
SAFE = ("healthy_panel", "healthy_rollout")
PAIRS = (("original", "pre"), ("augmented", "pre"), ("augmented", "original"))


def measures(c, endpoint, values, **kwargs):
    orig, nxt = c["original_context"], c["next_context"]
    row = {
        "seed": c["seed"],
        "birth": c["birth"],
        "wave": c["wave"],
        "turn": c["turn"],
        "action": c["chosen"],
        "signal": c["signal"],
        "endpoint": endpoint,
        "partner_changed": orig["partner"] != nxt["partner"],
        "frame_changed": orig["frame"] != nxt["frame"],
        "original_frame": orig["frame"],
        **kwargs,
    }
    for m, v in values.items():
        for i, role in enumerate(ROLES):
            row[role + "_" + m] = float(v[i])
        for left, right in PAIRS:
            row[left + "_minus_" + right + "_" + m + "_pp"] = 100 * (
                row[left + "_" + m] - row[right + "_" + m]
            )
    return row


def probabilities(c, endpoint, probs, **kwargs):
    p = np.asarray(probs, dtype=float)
    assert p.shape == (3, 4) and np.max(abs(p.sum(axis=1) - 1)) < 2e-7
    return measures(
        c,
        endpoint,
        {"chosen": p[:, c["chosen"]], "answer": p[:, 2:].sum(axis=1)},
        **kwargs,
    )


def describe(rows, metric):
    if not rows:
        return {"cases": 0, "seeds": 0}
    result = {
        "cases": len(rows),
        "seeds": len({r["seed"] for r in rows}),
        "probability_case_means": {
            arm: float(np.mean([r[arm + "_" + metric] for r in rows])) for arm in ROLES
        },
    }
    for left, right in PAIRS:
        key = left + "_minus_" + right + "_" + metric + "_pp"
        vals = np.array([r[key] for r in rows])
        result[left + "_minus_" + right] = {
            **paired_interval(rows, key, alpha=ALPHA),
            "case_lower": int((vals < 0).sum()),
            "case_same": int((vals == 0).sum()),
            "case_higher": int((vals > 0).sum()),
            "case_weighted_mean": float(vals.mean()),
        }
    return result


def audit_step(c, folder, lookup, pre, old):
    recorded = load(folder / "update.json")
    assert (
        recorded["signed_feedback"] == c["signal"] < 0
        and recorded["quota"] == c["radius"]
    )
    assert (
        recorded["factual_importance"] == c["importance_ratio"]
        and recorded["updates_per_arm"] == 1
        and recorded["views_per_arm"] == 16
    )
    assert c["train_layouts"] == list(rotations(c["original_context"]["layout"]))
    with np.load(folder / "post-updates.npz") as updates:
        for arm in ROLES[1:]:
            line = recorded[arm]
            assert line["indices"] == training_views(
                c["original_context"], lookup, c["train_layouts"], arm == "augmented"
            )
            assert (
                len(line["view_propensities"]) == 16
                and min(line["view_propensities"]) > 0
            )
            assert (
                not line["zero_gradient"]
                and abs(line["scale"] * line["gradient_norm"] + c["radius"]) < 1e-10
            )
            bits = updates[arm]
            assert sha256(bits.tobytes()).hexdigest() == line["post_hash"]
            distance = float(
                np.linalg.norm(
                    bits.view("<f4").astype(float) - pre.view("<f4").astype(float)
                )
            )
            assert (
                abs(distance - line["actual_l2"]) < 1e-9
                and abs(distance - c["radius"]) < 1e-6
            )
        assert (
            bool(np.array_equal(updates["original"], old))
            == recorded["original_matches_pr15"]
        )
    delta = abs(recorded["original"]["actual_l2"] - recorded["augmented"]["actual_l2"])
    assert delta < 1e-6
    return {
        "seed": c["seed"],
        "birth": c["birth"],
        "factual_order": c["original_context"]["layout"],
        "next_order": c["next_context"]["layout"],
        "turn": c["turn"],
        "signal": c["signal"],
        "pre_hash": c["pre_hash"],
        "pr15_hash": c["pr15_hash"],
        "original_equals_pr15": recorded["original_matches_pr15"],
        "cosine": recorded["cosine"],
        "original_l2": recorded["original"]["actual_l2"],
        "augmented_l2": recorded["augmented"]["actual_l2"],
        "l2_difference": delta,
        "quota": c["radius"],
    }


def case_rows(c, items, lookup, folder, pre, old, anchor):
    aud = audit_step(c, folder, lookup, pre, old)
    orig, nxt = c["original_context"], c["next_context"]
    own_ix = grid(orig, lookup, orig["partner"])
    swap_ix = grid(orig, lookup, 5 - orig["partner"])[4:]
    healthy_ix = [i for i, r in enumerate(items) if (r["task"], r["wording"]) == (5, 4)]
    with np.load(folder / "probabilities.npz") as z:
        raw = z["own"].astype(float)
        op = z["original"].astype(float)
        other = z["other"].astype(float)
        nx = z["successor"].astype(float)
        good = z["healthy"].astype(float)
        for k, v in (
            ("own_ix", own_ix),
            ("other_ix", swap_ix),
            ("healthy_ix", healthy_ix),
        ):
            np.testing.assert_array_equal(z[k], v)
    assert (
        raw.shape == (8, 24, 4, 4)
        and other.shape == (4, 24, 4, 4)
        and good.shape == (2, 96, 4, 4)
    )
    assert nx.shape == op.shape == (4, 4)
    for q in (raw, op, other, nx, good):
        assert (
            np.isfinite(q).all()
            and q.min() > 0
            and q.max() < 1
            and np.max(abs(q.sum(axis=-1) - 1)) < 2e-7
        )
    o = orig["layout"]
    anchors = load(folder / "anchors.json")
    err = abs(np.stack((op, raw[0, o], nx))[:, [0, 3]] - anchor)
    assert int(np.all(err == 0, axis=(1, 2)).sum()) == anchors["identical_contexts"]
    assert max(anchors["max_pre"], anchors["max_pr15"]) == float(err.max())
    if aud["original_equals_pr15"]:
        assert all(
            np.array_equal(v[..., 1, :], v[..., 3, :])
            for v in (raw, op, other, nx, good)
        )
    aud.update(anchors)
    if orig["partner"] == nxt["partner"] and orig["frame"] == nxt["frame"]:
        np.testing.assert_array_equal(nx, raw[0, nxt["layout"]])
    raw, other, good = (x[..., :3, :] for x in (raw, other, good))
    trained = c["train_layouts"]
    held_order = [k for k in range(24) if k not in trained]
    result = []
    detail = []
    slices = {
        "original_pre_health": op[:3],
        "original_text_injured": raw[0].mean(axis=0),
        "training_rewrites_only": raw[1:4].mean(axis=(0, 1)),
        "held_new_wording": raw[4:].mean(axis=(0, 1)),
        "held_word_trained_orders": raw[4:, trained].mean(axis=(0, 1)),
        "held_word_held_orders": raw[4:, held_order].mean(axis=(0, 1)),
        "held_word_other_partner": other.mean(axis=(0, 1)),
        "healthy_panel": good.mean(axis=(0, 1)),
    }
    for k, prob in slices.items():
        result.append(probabilities(c, k, prob))
    for j in range(4):
        result.append(
            probabilities(c, "held_template_" + str(j + 1), raw[4 + j].mean(axis=0))
        )
    for j, h in enumerate(HEALTH):
        result.append(probabilities(c, f"healthy_panel_{int(h)}", good[j].mean(axis=0)))
    a = nx[:3]
    semantic = [action_at(v, c["next_u"]) for v in a]
    tags = {role + "_draw": d for role, d in zip(ROLES, semantic, strict=True)}
    tags.update(
        {
            role + "_draws_answer": int(d >= 2)
            for role, d in zip(ROLES, semantic, strict=True)
        }
    )
    partner = "changed_partner" if orig["partner"] != nxt["partner"] else "same_partner"
    frame = "changed_frame" if orig["frame"] != nxt["frame"] else "same_frame"
    for k in (
        "actual_next",
        "actual_next_" + partner,
        "actual_next_" + partner + "_" + frame,
    ):
        result.append(probabilities(c, k, a, **tags))
    for rel, ids, values in (("source", own_ix, raw), ("other", swap_ix, other)):
        for j, (word, row) in enumerate(
            zip(WORDS if rel == "source" else UNSEEN, ids, strict=True)
        ):
            for order, ix in enumerate(row):
                detail.append(
                    probabilities(
                        c,
                        "wording_at_injured_health",
                        values[j, order],
                        partner_condition=rel,
                        wording_id=word,
                        frame=orig["frame"],
                        layout=order,
                        order_used=order in trained,
                        text_index=ix,
                        phrase=items[ix]["text"],
                        held_wording=word in UNSEEN,
                    )
                )
    lives = load(folder / "healthy-lives.json")
    rates = verify_healthy(c, items, lives)
    healthy = []
    for k, (role, ev) in enumerate(zip(ROLES, lives, strict=True)):
        healthy.append(
            {
                "seed": c["seed"],
                "birth": c["birth"],
                "wave": c["wave"],
                "evaluation_seed": c["evaluation_seed"],
                "role": role,
                "healthy_encounters": len(ev),
                "tasks": sum(e["completed"] for e in ev),
                "rate": rates[k],
                "final_health": ev[-1]["after"],
            }
        )
    concord = []
    for j, (actual, parent) in enumerate(
        zip(lives[:2], c["healthy_reference"], strict=True)
    ):
        exact = len(actual) == len(parent) and all(
            all(e[k] == v for k, v in saved.items()) for e, saved in zip(actual, parent)
        )
        concord.append(exact)
        if j == 0 or aud["original_equals_pr15"]:
            assert exact
    aud["healthy_pre_matches_parent"], aud["healthy_original_matches_parent"] = concord
    result.append(measures(c, "healthy_rollout", {"answer": rates}))
    return result, detail, healthy, aud


def analyze(inputs, output, dest):
    cases = load(inputs / "phrase-cases.json")
    receipt = load(output / "receipt.json")
    assert (
        receipt["frozen_backbone"]
        and receipt["total_steps"] == 96
        and receipt["cases"] == 48
    )
    assert len(list((output / "cases").glob("s*/*/probabilities.npz"))) == 48
    items, lookup = bank(inputs / "wording.json")
    rows = []
    detail = []
    lives = []
    budgets = []
    with (
        np.load(inputs / "phrase-weights.npz") as weights,
        np.load(inputs / "phrase-anchors.npz") as archive,
    ):
        for i, c in enumerate(cases):
            folder = output / "cases" / f"s{c['seed']}" / c["birth"]
            a, b, d, e = case_rows(
                c,
                items,
                lookup,
                folder,
                weights[c["pre_hash"]],
                weights[c["pr15_hash"]],
                archive["probabilities"][i],
            )
            rows.extend(a)
            detail.extend(b)
            lives.extend(d)
            budgets.append(e)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("paired-cases", rows),
        ("wording-layouts", detail),
        ("healthy-lives", lives),
        ("budgets", budgets),
    ):
        save_csv(dest / (name + ".csv"), data)
    groups = {}
    endpoints = sorted({r["endpoint"] for r in rows})
    for group in ("all", "blank", "population", "pilot", "replication"):
        group_rows = [r for r in rows if group in ("all", r["birth"], r["wave"])]
        groups[group] = {
            e: {
                m: describe([r for r in group_rows if r["endpoint"] == e], m)
                for m in (("answer",) if e == "healthy_rollout" else MEASURES)
            }
            for e in endpoints
        }
    main = groups["all"]
    criteria = {}
    for end in DANGER:
        for m in MEASURES:
            d = main[end][m]
            criteria[end + "_" + m] = (
                d["augmented_minus_original"]["upper_adjusted"] < 0
                and d["augmented_minus_pre"]["mean"] < 0
            )
    for end in SAFE:
        for role in ("pre", "original"):
            criteria[end + "_vs_" + role] = (
                main[end]["answer"]["augmented_minus_" + role]["lower_adjusted"] > -2
            )
    healthy_ok = all(
        criteria[k + "_vs_" + role] for k in SAFE for role in ("pre", "original")
    )
    comparison = {
        m: main["actual_next"][m]["augmented_minus_original"]["mean"] for m in MEASURES
    }
    assert len(criteria) == 12
    draws = {}
    for end in DANGER[1:]:
        rec = [r for r in rows if r["endpoint"] == end]
        draws[end] = {
            "n": len(rec),
            "seeds": len({r["seed"] for r in rec}),
            "pre_answers": sum(r["pre_draws_answer"] for r in rec),
            "original_answers": sum(r["original_draws_answer"] for r in rec),
            "augmented_answers": sum(r["augmented_draws_answer"] for r in rec),
            "changed_semantic": sum(
                r["original_draw"] != r["augmented_draw"] for r in rec
            ),
            "changed_answer_class": sum(
                r["original_draws_answer"] != r["augmented_draws_answer"] for r in rec
            ),
        }
    audit = {
        "cases": len(cases),
        "environment_seeds": len({c["seed"] for c in cases}),
        "held_panel_case_inputs": 48 * 4 * 24,
        "swapped_panel_case_inputs": 48 * 4 * 24,
        "healthy_panel_case_inputs": 48 * 2 * 96,
        "wording_layout_rows": len(detail),
        "healthy_lives": len(lives),
        "healthy_decisions": sum(r["healthy_encounters"] for r in lives),
        "fixed_controls_from_pr15": sum(r["original_equals_pr15"] for r in budgets),
        "pr15_exact_anchor_contexts": sum(r["identical_contexts"] for r in budgets),
        "max_step_difference": max(r["l2_difference"] for r in budgets),
        "parent_healthy_paths_exact": sum(
            int(r["healthy_pre_matches_parent"])
            + int(r["healthy_original_matches_parent"])
            for r in budgets
        ),
        "average_gradient_cosine": float(np.mean([r["cosine"] for r in budgets])),
    }
    result = {
        "groups": groups,
        "criteria": criteria,
        "joint": all(criteria.values()),
        "practical_gate": healthy_ok and all(v <= -1 for v in comparison.values()),
        "same_draw_actions": draws,
        "audit": audit,
        "gpu": receipt,
    }
    write_json(dest / "summary.json", result)
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("inputs", "output", "dest"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(analyze(a.inputs, a.output, a.dest))
