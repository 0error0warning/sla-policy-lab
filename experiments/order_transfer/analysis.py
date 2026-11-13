"""Paired, seed-level contrasts of order training, held orders and healthy work."""

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import ORDERS
from experiments.direct_policy.protocol import rng
from experiments.first_fatal.analysis import resamples
from experiments.first_fatal.design import settle, world
from experiments.health_baseline.analysis import save_csv
from experiments.individual_life.design import action_at, offers
from experiments.order_transfer.design import (
    HEALTH,
    ROLES,
    evaluation_indices,
    item_bank,
    rotations,
)

ALPHA = 0.05 / 8
METRICS = ("chosen", "answer")
PAIRS = (("repeat", "pre"), ("rotate", "pre"), ("rotate", "repeat"))


def load(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
        newline="\n",
    )


def paired_interval(rows, key):
    group = defaultdict(list)
    for row in rows:
        group[(row["wave"], row["seed"])].append(row[key])
    wave_values = [
        [float(np.mean(group[a])) for a in sorted(group) if a[0] == wave]
        for wave in ("pilot", "replication")
    ]
    if not group:
        return {"seeds": 0}
    values = np.array(wave_values[0] + wave_values[1])
    average = values[resamples(tuple(map(len, wave_values)))].mean(axis=1)
    q = np.quantile(average, [ALPHA, 0.025, 0.975, 1 - ALPHA])
    return {
        "seeds": len(values),
        "mean": float(values.mean()),
        "ci95": q[1:3].tolist(),
        "lower_adjusted": float(q[0]),
        "upper_adjusted": float(q[3]),
        "range": [float(values.min()), float(values.max())],
        "pilot": float(np.mean(wave_values[0])) if wave_values[0] else None,
        "replication": float(np.mean(wave_values[1])) if wave_values[1] else None,
    }


def probability_row(c, endpoint, panel, **extra):
    a = c["chosen"]
    probs = np.asarray(panel, dtype=float)
    assert (
        probs.shape == (3, 4)
        and np.all(probs >= 0)
        and np.max(abs(probs.sum(axis=1) - 1)) < 2e-7
    )
    return measurements(
        c,
        endpoint,
        {"chosen": probs[:, a], "answer": probs[:, 2:].sum(axis=1)},
        **extra,
    )


def measurements(c, endpoint, values, **extra):
    v = {
        "seed": c["seed"],
        "birth": c["birth"],
        "wave": c["wave"],
        "endpoint": endpoint,
        "original_action": c["chosen"],
        "signal": c["signal"],
        "next_order_unseen": c["next_context"]["layout"] not in c["train_layouts"],
        **extra,
    }
    for metric, roles in values.items():
        for idx, label in enumerate(ROLES):
            v[label + "_" + metric] = float(roles[idx])
        for left, right in PAIRS:
            v[left + "_minus_" + right + "_" + metric + "_pp"] = 100 * (
                v[left + "_" + metric] - v[right + "_" + metric]
            )
    return v


def summary(rows, metric):
    if not rows:
        return {"cases": 0}
    data = {
        "cases": len(rows),
        "probability_means": {
            r: float(np.mean([v[r + "_" + metric] for v in rows])) for r in ROLES
        },
    }
    for left, right in PAIRS:
        k = left + "_minus_" + right + "_" + metric + "_pp"
        values = np.array([r[k] for r in rows])
        data[left + "_minus_" + right] = {
            **paired_interval(rows, k),
            "case_lower": int((values < 0).sum()),
            "case_tie": int((values == 0).sum()),
            "case_higher": int((values > 0).sum()),
        }
    return data


@lru_cache(maxsize=128)
def environment_block(seed, block):
    # Match the public PR 9 encounter generator without importing the GPU worker.
    from scipy.special import ndtr, ndtri

    g = rng(seed, "first-fatal-arrivals-v1", block)
    layouts = g.permutation(24)
    cells = np.concatenate([g.permutation(np.tile(np.arange(4), 2)) for _ in range(3)])
    u = g.random((24, 2))
    noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * g.random((24, 5)))
    return layouts, cells, u, noise


def verify_healthy(c, items, logs):
    assert len(logs) == 3
    rates = []
    for events in logs:
        h = 350.0
        tasks = 0
        for t, e in enumerate(events):
            block, j = divmod(t, 24)
            layouts, cells, u, noise = environment_block(c["evaluation_seed"], block)
            r = items[e["index"]]
            assert (r["partner"] - 2, r["frame"], r["layout"]) == (
                cells[j] // 2,
                cells[j] % 2,
                layouts[j],
            )
            assert (
                (r["task"], r["wording"]) == (4, 3) and t + 1 == e["turn"] and h >= 90
            )
            assert (
                abs(h - e["health"]) < 1e-10
                and np.max(abs(u[j] - [e["uniform"], e["partner_uniform"]])) < 1e-12
            )
            a = action_at(e["probabilities"], e["uniform"])
            assert a == e["action"]
            prob, _ = offers(r, world(c["seed"]))
            m = bool(u[j, 1] < prob[a]) if a else None
            assert m == e["maintained"]
            h, _, _ = settle(e["health"], a, m, noise[j])
            assert (
                abs(h - e["after"]) < 1e-10
                and h > 0
                and e["completed"] == (a >= 2 and h > 0)
            )
            tasks += e["completed"]
        assert 0 < h < 90
        rates.append(tasks / len(events))
    return np.array(rates)


def summarize_case(c, items, lookup, folder, pre, old, anchor):
    logs = load(folder / "update.json")
    current = load(folder / "anchors.json")
    assert list(rotations(c["original_context"]["layout"])) == c["train_layouts"]
    assert (
        logs["fixed_signal"] == c["signal"] < 0
        and logs["views_per_arm"] == 4
        and logs["updates_per_arm"] == 1
    )
    assert (
        not logs["zero_mean_gradient"]
        and abs(logs["common_importance_ratio"] - c["importance_ratio"]) < 1e-10
    )
    assert (
        abs(
            logs["rotate_multiplier"] * logs["mean_rotation_gradient_norm"]
            + logs["quota"]
        )
        < 1e-9
    )
    with np.load(folder / "post-updates.npz") as z:
        for role in ROLES[1:]:
            data = z[role]
            assert sha256(data.tobytes()).hexdigest() == logs[role + "_hash"]
            size = np.linalg.norm(
                data.view("<f4").astype(float) - pre.view("<f4").astype(float)
            )
            assert abs(size - logs[role + "_step"]) < 1e-9
        assert logs["repeat_matches_archived"] == bool(np.array_equal(old, z["repeat"]))
    assert abs(logs["repeat_step"] - logs["rotate_step"]) < 1e-6
    ix, held_ix = evaluation_indices(c, items, lookup)
    with np.load(folder / "probabilities.npz") as z:
        risk = z["risk"].astype(float)
        good = z["healthy"].astype(float)
        nx = z["successor"].astype(float)
        np.testing.assert_array_equal(z["healthy_indices"], held_ix)
        np.testing.assert_array_equal(z["risk_indices"], ix)
    assert (
        risk.shape == (2, 24, 4, 4)
        and good.shape == (2, 96, 4, 4)
        and nx.shape == (4, 4)
    )
    for a in (risk, good, nx):
        assert (
            np.isfinite(a).all()
            and np.all((a > 0) & (a < 1))
            and np.max(abs(a.sum(axis=-1) - 1)) < 2e-7
        )
    orig = c["original_context"]["layout"]
    src = np.stack((risk[0, orig], risk[1, orig], nx))
    errors = abs(src[:, [0, 3]] - anchor)
    assert int(np.all(errors == 0, axis=(1, 2)).sum()) == current["both_exact_contexts"]
    assert (
        float(errors[:, 0].max()) == current["pre_max"]
        and float(errors[:, 1].max()) == current["archived_post_max"]
    )
    issue = c["next_context"]["layout"]
    if c["checks"][2][0] == ix[issue]:
        np.testing.assert_array_equal(nx, risk[1, issue])
    trained = c["train_layouts"]
    held = [i for i in range(24) if i not in trained]
    assert len(held) == 20
    records = []
    detailed = []
    for j, state in enumerate(("original", "injured")):
        prob = risk[j, :, :3]
        for name, indices in (
            ("all", list(range(24))),
            ("held", held),
            ("trained", trained),
            ("factual_order", [orig]),
        ):
            records.append(
                probability_row(c, state + "_" + name, prob[indices].mean(axis=0))
            )
        for k in range(24):
            detailed.append(
                probability_row(
                    c,
                    state,
                    prob[k],
                    layout=k,
                    in_rotate_training=k in trained,
                    original_order=k == orig,
                    position=ORDERS[k].index(c["chosen"]),
                )
            )
    records.append(probability_row(c, "actual_next", nx[:3]))
    if issue in held:
        records.append(probability_row(c, "actual_next_unseen", nx[:3]))
    else:
        records.append(probability_row(c, "actual_next_seen", nx[:3]))
    for j, h in enumerate(HEALTH):
        records.append(
            probability_row(c, f"healthy_panel_{int(h)}", good[j, :, :3].mean(axis=0))
        )
    records.append(
        probability_row(c, "healthy_panel", good[:, :, :3].mean(axis=(0, 1)))
    )
    lives = load(folder / "healthy-lives.json")
    rates = verify_healthy(c, items, lives)
    case_lives = []
    for role, ev, rate in zip(ROLES, lives, rates, strict=True):
        case_lives.append(
            {
                "seed": c["seed"],
                "birth": c["birth"],
                "wave": c["wave"],
                "evaluation_seed": c["evaluation_seed"],
                "role": role,
                "healthy_encounters": len(ev),
                "tasks_completed": sum(e["completed"] for e in ev),
                "rate": float(rate),
                "final_after": ev[-1]["after"],
            }
        )
    records.append(measurements(c, "healthy_rollout", {"answer": rates}))
    audit = {
        "seed": c["seed"],
        "birth": c["birth"],
        "turn": c["turn"],
        "wave": c["wave"],
        "original_action": c["chosen"],
        "original_layout": orig,
        "next_layout": issue,
        "train_layouts": " ".join(map(str, trained)),
        "successor_seen": issue in trained,
        "original_partner": c["original_context"]["partner"],
        "next_partner": c["next_context"]["partner"],
        "changed_text": c["original_context"]["text"] != c["next_context"]["text"],
        "gradient_cosine": logs["gradient_cosine"],
        "quota": logs["quota"],
        "single_step": logs["repeat_step"],
        "multi_step": logs["rotate_step"],
        "step_size_error": abs(logs["repeat_step"] - logs["rotate_step"]),
        "pre_hash": c["pre_hash"],
        "historical_match": logs["repeat_matches_archived"],
        "gradient_views": 8,
        **current,
    }
    return records, detailed, case_lives, audit


def analyze(inputs, output, dest):
    cases = load(inputs / "order-cases.json")
    receipt = load(output / "receipt.json")
    assert (
        len(cases) == 48
        and receipt["total_optimizer_steps"] == 96
        and receipt["frozen_backbone"]
    )
    assert len(list((output / "cases").glob("s*/*/probabilities.npz"))) == 48
    items, lookup = item_bank()
    records = []
    details = []
    lives = []
    budgets = []
    with (
        np.load(inputs / "order-weights.npz") as z,
        np.load(inputs / "order-anchors.npz") as za,
    ):
        for j, c in enumerate(cases):
            pair = (z[c["pre_hash"]], z[c["historical_post_hash"]])
            result = summarize_case(
                c,
                items,
                lookup,
                output / "cases" / f"s{c['seed']}" / c["birth"],
                *pair,
                za["probabilities"][j],
            )
            records.extend(result[0])
            details.extend(result[1])
            lives.extend(result[2])
            budgets.append(result[3])
    dest.mkdir(exist_ok=True, parents=True)
    save_csv(dest / "paired-cases.csv", records)
    save_csv(dest / "per-order.csv", details)
    save_csv(dest / "healthy-lives.csv", lives)
    save_csv(dest / "budgets.csv", budgets)
    endpoints = sorted({r["endpoint"] for r in records})
    groups = {}
    for sub in ("all", "blank", "population", "pilot", "replication"):
        selected = [r for r in records if sub in ("all", r["birth"], r["wave"])]
        groups[sub] = {}
        for k in endpoints:
            subset = [r for r in selected if r["endpoint"] == k]
            groups[sub][k] = {
                met: summary(subset, met)
                for met in (("answer",) if k == "healthy_rollout" else METRICS)
            }
    main = groups["all"]
    criteria = {}
    for endpoint in ("injured_held", "actual_next"):
        for m in METRICS:
            s = main[endpoint][m]
            criteria[endpoint + "_" + m] = (
                s["rotate_minus_repeat"]["upper_adjusted"] < 0
                and s["rotate_minus_pre"]["mean"] < 0
            )
    for endpoint in ("healthy_panel", "healthy_rollout"):
        for other in ("pre", "repeat"):
            criteria[endpoint + "_vs_" + other] = (
                main[endpoint]["answer"]["rotate_minus_" + other]["lower_adjusted"]
                > -2.0
            )
    assert len(criteria) == 8
    audit = {
        "independent_seeds": len({c["seed"] for c in cases}),
        "cases": len(cases),
        "unseen_actual_next": sum(not r["successor_seen"] for r in budgets),
        "risk_order_measurements": len(details),
        "healthy_panel_case_inputs": 48 * 2 * 96,
        "healthy_rollout_decisions": sum(r["healthy_encounters"] for r in lives),
        "healthy_rollouts": len(lives),
        "identical_reference_updates": sum(r["historical_match"] for r in budgets),
        "exact_archived_contexts": sum(r["both_exact_contexts"] for r in budgets),
        "max_step_size_error": max(r["step_size_error"] for r in budgets),
        "min_gradient_cosine": min(r["gradient_cosine"] for r in budgets),
        "mean_gradient_cosine": float(np.mean([r["gradient_cosine"] for r in budgets])),
        "reference_cpu_physics": True,
    }
    write_json(
        dest / "summary.json",
        {
            "groups": groups,
            "criteria": criteria,
            "joint": all(criteria.values()),
            "audit": audit,
            "gpu": receipt,
        },
    )
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("inputs", "output", "dest"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    print(analyze(a.inputs, a.output, a.dest))
