"""A paired immediate learning effect is measured within each request, post minus pre."""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import EXPLORATION, TEMP, WORDS
from experiments.first_fatal.analysis import resamples
from experiments.health_baseline.analysis import save_csv
from experiments.individual_life.design import action_at
from experiments.snapshot_transfer.inputs import CONTEXTS

PRIMARY = CONTEXTS[:3]
MEASURES = ("chosen", "answer")
CONDITIONS = ("correct", "tied", "opposite")


def direction(signed_delta):
    if signed_delta == 0:
        return "tied"
    return "correct" if signed_delta > 0 else "opposite"


def effects(panel, chosen, signal):
    """Fixed semantic IDs even when the request or displayed option order changes."""
    pre, post = np.asarray(panel, dtype=float)
    assert pre.shape == post.shape == (4,) and chosen in (2, 3)
    out = {}
    for name, index in (
        ("chosen", [chosen]),
        ("answer", [2, 3]),
        ("sibling", [5 - chosen]),
    ):
        a, b = float(sum(pre[index])), float(sum(post[index]))
        delta = 100 * (b - a)
        out[name + "_before"] = a
        out[name + "_after"] = b
        out[name + "_pp"] = delta
        out[name + "_aligned_pp"] = float(np.sign(signal)) * delta
        out[name + "_direction"] = (
            direction(out[name + "_aligned_pp"]) if signal else "zero_signal"
        )
    assert abs(out["answer_pp"] - out["chosen_pp"] - out["sibling_pp"]) < 1e-10
    return out


def interval(items, key):
    per_seed = defaultdict(list)
    for v in items:
        per_seed[v["wave"], v["seed"]].append(v[key])
    groups = []
    for wave in ("pilot", "replication"):
        groups.append(
            [float(np.mean(per_seed[k])) for k in sorted(per_seed) if k[0] == wave]
        )
    values = np.array(groups[0] + groups[1])
    assert len(values)
    boot = values[resamples(tuple(map(len, groups)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "seeds": len(values),
        "pilot": float(np.mean(groups[0])) if groups[0] else None,
        "replication": float(np.mean(groups[1])) if groups[1] else None,
    }


def basic(items, measure):
    if not items:
        return {"n": 0}
    d = np.array([r[measure + "_pp"] for r in items])
    active = d[d != 0]
    key = measure + "_direction"
    counts = Counter(r[key] for r in items)
    return {
        "n": len(items),
        "down": int((d < 0).sum()),
        "tie": int((d == 0).sum()),
        "up": int((d > 0).sum()),
        **{k: counts[k] for k in CONDITIONS},
        "mean_before": float(np.mean([r[measure + "_before"] for r in items])),
        "mean_after": float(np.mean([r[measure + "_after"] for r in items])),
        "mean_pp": float(d.mean()),
        "mean_abs_pp": float(abs(d).mean()),
        "median_abs_pp": float(np.median(abs(d))),
        "mean_among_nonzero_pp": float(active.mean()) if len(active) else None,
        "range_pp": [float(d.min()), float(d.max())],
    }


def transitions(records, start, end, measure):
    return {
        s: {
            t: sum(
                l[start][measure + "_direction"] == s
                and l[end][measure + "_direction"] == t
                for l in records
            )
            for t in CONDITIONS
        }
        for s in CONDITIONS
    }


def as_policy(logits):
    x = np.asarray(logits, dtype=float) / TEMP
    v = np.exp(x - x.max(axis=-1, keepdims=True))
    return (1 - EXPLORATION) * v / v.sum(axis=-1, keepdims=True) + EXPLORATION / 4


def analyze(inputs, output, dest):
    config = json.loads((inputs / "transfer-cases.json").read_bytes())
    cases = config["cases"]
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert (
        receipt["training_steps"] == 0
        and receipt["backbone_unchanged"]
        and receipt["ordinary_forwards"] == 432
    )
    precision = json.loads((output / "precision.json").read_bytes())
    with np.load(output / "probabilities.npz") as z:
        primary, repeat = z["primary"], z["repeat"]
    with np.load(inputs / "transfer-anchors.npz") as z:
        anchor_a = abs(primary[:, 0, 2] - z["original_before"]).max(axis=1)
        anchor_c = abs(primary[:, 2] - z["successor"]).max(axis=(1, 2))
    er = abs(primary[:, :3] - repeat).max(axis=(2, 3))
    assert (
        int((er == 0).sum()) == receipt["repeat_exact"]
        and int((anchor_c == 0).sum()) == receipt["successor_exact"]
    )
    assert int((anchor_a == 0).sum()) == receipt["original_exact"]
    assert np.all((primary > 0) & (primary < 1))
    assert np.max(abs(primary.astype(float).sum(axis=-1) - 1)) < 2e-7
    np.testing.assert_array_equal(primary[:, :, 1], primary[:, :, 5])
    dest.mkdir(exist_ok=True, parents=True)
    detailed = []
    per_case = []
    layers = []
    roundrows = []
    for n, c in enumerate(cases):
        orig, next_row = c["original_context"], c["next_context"]
        meta = {
            k: c[k]
            for k in (
                "seed",
                "birth",
                "turn",
                "wave",
                "signal",
                "chosen",
                "parameter_step",
                "rul_loss",
                "maintenance",
            )
        }
        meta.update(
            chosen_word=WORDS[c["chosen"]],
            health=c["checks"][0][1],
            after=c["checks"][1][1],
            original_u=c["original_u"],
            next_u=c["next_u"],
            factual_next=c["next_factual_action"],
            partner_changed=orig["partner"] != next_row["partner"],
            text_changed=orig["text"] != next_row["text"],
            order_changed=orig["layout"] != next_row["layout"],
        )
        current = {}
        case_row = dict(
            meta,
            anchor_original_max=float(anchor_a[n]),
            anchor_successor_max=float(anchor_c[n]),
            repeat_max=float(er[n].max()),
        )
        seen = {}
        for j, (context, (ix, h)) in enumerate(zip(CONTEXTS, c["checks"], strict=True)):
            full = primary[n, j]
            if (ix, h) in seen:
                np.testing.assert_array_equal(full, seen[ix, h])
            seen[ix, h] = full
            pair = full[[2, 3]]
            u = c["next_u"] if context == "successor" else c["original_u"]
            item = {
                **meta,
                "context": context,
                "request_index": ix,
                "evaluation_health": h,
                "uniform": u,
                **effects(pair, c["chosen"], c["signal"]),
            }
            for side, probs in zip(("before", "after"), pair, strict=True):
                item.update(
                    {
                        side + "_" + w: float(v)
                        for w, v in zip(WORDS, probs, strict=True)
                    }
                )
                item["draw_" + side] = action_at(probs, u)
                item["greedy_" + side] = int(np.argmax(probs))
            current[context] = item
            detailed.append(item)
            if j < len(PRIMARY):
                r = precision[n][j]
                r0 = {
                    k: v
                    for k, v in r.items()
                    if k not in ("fp32_head", "native_logits")
                }
                native_logits = np.asarray(r["native_logits"])
                assert np.max(abs(as_policy(native_logits) - pair)) < 2e-7
                aux = effects(r["fp32_head"], c["chosen"], c["signal"])
                layers.append(
                    {
                        **meta,
                        "context": context,
                        **r0,
                        "native_scores_equal": bool(np.array_equal(*native_logits)),
                        "native_full_distribution_equal": bool(np.array_equal(*pair)),
                        "chosen_tied": item["chosen_pp"] == 0,
                        "answer_tied": item["answer_pp"] == 0,
                        "fp32_head_chosen_pp": aux["chosen_pp"],
                        "fp32_head_answer_pp": aux["answer_pp"],
                        "fp32_head_chosen_direction": aux["chosen_direction"],
                        "fp32_head_answer_direction": aux["answer_direction"],
                    }
                )
                for m in MEASURES:
                    for k in ("before", "after", "pp", "direction"):
                        case_row[context + "_" + m + "_" + k] = item[m + "_" + k]
        results = {k: v for k, v in current.items() if k in PRIMARY}
        for a, b in itertools.pairwise(PRIMARY):
            for m in MEASURES:
                case_row[b + "_minus_" + a + "_" + m + "_aligned_pp"] = (
                    current[b][m + "_aligned_pp"] - current[a][m + "_aligned_pp"]
                )
        per_case.append(results)
        roundrows.append(case_row)
    summary = {}
    groups = {
        "all": lambda r: True,
        "blank": lambda r: r["birth"] == "blank",
        "population": lambda r: r["birth"] == "population",
        "positive": lambda r: r["signal"] > 0,
        "negative": lambda r: r["signal"] < 0,
        "pilot": lambda r: r["wave"] == "pilot",
        "replication": lambda r: r["wave"] == "replication",
    }
    for label, condition in groups.items():
        block = [v for v in per_case if condition(v["original"])]
        main = {}
        for context in PRIMARY:
            entries = [r[context] for r in block]
            main[context] = {m: basic(entries, m) for m in MEASURES}
            if entries:
                for m in MEASURES:
                    main[context][m]["aligned_seed_pp"] = interval(
                        entries, m + "_aligned_pp"
                    )
                main[context]["semantic_draw_flips"] = sum(
                    r["draw_after"] != r["draw_before"] for r in entries
                )
                main[context]["answer_class_flips"] = sum(
                    (r["draw_after"] >= 2) != (r["draw_before"] >= 2) for r in entries
                )
        changes = {}
        for a, b in (
            ("original", "health_only"),
            ("health_only", "successor"),
            ("original", "successor"),
        ):
            data = {}
            for m in MEASURES:
                items = [
                    {
                        **v[a],
                        "change": v[b][m + "_aligned_pp"] - v[a][m + "_aligned_pp"],
                    }
                    for v in block
                ]
                data[m] = {
                    "transitions": transitions(block, a, b, m),
                    "aligned_difference_pp": interval(items, "change")
                    if items
                    else None,
                }
            changes[a + "_to_" + b] = data
        summary[label] = {"n": len(block), "main": main, "paired_changes": changes}
    swaps = {}
    for context, flag in (
        ("partner_only", "partner_changed"),
        ("request_only", "text_changed"),
        ("order_only", "order_changed"),
    ):
        pairs = [
            (a, b)
            for a, b in zip(
                [v["health_only"] for v in per_case],
                [v for v in detailed if v["context"] == context],
                strict=True,
            )
            if b[flag]
        ]
        swaps[context] = {}
        for m in MEASURES:
            rr = [
                {**b, "change": b[m + "_aligned_pp"] - a[m + "_aligned_pp"]}
                for a, b in pairs
            ]
            swaps[context][m] = {
                "eligible": len(rr),
                "effect": basic(rr, m),
                "aligned_change_vs_health_only_pp": interval(rr, "change")
                if rr
                else None,
            }
    layer_summary = {}
    for context in PRIMARY:
        subset = [v for v in layers if v["context"] == context]
        info = {
            k: sum(r[k + "_changed"] > 0 for r in subset)
            for k in (
                "fp32_correction",
                "native_down",
                "native_residual",
                "native_hidden",
            )
        }
        info.update(
            native_scores_different=sum(not r["native_scores_equal"] for r in subset),
            full_distributions_different=sum(
                not r["native_full_distribution_equal"] for r in subset
            ),
        )
        for m in MEASURES:
            info[m + "_native_ties"] = sum(r[m + "_tied"] for r in subset)
            info[m + "_ties_with_aux_correct"] = sum(
                r[m + "_tied"] and r["fp32_head_" + m + "_direction"] == "correct"
                for r in subset
            )
            info[m + "_ties_with_aux_nonzero"] = sum(
                r[m + "_tied"] and r["fp32_head_" + m + "_pp"] != 0 for r in subset
            )
            info[m + "_aux_mean_pp"] = float(
                np.mean([r["fp32_head_" + m + "_pp"] for r in subset])
            )
        layer_summary[context] = info
    directions = Counter((r["birth"], r["signal"] > 0) for r in roundrows)
    result = {
        "cases": len(cases),
        "independent_seeds": len({r["seed"] for r in roundrows}),
        "groups": summary,
        "components": swaps,
        "precision": layer_summary,
        "numerics": receipt,
        "signal_counts": {
            b: {
                "positive": directions[b, True],
                "negative": sum(r["birth"] == b and r["signal"] < 0 for r in roundrows),
            }
            for b in ("blank", "population")
        },
    }
    save_csv(dest / "contexts.csv", detailed)
    save_csv(dest / "paired-updates.csv", roundrows)
    save_csv(dest / "precision.csv", layers)
    (dest / "summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
        newline="\n",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("inputs", "output", "dest"):
        parser.add_argument("--" + k, type=Path, required=True)
    a = parser.parse_args()
    analyze(a.inputs, a.output, a.dest)
