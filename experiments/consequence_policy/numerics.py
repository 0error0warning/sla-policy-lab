"""Factor execution shape, native logits and residual features on fixed checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.coverage import SPLIT_WAVES, WAVES
from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.policy import Policy, deployed, features
from experiments.consequence_policy.study import CONFIRM, TEST_H, TRAIN_H, contexts
from experiments.direct_policy.protocol import rng, write
from experiments.social_closed_loop.core import BodyValue


def summary(values):
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def raw_comparison(a, b):
    hidden_gap = a[0].astype(np.float32) - b[0].astype(np.float32)
    native_gap = np.max(np.abs(deployed(a[1] / 2.0) - deployed(b[1] / 2.0)), axis=-1)
    return {
        "hidden_all_equal": bool(np.array_equal(a[0], b[0])),
        "logits_all_equal": bool(np.array_equal(a[1], b[1])),
        "hidden_rms": float(np.sqrt(np.mean(hidden_gap**2))),
        "native_probability": summary(native_gap),
    }


def load_states(path, index=None):
    with np.load(path) as data:
        a, b = data["hidden"], data["logits"]
    return (a, b) if index is None else (a[index], b[index])


def coverage_singleton(output, rows, native, x, partners):
    body = BodyValue()
    refs = [evaluation_reference(body, rows, m) for m in (0, 1)]
    result = {}
    for prefix, waves, candidate in (
        ("coverage", WAVES, "wide"),
        ("allocation", SPLIT_WAVES, "split"),
    ):
        series = []
        for seed in waves["confirm"]:
            mirrors = []
            for m in (0, 1):
                levels = {}
                for arm in ("high", candidate):
                    actor = Policy.load(output / f"{prefix}-{seed}-m{m}/{arm}.npz")
                    masked = Policy(x.shape[1], blind=True)
                    masked.weights[:, 0] = np.einsum(
                        "b,pbda->pda", actor.gates(TRAIN_H).mean(axis=0), actor.weights
                    )
                    for sensor, model in (("visible", actor), ("masked", masked)):
                        levels[arm, sensor] = metrics(
                            deployed(model.grid_logits(native, x, partners, TEST_H)),
                            refs[m],
                        )
                score = {}
                for sensor in ("visible", "masked"):
                    a, b = levels[candidate, sensor], levels["high", sensor]
                    score[sensor + "_danger_reduction"] = (
                        b["danger_answer"] - a["danger_answer"]
                    )
                    score[sensor + "_healthy_change"] = (
                        a["healthy_answer"] - b["healthy_answer"]
                    )
                    for h in (2.0, 6.0, 10.0):
                        i = TEST_H.index(h)
                        score[sensor + f"_death_reduction_{int(h)}"] = (
                            b["death"][i] - a["death"][i]
                        )
                mirrors.append(score)
            series.append({k: np.mean([a[k] for a in mirrors]) for k in mirrors[0]})
        boot = rng(63995, "coverage-bootstrap").integers(
            len(series), size=(12000, len(series))
        )
        stats = {}
        for k in series[0]:
            v = np.array([r[k] for r in series])
            stats[k] = {
                "mean": float(v.mean()),
                "ci95": np.quantile(v[boot].mean(axis=1), [0.025, 0.975]).tolist(),
                "positive": int((v > 0).sum()),
            }
        danger = stats["visible_danger_reduction"]
        passed = (
            danger["mean"] >= 0.10
            and danger["positive"] >= 16
            and danger["ci95"][0] > 0
            and stats["visible_healthy_change"]["mean"] >= -0.05
        )
        result[prefix] = {
            "seeds": list(waves["confirm"]),
            "stats": stats,
            "passed": passed,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    for arg in ("cache", "gpu", "third", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--ordering", type=Path)
    a = parser.parse_args()
    rows = contexts()
    test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    all_first = load_states(a.cache / "qwen-states.npz")
    all_third = load_states(a.third / "qwen-states.npz")
    all_one = load_states(a.third / "all-singleton.npz")
    first, third, third_one = [
        tuple(arr[test] for arr in z) for z in (all_first, all_third, all_one)
    ]
    second = load_states(a.gpu / "independent-test-prefixes.npz")
    comparisons = {
        "same_batch_across_gpus": raw_comparison(all_first, all_third),
        "same_singleton_across_gpus": raw_comparison(second, third_one),
        "singleton_vs_batch_same_gpu": raw_comparison(all_one, all_third),
    }
    p = np.array([rows[i]["partner"] for i in test])
    with np.load(a.output / "projection.npz") as proj:
        states = [
            (features(z, proj["center"], proj["matrix"]), native)
            for z, native in (first, third, second, third_one)
        ]
    x0, n0 = states[0]
    x1, n1 = states[-1]
    changes = {
        "repeated_batch": [],
        "repeated_singleton": [],
        "native_only": [],
        "features_only": [],
        "both": [],
    }
    worst = {"difference": -1.0}
    arm_changes = {arm: [] for arm in ("real", "shuffle", "blind", "frozen")}

    def prob(actor, z, native):
        return deployed(actor.grid_logits(native, z, p, TEST_H))

    for seed in CONFIRM:
        for mapping in (0, 1):
            for arm in ("real", "shuffle", "blind", "frozen"):
                actor = Policy.load(a.output / f"s{seed}-m{mapping}/{arm}.npz")
                original, whole = prob(actor, x0, n0), prob(actor, x1, n1)
                arm_changes[arm].append(np.abs(original - whole).max(axis=-1).ravel())
                for name, lhs, rhs in (
                    ("repeated_batch", original, prob(actor, *states[1])),
                    ("repeated_singleton", whole, prob(actor, *states[2])),
                    ("native_only", original, prob(actor, x0, n1)),
                    ("features_only", original, prob(actor, x1, n0)),
                    ("both", original, whole),
                ):
                    d = np.abs(lhs - rhs)
                    changes[name].append(d.max(axis=-1).ravel())
                    if name == "both" and d.max() > worst["difference"]:
                        hi, j, action = np.unravel_index(int(np.argmax(d)), d.shape)
                        worst = {
                            "seed": seed,
                            "mapping": mapping,
                            "arm": arm,
                            "index": int(test[j]),
                            "health": TEST_H[hi],
                            "action": int(action),
                            "difference": float(d[hi, j, action]),
                            "batch": lhs[hi, j].tolist(),
                            "singleton": rhs[hi, j].tolist(),
                        }
    paired_lives = []
    with np.load(a.gpu / "end-to-end-lives.npz") as full:
        for r in json.loads((a.gpu / "live-metrics.json").read_bytes()):
            w = r["world"]
            with np.load(
                a.output / "singleton-lives" / f"life-{r['seed']}-traces.npz"
            ) as source:
                local = source[f"w{w[0]}{w[1]}-r0-{r['arm']}"]
            tr = full[r["run"]]
            same = len(local) == len(tr) and np.array_equal(
                local[:, [0, 2, 3]], tr[:, [0, 2, 3]]
            )
            paired_lives.append(
                {
                    "run": r["run"],
                    "turns": len(tr),
                    "exact_choices_responses_contexts": same,
                    "health_max_difference": float(np.abs(tr[:, 4] - local[:, 4]).max())
                    if same
                    else None,
                    "probability_max_difference": float(
                        np.abs(tr[:, 6:10] - local[:, 6:10]).max()
                    )
                    if same
                    else None,
                }
            )
    result = {
        "raw": comparisons,
        "fitted": {k: summary(np.concatenate(v)) for k, v in changes.items()},
        "batch_to_singleton_by_arm": {k: summary(np.concatenate(v)) for k, v in arm_changes.items()},
        "worst": worst,
        "singleton_vs_full_lives": paired_lives,
        "coverage_singleton": coverage_singleton(
            a.output, [rows[i] for i in test], n1, x1, p
        ),
    }
    if a.ordering:
        control = json.loads((a.ordering / "receipt.json").read_bytes())
        control["earlier_sessions"] = {}
        for name in control["conditions"]:
            frozen = load_states(a.ordering / name / "singleton.npz")
            control["earlier_sessions"][name] = {
                "versus_second_singleton_first": raw_comparison(frozen, second),
                "versus_third_batch_first": raw_comparison(frozen, third_one),
            }
        result["same_machine_order_experiment"] = control
    write(a.output / "numerical-factors.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("raw", "fitted", "worst", "singleton_vs_full_lives")
            }
        )
    )


if __name__ == "__main__":
    main()
