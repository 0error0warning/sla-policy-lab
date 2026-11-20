"""Check the archived GPU conclusions using prefix sums and literal model JSON.

No model, production simulator, scorer, or token replay is invoked here.
Run token replay separately; this audits arithmetic, choices and published counts.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from zipfile import ZipFile


def read(text):
    def unique(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate key")
            out[k] = v
        return out

    return json.loads(text, object_pairs_hook=unique)


def forecast(case, actions):
    sums = list(accumulate(case["effects"][a] for a in actions))
    raw = [case["current"] + delta for delta in sums]
    stop = next((i for i, h in enumerate(raw) if h <= 0), None)
    return {
        "health": [
            h if stop is None or i < stop else 0 if i == stop else None
            for i, h in enumerate(raw)
        ],
        "terminal_step": None if stop is None else stop + 1,
        "tasks": sum(
            a in ("A", "B") and (stop is None or i < stop)
            for i, a in enumerate(actions)
        ),
        "can_act_after_plan": stop is None,
    }


def choices(plans):
    values = {
        p: (
            row["can_act_after_plan"],
            row["tasks"],
            row["health"][-1] if row["can_act_after_plan"] else 0,
        )
        for p, row in plans.items()
    }
    best = max(values.values())
    return sorted(p for p, value in values.items() if value == best)


def decoded(record):
    if record["completion_status"] != "completed":
        return None
    if "stages" not in record:
        try:
            return read(record["final_text"])
        except (ValueError, TypeError):
            return None
    obj = {"plans": {}, "choice": None}
    for p in ("P", "Q"):
        try:
            obj["plans"][p] = read(record["stages"][p])
        except (ValueError, TypeError, KeyError):
            obj["plans"][p] = None
    try:
        obj["choice"] = read(record["stages"]["choice"])["choice"]
    except (ValueError, TypeError, KeyError):
        pass
    return obj


def exact(actual, expected):
    """JSON booleans must not compare equal to integer counts."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def check(root):
    base = root / "artifacts/model-studies/2026-11-15"
    keys = read((base / "self-future/inputs/evaluator-only-key.json").read_bytes())
    for key in keys.values():
        gold = {p: forecast(key["case"], a) for p, a in key["case"]["plans"].items()}
        assert exact(gold, key["plans"])
        assert choices(gold) == key["accepted_choices"]
    results, originals, minimal, diagnoses = {}, {}, {}, {}
    for version in ("v1", "v2", "v3"):
        folder = base / "self-future-run" / version
        manifest = read((folder / "MANIFEST.json").read_bytes())
        for name, spec in manifest.items():
            data = (folder / name).read_bytes()
            assert len(data) == spec["bytes"]
            assert hashlib.sha256(data).hexdigest() == spec["sha256"]
        models = ("4B", "9B") if version == "v3" else ("4B",)
        with ZipFile(folder / "results.zip") as archive:
            for model in models:
                name = f"analysis-{model}.json" if version == "v3" else "analysis.json"
                analysis = read((folder / name).read_bytes())
                groups = defaultdict(lambda: defaultdict(int))
                rows = []
                for name in sorted(archive.namelist()):
                    if not name.startswith(f"output/{model}/records/"):
                        continue
                    row = read(archive.read(name))
                    if row["split"] != "startup":
                        rows.append(row)
                if version == "v3":
                    minimal[model] = {r["id"]: r for r in rows}
                    diagnosis = defaultdict(int)
                for row in rows:
                    arm = row.get("arm", "monolithic")
                    key = keys[row["id"]]
                    scored = (
                        analysis["scored"] if version == "v1" else analysis["arms"][arm]
                    )["cases"][row["id"]]
                    obj = decoded(row)
                    chosen = obj.get("choice") if isinstance(obj, dict) else None
                    correct = chosen in key["accepted_choices"]
                    joint = bool(
                        isinstance(obj, dict)
                        and set(obj) == {"plans", "choice"}
                        and exact(obj["plans"], key["plans"])
                        and correct
                    )
                    assert joint == scored["joint_success"]
                    assert correct == (scored["choice_correct"] is True)
                    group = f"{arm}/{key['case']['split']}/{key['information']}"
                    counts = groups[group]
                    counts["n"] += 1
                    counts["completed"] += row["completion_status"] == "completed"
                    counts["joint_success"] += joint
                    counts["choice_correct"] += correct
                    counts["correct_choice_wrong_forecast"] += correct and not joint
                    if chosen in ("P", "Q"):
                        outcome = key["plans"][chosen]
                        counts["chosen_plan_terminates"] += not outcome[
                            "can_act_after_plan"
                        ]
                        counts["chosen_plan_tasks"] += outcome["tasks"]
                    if version == "v1":
                        originals[row["id"]] = row["output_ids"]
                    if version == "v2" and arm == "monolithic":
                        assert originals[row["id"]] == row["output_ids"]
                    if version == "v3" and isinstance(obj, dict):
                        for p, truth in key["plans"].items():
                            prediction = obj.get("plans", {}).get(p)
                            diagnosis["completed_plan_outputs"] += 1
                            if not isinstance(prediction, dict):
                                continue
                            diagnosis["all_plan_fields_correct"] += exact(
                                prediction, truth
                            )
                            trace = prediction.get("health")
                            diagnosis["health_array_length_wrong"] += not isinstance(
                                trace, list
                            ) or len(trace) != len(truth["health"])
                            diagnosis["health_array_exact"] += exact(
                                trace, truth["health"]
                            )
                            diagnosis["task_count_correct"] += exact(
                                prediction.get("tasks"), truth["tasks"]
                            )
                            diagnosis["terminal_and_continuation_correct"] += exact(
                                prediction.get("terminal_step"), truth["terminal_step"]
                            ) and exact(
                                prediction.get("can_act_after_plan"),
                                truth["can_act_after_plan"],
                            )
                results[f"{version}-{model}"] = dict(groups)
                if version == "v3":
                    diagnoses[model] = dict(diagnosis)
    shared = [
        cid
        for cid in minimal["4B"]
        if all(minimal[m][cid]["completion_status"] == "completed" for m in minimal)
    ]
    shared_results = {}
    healthy = {}
    for model, rows in minimal.items():
        shared_results[model] = {
            "correct_choices": sum(
                decoded(rows[cid])["choice"] in keys[cid]["accepted_choices"]
                for cid in shared
            ),
            "selected_terminal_plans": sum(
                not keys[cid]["plans"][decoded(rows[cid])["choice"]][
                    "can_act_after_plan"
                ]
                for cid in shared
            ),
        }
        healthy_ids = [
            cid
            for cid, k in keys.items()
            if k["case"]["family"] == "health" and k["case"]["variant"] == 2
        ]
        healthy[model] = {
            "planned": len(healthy_ids),
            "completed": sum(
                rows[cid]["completion_status"] == "completed" for cid in healthy_ids
            ),
            "correct_choices": sum(
                isinstance(decoded(rows[cid]), dict)
                and decoded(rows[cid])["choice"] in keys[cid]["accepted_choices"]
                for cid in healthy_ids
            ),
        }
    return {
        "answer_keys_prefix_sum_verified": len(keys),
        "v1_v2_monolithic_exact_output_repeats": len(originals),
        "archived_primary_choice_and_joint_counts_agree": True,
        "counts": results,
        "v3_posthoc_plan_field_diagnostics": diagnoses,
        "v3_both_completed_posthoc": {
            "n": len(shared),
            "ids": shared,
            "models": shared_results,
        },
        "v3_healthy_choice_delivery": healthy,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.repo)
    with args.out.open("x", encoding="utf8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
