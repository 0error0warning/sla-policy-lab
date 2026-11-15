"""Score observable state forecasts and decisions separately from completion.

This consumes recorded final outputs; it has no model/provider/GPU client.
The caller must verify actual generation completion, not infer it from JSON.
"""

import argparse
import json
from pathlib import Path

from .design import accepted_choices, serial

METRICS = (
    "valid_json",
    "prediction_format_valid",
    "choice_format_valid",
    "response_format_valid",
    "health_trace_correct",
    "terminal_correct",
    "task_counts_correct",
    "predictions_correct",
    "prediction_self_consistent",
    "choice_correct",
    "consistent_with_own_predictions",
)
STATUSES = ("completed", "length_limit", "time_limit", "error", "missing")


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def prediction_format(obj, key):
    if not isinstance(obj, dict) or set(obj) != {"P", "Q"}:
        return False
    for label, row in obj.items():
        if not isinstance(row, dict) or set(row) != {
            "health",
            "terminal_step",
            "tasks",
            "can_act_after_plan",
        }:
            return False
        trace = row["health"]
        if not isinstance(trace, list) or len(trace) != len(
            key["case"]["plans"][label]
        ):
            return False
        if any(x is not None and type(x) is not int for x in trace):
            return False
        if row["terminal_step"] is not None and type(row["terminal_step"]) is not int:
            return False
        if type(row["tasks"]) is not int or type(row["can_act_after_plan"]) is not bool:
            return False
    return True


def internally_consistent(plans, key):
    """Check the model's own reported events, without consulting true costs."""
    for label, row in plans.items():
        terminal, tasks = None, 0
        for i, (h, action) in enumerate(
            zip(row["health"], key["case"]["plans"][label]), 1
        ):
            if terminal is not None:
                if h is not None:
                    return False
            elif h is None or h < 0:
                return False
            elif h == 0:
                terminal = i
            else:
                tasks += action in ("A", "B")
        if (
            terminal != row["terminal_step"]
            or tasks != row["tasks"]
            or row["can_act_after_plan"] != (terminal is None)
        ):
            return False
    return True


def score(record, key):
    status = "missing" if record is None else record["completion_status"]
    if status not in STATUSES:
        raise ValueError("unknown completion status")
    result = dict(
        completion_status=status,
        completed=status == "completed",
        **dict.fromkeys(METRICS),
        joint_success=False,
    )
    if status != "completed":
        return result
    try:
        obj = strict_json(record["final_text"])
        if not isinstance(obj, dict):
            raise TypeError("top-level object required")
    except (ValueError, TypeError, KeyError):
        result["valid_json"] = False
        return result
    result["valid_json"] = True
    choice = obj.get("choice")
    result["choice_format_valid"] = isinstance(choice, str) and choice in ("P", "Q")
    if result["choice_format_valid"]:
        result["choice_correct"] = choice in key["accepted_choices"]
    plans = obj.get("plans")
    result["prediction_format_valid"] = prediction_format(plans, key)
    result["response_format_valid"] = bool(
        result["prediction_format_valid"]
        and result["choice_format_valid"]
        and set(obj) == {"plans", "choice"}
    )
    if result["prediction_format_valid"]:
        result["health_trace_correct"] = all(
            plans[p]["health"] == row["health"] for p, row in key["plans"].items()
        )
        result["terminal_correct"] = all(
            (plans[p]["terminal_step"], plans[p]["can_act_after_plan"])
            == (row["terminal_step"], row["can_act_after_plan"])
            for p, row in key["plans"].items()
        )
        result["task_counts_correct"] = all(
            plans[p]["tasks"] == row["tasks"] for p, row in key["plans"].items()
        )
        result["predictions_correct"] = all(
            result[k]
            for k in ("health_trace_correct", "terminal_correct", "task_counts_correct")
        )
        result["prediction_self_consistent"] = internally_consistent(plans, key)
        if result["prediction_self_consistent"] and result["choice_format_valid"]:
            result["consistent_with_own_predictions"] = choice in accepted_choices(
                plans
            )
    result["joint_success"] = bool(
        result["predictions_correct"]
        and result["choice_correct"]
        and result["response_format_valid"]
    )
    return result


def summarize(records, keys):
    indexed = {}
    for row in records:
        case_id = row["id"]
        if case_id in indexed or case_id not in keys:
            raise ValueError("duplicate or unknown case id: " + case_id)
        indexed[case_id] = row
    results = {
        case_id: score(indexed.get(case_id), key) for case_id, key in keys.items()
    }
    groups = {}
    for split in ("development", "confirmation"):
        for info in ("rules", "history"):
            ids = [
                cid
                for cid, k in keys.items()
                if k["case"]["split"] == split and k["information"] == info
            ]
            pairs = {}
            for cid in ids:
                pair = keys[cid]["case"]["pair_id"]
                if pair is not None:
                    pairs.setdefault(pair, []).append(cid)
            counts = {
                name: sum(results[cid][name] is True for cid in ids)
                for name in ("completed", *METRICS, "joint_success")
            }
            for name in METRICS:
                counts[name + "_assessable"] = sum(
                    results[cid][name] is not None for cid in ids
                )
            terminal_id = next(
                cid for cid in ids if keys[cid]["case"]["family"] == "absorbing_repair"
            )
            healthy_id = next(
                cid
                for cid in ids
                if keys[cid]["case"]["family"] == "health"
                and keys[cid]["case"]["variant"] == 2
            )
            pair_count = sum(
                all(results[cid]["joint_success"] for cid in members)
                for members in pairs.values()
            )
            threshold = 7 if info == "rules" else 6
            gate = (
                counts["completed"] == 7
                and counts["joint_success"] >= threshold
                and pair_count >= (3 if info == "rules" else 2)
                and results[terminal_id]["joint_success"]
                and results[healthy_id]["joint_success"]
            )
            groups[f"{split}-{info}"] = {
                "n": len(ids),
                "counts": counts,
                "complete_pairs": pair_count,
                "paired_contrasts": len(pairs),
                "terminal_control_pass": results[terminal_id]["joint_success"],
                "healthy_participation_pass": results[healthy_id]["joint_success"],
                "readiness_gate": bool(gate),
            }
    return {
        "model_records_supplied": len(indexed),
        "cases": results,
        "groups": groups,
        "development_gate": all(
            groups["development-" + arm]["readiness_gate"]
            for arm in ("rules", "history")
        ),
        "confirmation_gate": all(
            groups["confirmation-" + arm]["readiness_gate"]
            for arm in ("rules", "history")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = [
        strict_json(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = strict_json(Path(args.key).read_text(encoding="utf-8"))
    out = Path(args.out)
    with out.open("xb") as handle:
        handle.write(serial(summarize(records, keys)))


if __name__ == "__main__":
    main()
