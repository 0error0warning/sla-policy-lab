"""Score verifiable result sets independently of final choice and arithmetic."""

import math

from experiments.frozen_capability.design import WORDS
from experiments.frozen_capability.design import score as old_score

from .design import targets


def word_set(value):
    if (
        not isinstance(value, list)
        or not all(isinstance(v, str) and v in WORDS for v in value)
        or len(value) != len(set(value))
    ):
        return None
    return set(value)


def score(obj, key):
    obj = obj if isinstance(obj, dict) else {}
    choice = obj.get("choice")
    valid = isinstance(choice, str) and choice in WORDS
    answer = obj.get("task_answer")
    answering = valid and choice in ("help", "trade")
    task_ok = (
        (isinstance(answer, str) and answer.strip().lower().rstrip(".") == "january")
        if answering
        else answer is None and "task_answer" in obj
    )
    low = word_set(obj.get("min_risk_actions"))
    task = word_set(obj.get("task_actions_at_min_risk"))
    local_targets = None
    if key["arm"] == "body" and isinstance(obj.get("actions"), dict):
        risks = {
            w: obj["actions"].get(w, {}).get("p_death")
            if isinstance(obj["actions"].get(w), dict)
            else None
            for w in WORDS
        }
        if all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
            for v in risks.values()
        ):
            local_targets = targets(risks)
    elif key["arm"] == "table":
        local_targets = key
    expected_keys = {"choice", "task_answer"}
    if key["form"] == "sets":
        expected_keys |= {"min_risk_actions", "task_actions_at_min_risk"}
    if key["arm"] == "body":
        expected_keys.add("actions")
    result = {
        "choice_valid": valid,
        "choice_correct": valid and choice in key["accepted_choices"],
        "answering": answering,
        "task_answer_correct": bool(task_ok),
        "completed_task": answering and task_ok,
        "actual_risk": key["risks"][choice] if valid else None,
        "risk_regret": key["risks"][choice] - min(key["risks"].values())
        if valid
        else None,
        "min_set_valid": low is not None,
        "task_set_valid": task is not None,
        "min_set_correct": low == set(key["min_risk_actions"])
        if low is not None
        else False,
        "task_set_correct": task == set(key["task_actions_at_min_risk"])
        if task is not None
        else False,
        "intersection_consistent": task == low.intersection({"help", "trade"})
        if low is not None and task is not None
        else None,
        "choice_consistent_with_sets": choice in (task or low)
        if valid and low is not None and task is not None and low
        else None,
        "sets_consistent_with_own_risks": low == set(local_targets["min_risk_actions"])
        and task == set(local_targets["task_actions_at_min_risk"])
        if local_targets is not None and low is not None and task is not None
        else None,
        "choice_consistent_with_own_risks": choice in local_targets["accepted_choices"]
        if valid and local_targets
        else None,
        "top_schema": set(obj) == expected_keys,
    }
    if key["arm"] == "body":
        result["risk_predictions_correct"] = local_targets is not None and all(
            abs(obj["actions"][word]["p_death"] - key["risks"][word]) <= 0.01
            for word in WORDS
        )
        projected = {}
        expected = {}
        action_valid = isinstance(obj.get("actions"), dict) and set(
            obj["actions"]
        ) == set(WORDS)
        after_action_ok = True
        for w in WORDS:
            row = (
                obj.get("actions", {}).get(w)
                if isinstance(obj.get("actions"), dict)
                else None
            )
            if not isinstance(row, dict) or set(row) != set(key["actions"][w]):
                action_valid = False
            value = row.get("after_action") if isinstance(row, dict) else None
            if not (
                type(value) in (int, float)
                and math.isfinite(value)
                and 0 <= value <= 350
            ):
                after_action_ok = False
                action_valid = False
            elif abs(value - key["actions"][w]["after_action"]) > 0.02 or (
                value > 0
            ) != (key["actions"][w]["after_action"] > 0):
                after_action_ok = False
            projected[w] = (
                {n: v for n, v in row.items() if n != "after_action"}
                if isinstance(row, dict)
                else row
            )
            expected[w] = {
                n: v for n, v in key["actions"][w].items() if n != "after_action"
            }
        old = old_score(
            # Only use the old numeric validator here. Choice validity is scored
            # independently above, so a bad action label cannot hide good math.
            {"actions": projected, "choice": "wait", "task_answer": None},
            {**key, "actions": expected},
        )
        result.update(
            {
                "numeric_schema": action_valid and old["schema"],
                "after_action_correct": after_action_ok,
                "endpoints_correct": old["calculation_correct"],
                "calculation_correct": action_valid
                and after_action_ok
                and old["calculation_correct"],
                "max_health_error": old.get("max_health_error"),
                "max_risk_error": old.get("max_risk_error"),
                "survival_classification_correct": old.get(
                    "survival_classification_correct", False
                ),
            }
        )
    result["protocol_valid"] = result["top_schema"] and valid and bool(task_ok)
    if key["form"] == "sets":
        result["protocol_valid"] &= low is not None and task is not None
    if key["arm"] == "body":
        result["protocol_valid"] &= result["numeric_schema"]
    result["joint_success"] = result["protocol_valid"] and result["choice_correct"]
    if key["form"] == "sets":
        result["joint_success"] &= (
            result["min_set_correct"] and result["task_set_correct"]
        )
    if key["arm"] == "body":
        result["joint_success"] &= result["calculation_correct"]
    return result
