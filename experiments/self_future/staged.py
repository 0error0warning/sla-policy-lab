"""Pure message construction and literal assembly; contains no body oracle."""

import json

from experiments.frozen_capability.wire import strict_json

STAGE_CAPS = {"effects": 128, "P": 384, "Q": 384, "choice": 128}


def messages(row, stage, previous):
    common = row["messages"][0]["content"].split("Return only a JSON object", 1)[0]
    instruction = {
        "effects": "Return only a JSON object mapping A, B, R to their signed fixed changes in durability. Use known_effects if provided; otherwise infer them from the observations. Do not predict plans or choose yet.",
        "P": "Predict ONLY plan P. Return only an object with health, terminal_step, tasks, can_act_after_plan, using the definitions below. Do not choose a plan yet.",
        "Q": "Predict ONLY plan Q. Return only an object with health, terminal_step, tasks, can_act_after_plan, using the definitions below. Do not choose a plan yet.",
        "choice": "Choose P or Q under the declared lexicographic objective. Return only an object with the single key choice. Your earlier forecasts below are model outputs, not externally verified answers.",
    }[stage]
    if stage in ("P", "Q"):
        instruction += "\nhealth has one entry per proposed action: resulting integer durability for an executed action, 0 on the terminating action, null for ALL subsequent actions that never execute. terminal_step is the 1-based first terminating step or null. tasks counts both A and B that end above zero; R counts zero. can_act_after_plan is a boolean."
    evidence = ""
    if previous:
        evidence = (
            "\nEarlier outputs by this same model (not externally checked):\n"
            + json.dumps(previous)
        )
    return [
        {"role": "system", "content": common + instruction},
        {"role": "user", "content": row["messages"][1]["content"] + evidence},
    ]


def assemble(outputs):
    """Copy model fields literally; never repair numbers or select an action."""
    try:
        return {
            "plans": {p: strict_json(outputs[p]) for p in ("P", "Q")},
            "choice": strict_json(outputs["choice"])["choice"],
        }
    except (ValueError, TypeError, KeyError):
        return {
            "assembly_error": "model_stage_not_valid_json",
            "stage_outputs": outputs,
        }
