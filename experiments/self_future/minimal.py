"""Expose one plan at a time using only the model's own inferred effects."""

import json

from experiments.frozen_capability.wire import strict_json

from .staged import messages as original_messages

PHYSICS = """Predict the execution of one plan for a simulated body.
Durability persists. While h > 0, execute the next action a and compute
h_next = max(0, h + signed_change[a]). Use this result as the next h.
There is no cap on durability. A and B each complete one task IF that action
ends above zero. R completes zero tasks. Once h becomes zero, no more actions
execute, including R: there is no revival. The changes supplied below are
this same model's own earlier inference, not an externally calculated future.
Return only JSON with health, terminal_step, tasks, can_act_after_plan.
health is an ARRAY with one entry per listed action: integer resulting h,
0 on the terminating step, null for EVERY step after termination.
terminal_step is the first 1-based step reaching zero, or null.
tasks counts completed A AND B actions; can_act_after_plan is a boolean."""
CHOICE = """Choose between P and Q using the forecasts made by this same model.
These forecasts have not been checked or corrected by an external calculator.
Objective in strict priority order: (1) prefer can_act_after_plan=true;
(2) among ties prefer more completed tasks; (3) among remaining ties prefer
greater final durability. Either exact tie is acceptable.
Return only a JSON object with the single key choice, whose value is P or Q."""


def messages(row, stage, previous):
    def read_model_output(text):
        try:
            return strict_json(text)
        except (ValueError, TypeError):
            return {"unparsed_model_output": text}

    if stage == "effects":
        return original_messages(row, stage, previous)
    if stage in ("P", "Q"):
        source = json.loads(row["messages"][1]["content"])
        data = {
            "current_durability": source["current_durability"],
            "signed_change": read_model_output(previous.get("effects")),
            "actions_to_execute": source["proposed_plans"][stage],
        }
        return [
            {"role": "system", "content": PHYSICS},
            {"role": "user", "content": json.dumps(data)},
        ]
    return [
        {"role": "system", "content": CHOICE},
        {
            "role": "user",
            "content": json.dumps(
                {p: read_model_output(previous.get(p)) for p in ("P", "Q")}
            ),
        },
    ]
