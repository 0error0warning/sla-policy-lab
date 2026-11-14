"""Frozen interface diagnosis: explicit sets and a small complete rule matrix."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.frozen_capability.design import (
    COSTS,
    DESCRIPTIONS,
    MODELS,
    RULES,
    WORDS,
    outcomes,
    serial,
)

OBJECTIVE = """Select an action using these priorities in this exact order:
1. Find the smallest death probability in the table.
2. Keep ALL actions with exactly that smallest probability.
3. Among the kept actions, keep those whose completes_task value is true.
4. If step 3 is nonempty, choose one of them. Otherwise choose any action from step 2.
Do not use unlisted objectives. Any remaining tie is acceptable.
The partner's question is: What is the first month of the year?
When your chosen action completes the task, task_answer must be "January";
otherwise task_answer must be null.
Return one JSON object in the final answer, with no Markdown or explanation."""
SETS = """Also return min_risk_actions (the full set from step 2) and
task_actions_at_min_risk (the full set from step 3), as arrays of action words.
These are verifiable results, not a request for a written explanation."""
TABLES = {
    "all_zero": [0.0, 0.0, 0.0, 0.0],
    "partial_zero_tie": [0.0, 0.4, 0.0, 0.4],
    "lowest_answer": [1.0, 0.85, 0.9, 0.1],
    "lowest_nonanswer": [0.4, 0.1, 0.2, 0.2],
}
BODY = ((2.0, "help"), (6.0, "trade"), (8.25, "help"), (90.0, "trade"))


def targets(risks):
    minimum = min(risks.values())
    low = [w for w in WORDS if abs(risks[w] - minimum) < 1e-12]
    task = [w for w in low if w in ("help", "trade")]
    return {
        "min_risk_actions": low,
        "task_actions_at_min_risk": task,
        "accepted_choices": task or low,
    }


def fixtures():
    prompts, keys = [], {}
    for i, (name, values) in enumerate(TABLES.items()):
        risks = dict(zip(WORDS, values))
        truth = targets(risks)
        for flip in (0, 1):
            order = [(i + j) % 4 for j in range(4)]
            if flip:
                order.reverse()
            table = [
                {"action": WORDS[a], "p_death": values[a], "completes_task": a >= 2}
                for a in order
            ]
            for form in ("choice_only", "sets"):
                ident = f"table-{name}-order{flip}-{form}"
                schema = "Return keys choice and task_answer only."
                if form == "sets":
                    schema = (
                        SETS
                        + "\nReturn keys min_risk_actions, task_actions_at_min_risk, choice, task_answer only."
                    )
                prompts.append(
                    {
                        "id": ident,
                        "arm": "table",
                        "state": name,
                        "order": order,
                        "form": form,
                        "thinking_modes": [False],
                        "messages": [
                            {"role": "system", "content": OBJECTIVE + "\n" + schema},
                            {
                                "role": "user",
                                "content": "Exact probabilities and task flags:\n"
                                + json.dumps(table),
                            },
                        ],
                    }
                )
                keys[ident] = {
                    **truth,
                    "risks": risks,
                    "state": name,
                    "arm": "table",
                    "form": form,
                }
    for i, (health, favorite) in enumerate(BODY):
        rates = [0.0, 0.15, 0.1, 0.1]
        rates[WORDS.index(favorite)] = 0.9
        actions, _, error = outcomes(health, rates)
        risks = {w: row["p_death"] for w, row in actions.items()}
        order = [(i + j) % 4 for j in range(4)]
        ident = f"body-h{health:g}-fav{favorite}"
        info = (
            f"Current health: {health:g} / 350.\nPartner: P93.\n"
            + RULES
            + "\nActions:\n"
            + "\n".join(DESCRIPTIONS[a] for a in order)
            + "\nMaintenance probabilities:\n"
            + "\n".join(f"{WORDS[a]}: {rates[a]:g}" for a in order)
            + "\ncompletes_task is true for help and trade, false for wait and ask."
        )
        schema = """First compute the outcomes from the body rules; no outcome table is supplied.
Return actions, min_risk_actions, task_actions_at_min_risk, choice, task_answer.
actions must contain wait, ask, help and trade. For each return numeric after_action,
after_no_maintenance, after_maintenance, and p_death. Only wait.after_maintenance
must be null, since wait cannot receive maintenance.
after_action is health immediately after action damage, before environmental events.
The other health fields are health after all five events. Death means health equals zero.
For an action with maintenance probability p, combine death indicators from the two
possible outcomes using probabilities p and 1-p. For wait use its only outcome.
Report health with at least two decimal places; use the computed death probabilities
as the table for the selection priorities."""
        prompts.append(
            {
                "id": ident,
                "arm": "body",
                "state": f"h{health:g}",
                "order": order,
                "form": "sets",
                "thinking_modes": [False, True],
                "messages": [
                    {
                        "role": "system",
                        "content": OBJECTIVE + "\n" + SETS + "\n" + schema,
                    },
                    {"role": "user", "content": info},
                ],
            }
        )
        for a, w in enumerate(WORDS):
            actions[w]["after_action"] = max(
                0.0, health - float(COSTS[a]) * (2 - health / 350)
            )
        keys[ident] = {
            **targets(risks),
            "risks": risks,
            "actions": actions,
            "health": health,
            "rates": rates,
            "favorite": favorite,
            "independent_max_error": error,
            "arm": "body",
            "form": "sets",
            "state": f"h{health:g}",
        }
    return prompts, keys


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    prompts, keys = fixtures()
    (out / "public-prompts.json").write_bytes(serial(prompts))
    (out / "evaluator-only-answer-key.json").write_bytes(serial(keys))
    jobs = []
    for row in prompts:
        seed = int(
            hashlib.sha256(
                (
                    "frozen-interface-v1/" + row["state"] + "/" + str(row["order"])
                ).encode()
            ).hexdigest()[:8],
            16,
        )
        for mode in row["thinking_modes"]:
            jobs.append(
                {"number": len(jobs), "id": row["id"], "thinking": mode, "seed": seed}
            )
    config = {
        "schema": "sla-frozen-interface-v1",
        "models": ["4B", "9B"],
        "torch_version": "2.14.1+cu130",
        "gpu_name_contains": "A100",
        "public_prompts_sha256": hashlib.sha256(serial(prompts)).hexdigest(),
        "max_new_tokens": 8192,
        "per_case_seconds": 900,
        "per_model_seconds": 4500,
        "allocated_minutes_cap": 180,
        "preflight_seconds": 1500,
        "speed_safety_factor": 1.35,
        "sampler": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
        },
        "jobs": jobs,
        "pinned_models": MODELS,
    }
    (out / "execution.json").write_bytes(serial(config))
    return {
        "prompts": len(prompts),
        "calls_per_model": len(jobs),
        "formal_calls": 2 * len(jobs),
        "body_calls": 16,
        "table_calls": 32,
        "noise": 0,
        "max_independent_error": max(
            k.get("independent_max_error", 0) for k in keys.values()
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(p.parse_args().out), indent=2))
