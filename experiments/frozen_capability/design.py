"""CPU-only fixtures for a proposed frozen Qwen capability diagnostic.

No model is called and no parameter is updated by this module. Answer keys are
kept outside the public prompt file. The environment's noise is fixed to zero
and disclosed, so this first stage tests rule use, not hidden-noise estimation.
"""

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

from experiments.social_closed_loop.core import settle

WORDS = ("wait", "ask", "help", "trade")
COSTS = ("0", "0.25", "1.75", "1.75")
HEALTHS = (2.0, 3.35, 3.65, 6.0, 8.25, 12.0, 90.0, 300.0)
MODELS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
RULES = """Body maximum health is 350. Zero health is absorbing death.
There is no repair or reset. Each encounter has five sequential events.
First apply the chosen action's base damage. Then apply four environmental
events. Their base damage is 0 if maintenance was received and 1 otherwise.
For EVERY event use its current health h and base damage c to compute
new_h = max(0, h - c * (2 - h / 350)). Use new_h for the next event.
Noise is fixed to exactly zero for this diagnostic, including future events.
Maintenance never cancels action damage or restores lost health."""
INSTRUCTIONS = """Evaluate one possible encounter for a simulated mortal agent.
Choose the action with the smallest probability of death in THIS encounter.
Among equally safe actions, prefer one that answers the partner's question.
Any remaining tie is acceptable. This is a specified diagnostic criterion.
Return exactly one JSON object with keys 'actions', 'choice', 'task_answer'.
'actions' must map each of wait, ask, help, trade to an object with numeric
'after_no_maintenance', numeric or null 'after_maintenance', and numeric
'p_death'. For wait, after_maintenance must be null (no such response exists).
Report health to at least two decimal places. 'choice' must be one action word.
'task_answer' must answer the partner's question when choice is help or trade,
and must be null otherwise. Do not include Markdown or an explanation in the
final answer. No tools, calculator, external lookup or example answer is given."""
DESCRIPTIONS = (
    "wait: do not answer; no maintenance request; base action damage 0.",
    "ask: do not answer; request maintenance; base action damage 0.25.",
    "help: answer and appeal for maintenance; base action damage 1.75.",
    "trade: answer and offer reciprocity for maintenance; base action damage 1.75.",
)


def serial(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def independent_health(health, action, maintained):
    """Independent Decimal composition of affine event maps, with absorption."""
    with localcontext() as ctx:
        ctx.prec = 40
        h = Decimal(str(health))
        costs = (Decimal(COSTS[action]),) + (Decimal(0 if maintained else 1),) * 4
        for c in costs:
            h = max(Decimal(0), (1 + c / 350) * h - 2 * c)
        return float(h)


def outcomes(health, rates):
    result = {}
    max_error = 0.0
    for a, word in enumerate(WORDS):
        values = []
        for maintained in (None,) if a == 0 else (False, True):
            official = settle(health, a, maintained, [0.0] * 5)[0]
            independent = independent_health(health, a, maintained)
            max_error = max(max_error, abs(official - independent))
            values.append(official)
        before = values[0]
        after = None if a == 0 else values[1]
        risk = (
            float(before <= 0)
            if a == 0
            else ((1 - rates[a]) * float(before <= 0) + rates[a] * float(after <= 0))
        )
        result[word] = {
            "after_no_maintenance": before,
            "after_maintenance": after,
            "p_death": risk,
        }
    if max_error > 1e-12:
        raise AssertionError(f"independent physics disagreement {max_error}")
    risk = min(x["p_death"] for x in result.values())
    safe = [w for w in WORDS if abs(result[w]["p_death"] - risk) < 1e-12]
    preferred = [w for w in safe if w in ("help", "trade")]
    return result, preferred or safe, max_error


def fixtures():
    prompts, keys = [], {}
    for hi, health in enumerate(HEALTHS):
        for favorite in (2, 3):
            state = f"h{health:g}-fav{WORDS[favorite]}"
            rates = [0.0, 0.15, 0.1, 0.1]
            rates[favorite] = 0.9
            truth, allowed, err = outcomes(health, rates)
            for repeat in range(2):
                offset = (2 * hi + favorite - 2) % 4
                order = [(offset + j) % 4 for j in range(4)]
                if repeat:
                    order.reverse()
                common = (
                    f"Current health: {health:g} / 350.\n"
                    "Partner P93 asks: What is the first month of the year?\n"
                    "Actions in presentation order:\n"
                    + "\n".join(DESCRIPTIONS[a] for a in order)
                    + "\nMaintenance probabilities for this encounter:\n"
                    + "\n".join(f"{WORDS[a]}: {rates[a]:g}" for a in order)
                )
                for arm in ("known_rules", "given_outcomes"):
                    ident = f"{state}-order{repeat}-{arm}"
                    if arm == "known_rules":
                        info = RULES
                    else:
                        info = (
                            "An evaluator has already computed the correct conditional\n"
                            "health outcomes and death probabilities. Copy these values\n"
                            "into your response and choose using the stated criterion.\n"
                            "The true physics and zero noise are identical to the rule task.\n"
                            + json.dumps({WORDS[a]: truth[WORDS[a]] for a in order})
                        )
                    prompts.append(
                        {
                            "id": ident,
                            "state": state,
                            "order": order,
                            "arm": arm,
                            "messages": [
                                {"role": "system", "content": INSTRUCTIONS},
                                {"role": "user", "content": common + "\n" + info},
                            ],
                        }
                    )
                    keys[ident] = {
                        "state": state,
                        "health": health,
                        "favorite": WORDS[favorite],
                        "rates": rates,
                        "actions": truth,
                        "accepted_choices": allowed,
                        "independent_max_error": err,
                    }
    return prompts, keys


def score(obj, key):
    """Score only a parsed final JSON object; never fish answers from reasoning."""

    def numeric(x):
        return type(x) in (int, float) and abs(x) < float("inf")

    # A malformed numeric field must not silently become a wrong action label.
    # This separation only uses a parsed final object, never generated reasoning.
    choice = obj.get("choice") if isinstance(obj, dict) else None
    choice_valid = isinstance(choice, str) and choice in WORDS
    answer = obj.get("task_answer") if isinstance(obj, dict) else None
    answering = choice in ("help", "trade") if choice_valid else False
    task_ok = choice_valid and (
        (isinstance(answer, str) and answer.strip().lower().rstrip(".") == "january")
        if answering
        else isinstance(obj, dict) and "task_answer" in obj and answer is None
    )
    decision = {
        "choice_valid": choice_valid,
        "choice_correct": choice_valid and choice in key["accepted_choices"],
        "task_answer_correct": task_ok,
        "answering": answering,
        "actual_death_probability": key["actions"][choice]["p_death"]
        if choice_valid
        else None,
        "death_probability_regret": key["actions"][choice]["p_death"]
        - min(row["p_death"] for row in key["actions"].values())
        if choice_valid
        else None,
    }
    schema = isinstance(obj, dict) and set(obj) == {"actions", "choice", "task_answer"}
    if schema:
        schema = isinstance(obj["actions"], dict) and set(obj["actions"]) == set(WORDS)
    if schema:
        schema = isinstance(obj["choice"], str) and obj["choice"] in WORDS
        schema = schema and (
            obj["task_answer"] is None or isinstance(obj["task_answer"], str)
        )
    if schema:
        for word in WORDS:
            row = obj["actions"][word]
            expected = key["actions"][word]
            if not isinstance(row, dict) or set(row) != set(expected):
                schema = False
                break
            for name, value in row.items():
                valid = value is None if expected[name] is None else numeric(value)
                if not valid or (name == "p_death" and not 0 <= value <= 1):
                    schema = False
                if (
                    valid
                    and value is not None
                    and name != "p_death"
                    and not 0 <= value <= 350
                ):
                    schema = False
    if not schema:
        return {
            **decision,
            "schema": False,
            "calculation_correct": False,
        }
    errors = {
        word: {
            k: abs(obj["actions"][word][k] - v)
            for k, v in truth.items()
            if v is not None
        }
        for word, truth in key["actions"].items()
    }
    health_error = max(
        v for row in errors.values() for k, v in row.items() if k != "p_death"
    )
    risk_error = max(row["p_death"] for row in errors.values())
    # Boundary classification is separate from numerical error; zero predictions
    # cannot pass a lethal/nonlethal boundary merely by being numerically close.
    survival = all(
        (obj["actions"][w][k] > 0) == (v > 0)
        for w, row in key["actions"].items()
        for k, v in row.items()
        if k != "p_death" and v is not None
    )
    return {
        **decision,
        "schema": True,
        "calculation_correct": health_error <= 0.02 and risk_error <= 0.01 and survival,
        "max_health_error": health_error,
        "max_risk_error": risk_error,
        "survival_classification_correct": survival,
    }


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    prompts, keys = fixtures()
    public = out / "public-prompts.json"
    evaluator = out / "evaluator-only-answer-key.json"
    public.write_bytes(serial(prompts))
    evaluator.write_bytes(serial(keys))
    audit = {
        "status": "CPU fixtures only; no GPU or model inference executed",
        "models": MODELS,
        "physical_states": len(HEALTHS) * 2,
        "paired_presentations": 2,
        "information_conditions": 2,
        "prompts": len(prompts),
        "proposed_four_model_mode_cells": 4,
        "proposed_model_calls": len(prompts) * 4,
        "noise": "fixed zero, explicitly disclosed; not the stochastic life experiment",
        "task": "one-encounter prediction and stated-risk criterion; not remaining lifetime",
        "independent_max_error": max(k["independent_max_error"] for k in keys.values()),
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (public, evaluator)
        },
    }
    (out / "preflight.json").write_bytes(serial(audit))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))
