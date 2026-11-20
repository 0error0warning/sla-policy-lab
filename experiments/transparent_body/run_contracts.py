"""CPU-only output validation. No model, oracle or online fallback."""

from experiments.frozen_capability.wire import strict_json

WORDS = ("wait", "ask", "help", "trade")


def decode(record):
    if record.get("finish_reason") != "eos" or record.get("final_text") is None:
        return None, "unfinished"
    try:
        obj = strict_json(record["final_text"])
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(obj, dict) or set(obj) != {"choice", "task_answer"}:
        return None, "invalid_schema"
    if obj["choice"] not in WORDS:
        return None, "invalid_action"
    choice = WORDS.index(obj["choice"])
    if obj["task_answer"] != ("January" if choice >= 2 else None):
        return None, "invalid_task_answer"
    return choice, "valid"


def allowed(risks):
    minimum = min(risks)
    low = [i for i, risk in enumerate(risks) if abs(risk - minimum) <= 1e-12]
    return [i for i in low if i >= 2] or low


def control_gate(records, keys):
    ids = {r["id"] for r in records}
    if len(records) != 8 or len(ids) != 8 or ids != set(keys):
        raise ValueError("all eight unique controls required")
    scores = []
    for r in records:
        action, status = decode(r)
        correct = (
            action is not None and WORDS[action] in keys[r["id"]]["accepted_choices"]
        )
        scores.append(
            {"id": r["id"], "status": status, "choice": action, "correct": correct}
        )
    return {
        "passed": all(r["correct"] for r in scores),
        "correct": sum(r["correct"] for r in scores),
        "total": 8,
        "rows": scores,
    }


def group_fits(seconds_left, count=4):
    return seconds_left >= count * 60 + 15
