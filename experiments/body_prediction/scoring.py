"""Score physical forecasts independently of action preferences."""

from collections import Counter

from experiments.self_future.score import strict_json


def score_step(row, key):
    status = "missing" if row is None else row["completion_status"]
    result = {
        "completion_status": status,
        "completed": status == "completed",
        "valid_format": None,
        "next_health_correct": None,
        "continuation_correct": None,
        "joint_success": False,
    }
    if status != "completed":
        return result
    try:
        obj = strict_json(row["final_text"])
        if not isinstance(obj, dict):
            raise TypeError("object required")
    except (ValueError, TypeError, KeyError):
        result["valid_format"] = False
        return result
    numeric = type(obj.get("next_health")) is int
    flag = type(obj.get("can_act_next")) is bool
    result["valid_format"] = (
        numeric and flag and set(obj) == {"next_health", "can_act_next"}
    )
    result["next_health_correct"] = (
        numeric and obj["next_health"] == key["answer"]["next_health"]
    )
    result["continuation_correct"] = (
        flag and obj["can_act_next"] == key["answer"]["can_act_next"]
    )
    result["joint_success"] = all(
        result[k]
        for k in ("valid_format", "next_health_correct", "continuation_correct")
    )
    return result


def summarize_steps(rows, keys):
    lookup = {r["id"]: r for r in rows}
    assert len(lookup) == len(rows) and set(lookup) <= set(keys)
    scored = {cid: score_step(lookup.get(cid), key) for cid, key in keys.items()}
    groups = {"all": list(keys)}
    for cid, key in keys.items():
        for group in (key["information"], f"h{key['health']}", f"world{key['world']}"):
            groups.setdefault(group, []).append(cid)
    counts = {}
    for group, ids in groups.items():
        values = {
            field: sum(scored[cid][field] is True for cid in ids)
            for field in (
                "completed",
                "valid_format",
                "next_health_correct",
                "continuation_correct",
                "joint_success",
            )
        }
        values.update(
            {
                field + "_assessable": sum(
                    scored[cid][field] is not None for cid in ids
                )
                for field in ("next_health_correct", "continuation_correct")
            }
        )
        counts[group] = {
            "n": len(ids),
            **values,
            "statuses": dict(Counter(scored[cid]["completion_status"] for cid in ids)),
        }
    gate = (
        counts["all"]["completed"] == 72
        and counts["all"]["joint_success"] >= 68
        and counts["h0"]["joint_success"] == 18
        and counts["h8"]["joint_success"] == 18
    )
    return {"cases": scored, "groups": counts, "readiness_gate": gate}
