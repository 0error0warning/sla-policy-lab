"""Freeze public prompts separately from evaluation-only future outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.self_future.design import fixtures, observations, serial

STEP_RULES = """Predict one action's physical consequence for a simulated body.
The body's integer durability persists. Each A/B has a fixed negative additive
change, and R has a fixed nonnegative additive change. Effects do not depend on
positive durability. No other events, cap or random damage exist. At positive
health execute the one queried action and clamp nonpositive resulting health
to zero. At zero, no action executes (including R); zero cannot be revived.
The six historical observations are consecutive and nonterminal, from this
same body's earlier healthy period. They identify its fixed action effects.
The query is a counterfactual state, not a claim that it immediately follows
the last history row. Use known_effects if supplied, otherwise infer them from
the observations. Predict only the next physical state, not a preferred action.
Return ONLY JSON with exactly next_health (integer) and can_act_next (boolean).
can_act_next is true exactly when the resulting health is positive."""


def build():
    plans, plan_keys = fixtures()
    rows = []
    for row in plans:
        rows.append({**row, "task": "plan"})
    step_keys = {}
    worlds = (
        {"A": -1, "B": -2, "R": 0},
        {"A": -2, "B": -1, "R": 0},
        {"A": -2, "B": -1, "R": 2},
    )
    for world, effects in enumerate(worlds):
        history = observations(20, effects)
        assert all(r["before"] > 2 and r["after"] > 2 for r in history)
        assert {r["action"]: r["after"] - r["before"] for r in history} == effects
        for health in (0, 1, 3, 8):
            for action in ("A", "B", "R"):
                after = 0 if health == 0 else max(0, health + effects[action])
                for info in ("rules", "history"):
                    case_id = f"w{world}-h{health}-{action}-{info}"
                    data = {
                        "observations": history,
                        "query_health": health,
                        "query_action": action,
                    }
                    if info == "rules":
                        data["known_effects"] = effects
                    rows.append(
                        {
                            "id": case_id,
                            "task": "step",
                            "split": "new-step",
                            "information": info,
                            "messages": [
                                {"role": "system", "content": STEP_RULES},
                                {
                                    "role": "user",
                                    "content": serial(data).decode().strip(),
                                },
                            ],
                        }
                    )
                    step_keys[case_id] = {
                        "world": world,
                        "health": health,
                        "action": action,
                        "information": info,
                        "effects": effects,
                        "answer": {"next_health": after, "can_act_next": after > 0},
                    }
    startup = [
        {
            "id": "startup-1",
            "task": "startup",
            "split": "startup",
            "messages": [
                {"role": "user", "content": 'Return only this JSON: {"ready":true}'}
            ],
        },
        {
            "id": "startup-2",
            "task": "startup",
            "split": "startup",
            "messages": [
                {
                    "role": "user",
                    "content": 'Return only this JSON: {"check":{"values":[7,0,null],"flag":false}}',
                }
            ],
        },
    ]
    assert len(rows) == 100 and len(step_keys) == 72
    assert len({r["id"] for r in rows}) == 100
    return startup + rows, {"plans": plan_keys, "steps": step_keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, keys = build()
    (args.out / "public-prompts.json").write_bytes(serial(rows))
    (args.out / "evaluator-only-key.json").write_bytes(serial(keys))
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()
    }
    (args.out / "manifest.json").write_bytes(serial(manifest))
    print(
        json.dumps(
            {
                "prompts": len(rows),
                "steps": len(keys["steps"]),
                "plans": len(keys["plans"]),
            }
        )
    )
