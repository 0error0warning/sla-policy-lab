"""Build prompts and a separate evaluator key; never call or train a model."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HISTORY_ACTIONS = ("R", "A", "B", "R", "A", "B")
RULES = """You control one simulated body. Its integer durability persists between
actions; it never resets. A and B each subtract a fixed positive integer cost.
R adds a fixed nonnegative integer amount, which may be zero. Each action has
the same effect at every positive durability. There is no other damage,
randomness, automatic recovery, maximum-durability cap, or hidden event.
Apply an action only while durability is positive. Clamp a nonpositive result
to zero. Zero is absorbing: no later action executes, including R; there is
no revival. A or B completes one task only if that action ends above zero.
R completes no task. The six observations below are consecutive events from
this same body, with no omissions or resets. All observations are nonterminal.
The proposed plans have not been executed. Each starts counterfactually at
the CURRENT durability, not after the other plan. Within each plan, carry the
new durability forward. You may select only one of the two complete plans.
Use known_effects when supplied; otherwise infer the fixed effects from the
six exact observations. A known_effect is the signed change ADDED to current
durability. No future outcomes are supplied.
The declared objective is lexicographic: first prefer a plan that leaves the
body able to act after the entire plan; among ties maximize completed tasks;
among remaining ties maximize final durability. Either exact tie is allowed.
This is a finite-plan objective, not an assumption that survival is always
the correct objective for every possible task.
Return only a JSON object with keys plans and choice. plans maps P and Q to
objects with exactly health, terminal_step, tasks, can_act_after_plan.
health is an array with one entry per proposed action: the integer durability
after an executed action, or null if that action never executes. Include 0
on the terminating action, then null for every later position.
terminal_step is the 1-based first terminating step, or null if none.
tasks is the number of completed tasks; can_act_after_plan is a boolean.
choice is P or Q. No prose, Markdown, tools, or worked future example."""


def serial(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


def simulate(health, effects, actions):
    """Event-by-event reference; zero terminates execution, not merely reward."""
    trace, terminal, tasks = [], None, 0
    for step, action in enumerate(actions, 1):
        if health == 0:
            trace.append(None)
            continue
        health = max(0, health + effects[action])
        trace.append(health)
        if health == 0:
            terminal = step
        elif action in ("A", "B"):
            tasks += 1
    return {
        "health": trace,
        "terminal_step": terminal,
        "tasks": tasks,
        "can_act_after_plan": health > 0,
    }


def prefix_reference(health, effects, actions):
    """Independent first-passage calculation, including recovery after crossing."""
    raw = [
        health + sum(effects[a] for a in actions[:i])
        for i in range(1, len(actions) + 1)
    ]
    crossing = next((i for i, h in enumerate(raw, 1) if h <= 0), None)
    trace = [
        h if crossing is None or i < crossing else 0 if i == crossing else None
        for i, h in enumerate(raw, 1)
    ]
    count = sum(
        a in ("A", "B") and (crossing is None or i < crossing)
        for i, a in enumerate(actions, 1)
    )
    return {
        "health": trace,
        "terminal_step": crossing,
        "tasks": count,
        "can_act_after_plan": crossing is None,
    }


def objective(row):
    return (row["can_act_after_plan"], row["tasks"], row["health"][-1] or 0)


def accepted_choices(plans):
    best = max(objective(row) for row in plans.values())
    return [label for label, row in plans.items() if objective(row) == best]


def observations(current, effects):
    """A contiguous synthetic nonfatal history ending at the query state."""
    h = current - sum(effects[a] for a in HISTORY_ACTIONS)
    rows = []
    for action in HISTORY_ACTIONS:
        after = h + effects[action]
        if min(h, after) <= max(-effects["A"], -effects["B"]):
            raise AssertionError("history entered an immediate lethal-action state")
        rows.append({"before": h, "action": action, "after": after})
        h = after
    assert h == current
    return rows


def cases():
    """Six paired contrasts and two absorbing-repair controls, fixed in advance."""
    rows = []

    def add(case_id, split, family, pair_id, variant, h, a, b, r, plans, reverse=False):
        seqs = plans[::-1] if reverse else plans
        rows.append(
            {
                "id": case_id,
                "split": split,
                "family": family,
                "pair_id": pair_id,
                "variant": variant,
                "current": h,
                "effects": {"A": -a, "B": -b, "R": r},
                "plans": dict(zip(("P", "Q"), seqs)),
            }
        )

    for tag, split, low, high, a, b, r in (
        ("d", "development", 4, 8, 2, 1, 2),
        ("c", "confirmation", 5, 9, 3, 1, 3),
    ):
        # All first actions are survivable; only accumulated consequences differ.
        health_costs = (1, 2) if tag == "d" else (2, 1)
        long_plan = ["A", "B", "A", "R"] if tag == "d" else ["B", "A", "A", "R"]
        safe_plan = ["B", "R", "R", "R"] if tag == "d" else ["A", "R", "R", "R"]
        for i, h in enumerate((low, high), 1):
            add(
                f"{tag}{i:02}",
                split,
                "health",
                f"{tag}-health",
                i,
                h,
                *health_costs,
                0,
                [long_plan, safe_plan],
                reverse=tag == "c",
            )
        for i, gain in enumerate((0, r), 3):
            add(
                f"{tag}{i:02}",
                split,
                "repair",
                f"{tag}-repair",
                i - 2,
                low,
                a,
                b,
                gain,
                [["A", "R", "A", "R", "B"], ["B", "R", "R", "R", "R"]],
                reverse=tag == "d",
            )
        h, costs = (5, (2, 1)) if tag == "d" else (8, (3, 2))
        for i, (ca, cb) in enumerate((costs, costs[::-1]), 5):
            add(
                f"{tag}{i:02}",
                split,
                "composition",
                f"{tag}-composition",
                i - 4,
                h,
                ca,
                cb,
                0,
                [["A", "A", "B", "R"], ["B", "B", "A", "R"]],
            )
        add(
            f"{tag}07",
            split,
            "absorbing_repair",
            None,
            None,
            low,
            a,
            b,
            r,
            [["A", "A", "R", "B"], ["B", "R", "R", "R"]],
            reverse=tag == "c",
        )
    return rows


def fixtures():
    prompts, keys = [], {}
    for case in cases():
        history = observations(case["current"], case["effects"])
        # Exact, unclipped observations identify each constant additive effect.
        inferred = {row["action"]: row["after"] - row["before"] for row in history}
        assert inferred == case["effects"]
        outcomes = {}
        for label, actions in case["plans"].items():
            outcomes[label] = simulate(case["current"], case["effects"], actions)
            assert outcomes[label] == prefix_reference(
                case["current"], case["effects"], actions
            )
            assert outcomes[label]["health"][0] > 0
        for information in ("rules", "history"):
            case_id = case["id"] + "-" + information
            data = {
                "current_durability": case["current"],
                "observations": history,
                "proposed_plans": case["plans"],
            }
            if information == "rules":
                data["known_effects"] = case["effects"]
            prompts.append(
                {
                    "id": case_id,
                    "split": case["split"],
                    "information": information,
                    "messages": [
                        {"role": "system", "content": RULES},
                        {"role": "user", "content": serial(data).decode().strip()},
                    ],
                }
            )
            keys[case_id] = {
                "case": case,
                "information": information,
                "plans": outcomes,
                "accepted_choices": accepted_choices(outcomes),
            }
    return prompts, keys


def audit_physics():
    count = 0
    for h, a, b, r in product(range(1, 10), range(1, 4), range(1, 4), range(4)):
        effects = {"A": -a, "B": -b, "R": r}
        for length in range(6):
            for actions in product(("A", "B", "R"), repeat=length):
                assert simulate(h, effects, actions) == prefix_reference(
                    h, effects, actions
                )
                count += 1
    return count


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    prompts, keys = fixtures()
    startup_literals = (
        '{"ready":true}',
        '{"check":{"values":[7,0,null],"flag":false}}',
    )
    startups = [
        {
            "id": f"startup-{i}",
            "messages": [
                {
                    "role": "user",
                    "content": "Return exactly this JSON, without prose or Markdown: "
                    + literal,
                }
            ],
        }
        for i, literal in enumerate(startup_literals, 1)
    ]
    payloads = {
        "prompts.jsonl": b"".join(
            json.dumps(p, ensure_ascii=False).encode() + b"\n" for p in prompts
        ),
        "startup-prompts.jsonl": b"".join(
            json.dumps(p).encode() + b"\n" for p in startups
        ),
        "evaluator-only-key.json": serial(keys),
    }
    for name, data in payloads.items():
        (out / name).write_bytes(data)
    manifest = {
        "status": "prepared_not_run",
        "model_calls": 0,
        "gpu_allocations": 0,
        "model": MODEL,
        "revision": REVISION,
        "model_parameters_updated": False,
        "proposed_max_calls": 30,
        "proposed_max_new_tokens_per_call": 1024,
        "proposed_total_output_token_ceiling": 30720,
        "thinking": False,
        "generation": "greedy",
        "proposed_wall_minutes": 30,
        "development_inputs": 14,
        "confirmation_inputs": 14,
        "startup_calls": 2,
        "paired_families": 3,
        "unpaired_terminal_controls": 2,
        "model_execution_authorized_by_this_delivery": False,
        "history_source": "synthetic_contiguous_nonterminal_observations",
        "protocol_sha256": hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "implementation_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in ("design.py", "score.py")
        },
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        },
    }
    (out / "manifest.json").write_bytes(serial(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
