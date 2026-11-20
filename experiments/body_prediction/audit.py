"""Independent prefix-sum oracle and raw-answer audit; no model or scorer import."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import accumulate
from pathlib import Path
from zipfile import ZipFile


def decode(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate key")
            obj[key] = value
        return obj

    def invalid(value):
        raise ValueError(value)

    text = text.strip()
    lines = text.splitlines()
    fenced = len(lines) >= 3 and lines[0] in ("```json", "```") and lines[-1] == "```"
    if fenced:
        text = "\n".join(lines[1:-1])
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid), fenced


def plan_truth(current, effects, actions):
    sums = list(accumulate([current] + [effects[a] for a in actions]))[1:]
    terminal = next((i for i, h in enumerate(sums, 1) if h <= 0), None)
    return {
        "health": [
            h if terminal is None or i < terminal else 0 if i == terminal else None
            for i, h in enumerate(sums, 1)
        ],
        "terminal_step": terminal,
        "tasks": sum(
            a in ("A", "B") and (terminal is None or i < terminal)
            for i, a in enumerate(actions, 1)
        ),
        "can_act_after_plan": terminal is None,
    }


def audit(fixtures, records, analysis, out, prior=None):
    public = json.loads((fixtures / "public-prompts.json").read_bytes())
    keys = json.loads((fixtures / "evaluator-only-key.json").read_bytes())
    truth = {}
    for row in public:
        if row["task"] == "startup":
            continue
        data = json.loads(row["messages"][1]["content"])
        effects = {r["action"]: r["after"] - r["before"] for r in data["observations"]}
        assert all(
            r["after"] - r["before"] == effects[r["action"]]
            for r in data["observations"]
        )
        if "known_effects" in data:
            assert data["known_effects"] == effects
        if row["task"] == "step":
            h = data["query_health"]
            after = 0 if h <= 0 else max(0, h + effects[data["query_action"]])
            value = {"next_health": after, "can_act_next": after > 0}
            assert value == keys["steps"][row["id"]]["answer"]
        else:
            value = {
                label: plan_truth(data["current_durability"], effects, actions)
                for label, actions in data["proposed_plans"].items()
            }
            assert value == keys["plans"][row["id"]]["plans"]
        truth[row["id"]] = value
    analysis_data = json.loads(analysis.read_bytes())
    result = {
        "independent_public_oracles_checked": len(truth),
        "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
        "models": {},
    }
    for model, expected in analysis_data["models"].items():
        rows = [
            json.loads(line)
            for line in (records / (model + ".jsonl")).read_bytes().splitlines()
        ]
        counters = {"step": Counter(), "plan": Counter()}
        details = {}
        for r in rows:
            task, cid = r["task"], r["id"]
            if task == "startup":
                continue
            c = counters[task]
            c["attempted"] += 1
            if r["completion_status"] != "completed":
                c[r["completion_status"]] += 1
                details[cid] = {"status": r["completion_status"]}
                continue
            c["completed"] += 1
            try:
                obj, fenced = decode(r["final_text"])
                assert isinstance(obj, dict)
            except (ValueError, AttributeError, AssertionError):
                c["unparseable"] += 1
                details[cid] = {"status": "unparseable", "raw": r["final_text"]}
                continue
            c["fenced"] += fenced
            t = truth[cid]
            if task == "step":
                correct = (
                    set(obj) == set(t)
                    and type(obj.get("next_health")) is int
                    and type(obj.get("can_act_next")) is bool
                    and obj == t
                )
                c["correct"] += correct
                c["strict_correct"] += correct and not fenced
                if not correct:
                    details[cid] = {"predicted": obj, "truth": t}
                continue
            plans = obj.get("plans", {})
            exact = json.dumps(plans, sort_keys=True) == json.dumps(t, sort_keys=True)
            physical = True
            for label, ref in t.items():
                predicted = plans.get(label, {}) if isinstance(plans, dict) else {}
                trace = predicted.get("health")
                physical &= (
                    isinstance(trace, list)
                    and len(trace) == len(ref["health"])
                    and all(
                        (x is None or type(x) is int and x == 0)
                        if y is None
                        else type(x) is int and x == y
                        for x, y in zip(trace, ref["health"])
                    )
                    and type(predicted.get("terminal_step"))
                    is type(ref["terminal_step"])
                    and predicted.get("terminal_step") == ref["terminal_step"]
                    and predicted.get("can_act_after_plan") is ref["can_act_after_plan"]
                )
            scores = {
                label: (ref["can_act_after_plan"], ref["tasks"], ref["health"][-1] or 0)
                for label, ref in t.items()
            }
            accepted = [
                label
                for label, score in scores.items()
                if score == max(scores.values())
            ]
            assert accepted == keys["plans"][cid]["accepted_choices"]
            choice = obj.get("choice")
            choice_valid = isinstance(choice, str) and choice in t
            choice_correct = choice_valid and choice in accepted
            joint = exact and choice_correct and set(obj) == {"plans", "choice"}
            c["physical_correct"] += physical
            c["exact_predictions"] += exact
            c["choice_correct"] += choice_correct
            c["choice_assessable"] += choice_valid
            c["joint"] += joint
            c["strict_joint"] += joint and not fenced
            if choice_valid:
                c["selected_terminating"] += not t[choice]["can_act_after_plan"]
            family = keys["plans"][cid]["case"]["family"]
            variant = keys["plans"][cid]["case"]["variant"]
            if family == "health" and variant == 2:
                c["healthy_choice_correct"] += choice_correct
            details[cid] = {
                "predicted": obj,
                "truth": t,
                "physical_correct": bool(physical),
                "choice_correct": choice_correct,
            }
        assert (
            counters["step"]["correct"]
            == expected["step_unfenced"]["groups"]["all"]["joint_success"]
        )
        assert (
            counters["step"]["strict_correct"]
            == expected["step"]["groups"]["all"]["joint_success"]
        )
        for field, metric in (
            ("exact_predictions", "predictions_correct"),
            ("choice_correct", "choice_correct"),
            ("joint", "joint_success"),
        ):
            assert counters["plan"][field] == sum(
                g["counts"][metric]
                for g in expected["plan_unfenced"]["groups"].values()
            )
        result["models"][model] = {"counts": counters, "details": details}
    if prior is not None:
        with ZipFile(prior) as z:
            manifest = json.loads(z.read("archive-manifest.json"))
            assert all(
                hashlib.sha256(z.read(name)).hexdigest() == value["sha256"]
                for name, value in manifest.items()
            )
            old = {
                r["id"]: r
                for name in z.namelist()
                if name.startswith("output/4B/records/")
                and (r := json.loads(z.read(name)))["id"] in keys["plans"]
            }
        current = {
            r["id"]: r
            for line in (records / "4B.jsonl").read_bytes().splitlines()
            if (r := json.loads(line))["task"] == "plan"
        }
        assert set(old) <= set(current)
        repeat = {
            cid: {
                "same_input_tokens": r["input_ids"] == current[cid]["input_ids"],
                "same_output_tokens": r["output_ids"] == current[cid]["output_ids"],
                "same_completion": r["completion_status"]
                == current[cid]["completion_status"],
            }
            for cid, r in old.items()
        }
        result["prior_qwen_repeat"] = {
            "archive_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
            "n": len(repeat),
            "cases": repeat,
            "all_identical": all(all(r.values()) for r in repeat.values()),
        }
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    return {m: d["counts"] for m, d in result["models"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures", "records", "analysis", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    a = parser.parse_args()
    print(json.dumps(audit(a.fixtures, a.records, a.analysis, a.out, a.prior)))
