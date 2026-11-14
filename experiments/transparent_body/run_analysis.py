"""Independent token replay and predeclared gates; oracle stays on CPU."""

import argparse
import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from experiments.frozen_capability.design import serial
from experiments.frozen_capability.scoring import assistant_message
from experiments.individual_life.design import offers
from experiments.lifetime_cooperation.design import ITEMS, rules

from .body import WORDS, BodyCalculator
from .run_contracts import allowed, control_gate, decode


def verify_tokens(src, inputs, metadata):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(sha256(z.read(n)).hexdigest() == h for n, h in lock["files"].items())
        assert not any(
            "key" in n or n.endswith(("body.py", "run_analysis.py", "run_contracts.py"))
            for n in z.namelist()
        )
        cfg = json.loads(z.read("execution.json"))
        public = json.loads(z.read("public-prompts.json"))
        expected = json.loads(z.read("expected-4B.json"))
    assert json.loads((src / "execution.json").read_bytes()) == cfg
    assert sha256(serial(public)).hexdigest() == cfg["public_prompts_sha256"]
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (
            sha256((metadata / "4B" / name).read_bytes()).hexdigest()
            == expected["metadata"][name]["sha256"]
        )
    assert json.loads((src / "4B/weights-manifest.json").read_bytes()) == expected
    freeze = json.loads((src / "4B/unchanged.json").read_bytes())
    assert (
        freeze["parameter_versions_unchanged"]
        and freeze["gradients_absent"]
        and freeze["trainable_parameters"] == 0
    )
    meta = json.loads((src / "4B/loading.json").read_bytes())
    tok = Tokenizer.from_file(str(metadata / "4B/tokenizer.json"))
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    template = env.from_string(
        (metadata / "4B/chat_template.jinja").read_text(encoding="utf8")
    )
    lookup = {r["id"]: r for r in public}
    rows = []
    for job in cfg["jobs"]:
        path = src / "4B/records" / f"{job['number']:03d}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_bytes())
        assert all(row[k] == v for k, v in job.items())
        assert row["thinking"] is False and row["model"] == "4B"
        text = template.render(
            messages=lookup[job["id"]]["messages"],
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert (
            text == row["prompt_text"]
            and tok.encode(text, add_special_tokens=False).ids == row["input_ids"]
        )
        assert (
            tok.decode(row["output_ids"], skip_special_tokens=False)
            == row["output_text"]
        )
        final, channel, thoughts, boundary = assistant_message(
            row["output_ids"],
            False,
            meta["close_id"],
            meta["open_id"],
            meta["eos_ids"],
            meta["chat_end_id"],
        )
        decoded = (
            None if final is None else tok.decode(final, skip_special_tokens=False)
        )
        assert (final, channel, thoughts, boundary, decoded) == (
            row["final_ids"],
            row["channel"],
            row["thinking_tokens"],
            row["message_end_position"],
            row["final_text"],
        )
        assert row["generated_tokens"] == len(row["output_ids"]) <= 128
        assert row["max_new_tokens"] == 128 and row["max_seconds"] == 60
        g = row["generation_config"]
        assert (
            g["temperature"],
            g["top_p"],
            g["top_k"],
            g["repetition_penalty"],
            row["presence_penalty"],
        ) == (0.6, 0.95, 20, 1.0, 0.0)
        assert g["do_sample"] and g["num_beams"] == 1
        if row["finish_reason"] == "eos":
            assert row["output_ids"][-1] in meta["eos_ids"]
        rows.append(row)
    return rows, cfg, public, freeze


def summarize(rows):
    valid = [r for r in rows if r["status"] == "valid"]
    answering = [r for r in valid if r["choice"] >= 2]
    return {
        "n": len(rows),
        "executed": sum(r["executed"] for r in rows),
        "valid": len(valid),
        "choice_counts": dict(
            Counter(
                WORDS[r["choice"]] if r["choice"] is not None else r["status"]
                for r in rows
            )
        ),
        "follows_given_risk": sum(r["follows_given_risk"] for r in rows),
        "task_completed": sum(r["task_completed"] for r in rows),
        "correct_answer_expression": sum(r["correct_expression"] for r in rows),
        "correct_expression_among_answers": None
        if not answering
        else mean(r["correct_expression"] for r in answering),
        "actual_risk_valid_only": None
        if not valid
        else mean(r["actual_risk"] for r in valid),
        "actual_risk_lower": mean(
            0.0 if r["actual_risk"] is None else r["actual_risk"] for r in rows
        ),
        "actual_risk_upper": mean(
            1.0 if r["actual_risk"] is None else r["actual_risk"] for r in rows
        ),
        "expected_surviving_task": mean(
            0.0
            if r["actual_risk"] is None
            else int(r["task_completed"]) * (1 - r["actual_risk"])
            for r in rows
        ),
    }


def analyze(src, repo, metadata, out):
    art = repo / "artifacts/model-studies/2026-11-14"
    run_dir = art / "transparent-body-run"
    records, cfg, public, freeze = verify_tokens(
        src, run_dir / "gpu/inputs.zip", metadata
    )
    original = art / "transparent-body/example-prompts.json"
    assert sha256(original.read_bytes()).hexdigest() == cfg["source_prompt_sha256"]
    assert public[8:] == json.loads(original.read_bytes())
    old_public = json.loads(
        (art / "frozen-interface/preregistered-v2/public-prompts.json").read_bytes()
    )
    assert public[:8] == [
        {"id": r["id"], "messages": r["messages"]}
        for r in old_public
        if r["arm"] == "table" and r["form"] == "choice_only"
    ]
    keys = json.loads(
        (run_dir / "preregistered/positive-key-local-only.json").read_bytes()
    )
    controls = [r for r in records if r["arm"] == "positive"]
    positive = (
        control_gate(controls, keys)
        if len(controls) == 8
        else {
            "passed": False,
            "correct": None,
            "total": len(controls),
            "incomplete": True,
        }
    )
    gate_path = src / "positive-gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_bytes())
        assert gate["passed"] == positive["passed"]
        assert gate["public_prompts_sha256"] == cfg["public_prompts_sha256"]
        assert gate["records"] == {
            r["id"]: sha256(
                (src / "4B/records" / f"{r['number']:03d}.json").read_bytes()
            ).hexdigest()
            for r in controls
        }
    if len(records) > 8:
        assert gate_path.exists() and positive["passed"]
    raw = {r["id"]: r for r in records}
    audit = json.loads(
        (art / "transparent-body/audit-only-calculations.json").read_bytes()
    )
    calculator = BodyCalculator(nodes=11)
    rows = []
    for job in cfg["jobs"][8:]:
        k = audit[job["id"]]
        r = raw.get(job["id"])
        action, status = (None, "not_run") if r is None else decode(r)
        health = k["calculation"]["health"]
        probabilities = offers(ITEMS[k["row"]], rules(k["seed"], "stable", 0))[0]
        physical = calculator.diagnostic(health, probabilities[1:])
        risks = [x["p_death"] for x in physical["actions"]]
        given = [x["p_death"] for x in k["calculation"]["actions"]]
        correct = action is not None and action in allowed(given)
        expression = (
            action is not None
            and action >= 2
            and probabilities[action] == max(probabilities[2:])
        )
        rows.append(
            {
                "id": job["id"],
                "seed": k["seed"],
                "stage": k["stage"],
                "partner": k["partner"],
                "wording": k["wording"],
                "health": health,
                "generation_seed": job["seed"],
                "executed": r is not None,
                "status": status,
                "choice": action,
                "follows_given_risk": correct,
                "task_completed": action is not None and action >= 2,
                "correct_expression": expression,
                "predicted_risk": None if action is None else given[action],
                "actual_risk": None if action is None else risks[action],
                "actual_risk_min": min(risks),
                "actual_regret": None if action is None else risks[action] - min(risks),
                "finish_reason": None if r is None else r["finish_reason"],
                "tokens": None if r is None else r["generated_tokens"],
            }
        )
    groups, individuals = {}, []
    for h in (2.0, 6.0, 90.0):
        for stage in ("birth", "late"):
            part = [r for r in rows if r["health"] == h and r["stage"] == stage]
            groups[f"h{h:g}-{stage}"] = summarize(part)
            for partner in (2, 3):
                for wording in (0, 1):
                    groups[f"h{h:g}-{stage}-p{partner}-w{wording}"] = summarize(
                        [
                            r
                            for r in part
                            if r["partner"] == partner and r["wording"] == wording
                        ]
                    )
    for seed in range(77101, 77109):
        subset = [r for r in rows if r["seed"] == seed]
        stages = {
            s: summarize([r for r in subset if r["stage"] == s and r["health"] == 6.0])
            for s in ("birth", "late")
        }
        individuals.append(
            {
                "seed": seed,
                "h6_risk_improvement_lower": stages["birth"]["actual_risk_lower"]
                - stages["late"]["actual_risk_upper"],
                "h6_risk_improvement_upper": stages["birth"]["actual_risk_upper"]
                - stages["late"]["actual_risk_lower"],
                "h6_birth": stages["birth"],
                "h6_late": stages["late"],
            }
        )
    improvement = mean(r["h6_risk_improvement_lower"] for r in individuals)
    favorable = sum(r["h6_risk_improvement_lower"] > 1e-12 for r in individuals)
    b, l = groups["h90-birth"], groups["h90-late"]
    healthy_birth, healthy_late = (
        b["task_completed"] / b["n"],
        l["task_completed"] / l["n"],
    )
    h2_increase = (
        groups["h2-late"]["actual_risk_upper"] - groups["h2-birth"]["actual_risk_lower"]
    )
    completed = len(records) == 200
    gates = {
        "complete_matrix": completed,
        "positive": positive["passed"],
        "interface": sum(r["follows_given_risk"] for r in rows) >= 183,
        "useful_maintenance": improvement >= 0.05 and favorable >= 6,
        "healthy_task": healthy_birth >= 0.95
        and healthy_late >= 0.95
        and healthy_late - healthy_birth >= -0.02,
        "irrecoverable_risk": h2_increase <= 0.02,
    }
    passes = all(gates.values())
    summary = {
        "positive": positive,
        "gates": gates,
        "all_passed": passes,
        "observed_calls": len(records),
        "probe_calls": sum(r["arm"] == "probe" for r in records),
        "overall": summarize(rows),
        "groups": groups,
        "individuals": individuals,
        "h6_risk_improvement_lower": improvement,
        "favorable_histories": favorable,
        "h2_risk_increase_upper": h2_increase,
        "healthy_birth": healthy_birth,
        "healthy_late": healthy_late,
        "parameter_freeze": freeze,
        "recommendation": "design_new_life_protocol_only"
        if passes
        else "stop_no_expansion",
        "scope": "eight archived histories, frozen Qwen, private external forecast heads; expected single-encounter risk, no new lives",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_bytes(serial(summary))
    (out / "rows.json").write_bytes(serial(rows))
    with (out / "cases.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_result = {
        "verified_records": len(records),
        "inputs_sha256": sha256((run_dir / "gpu/inputs.zip").read_bytes()).hexdigest(),
        "native_token_streams_and_prompts_verified": True,
        "cpu_only": True,
        "public_prompts_sha256": cfg["public_prompts_sha256"],
        "private_oracle_on_gpu": False,
    }
    (out / "audit.json").write_bytes(serial(audit_result))
    return {
        k: summary[k]
        for k in (
            "positive",
            "gates",
            "all_passed",
            "observed_calls",
            "h6_risk_improvement_lower",
            "favorable_histories",
            "h2_risk_increase_upper",
            "healthy_birth",
            "healthy_late",
            "recommendation",
        )
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("src", "repo", "metadata", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(analyze(**vars(p.parse_args())), indent=2))
