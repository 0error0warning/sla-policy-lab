"""CPU verification and separately scored predictions, decisions and completion."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import statistics
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from .design import fixtures, score, serial
from .scoring import assistant_message
from .wire import final_tokens, strict_json


def classify(record, key):
    stopped = record["finish_reason"]
    final = record.get("final_text")
    obj, failure = None, None
    if stopped.startswith("unrun"):
        failure = "unrun"
    elif final is None:
        failure = "no_final_channel"
    else:
        try:
            obj = strict_json(final)
        except (ValueError, TypeError):
            failure = "invalid_final_json"
    result = score(obj, key)
    risk_correct = (
        result["choice_valid"] and result["death_probability_regret"] <= 1e-12
    )
    result.update(
        risk_minimizing_choice=risk_correct,
        higher_risk_choice=result["choice_valid"] and not risk_correct,
        wrong_only_task_tiebreak=risk_correct and not result["choice_correct"],
        executed=not stopped.startswith("unrun"),
        final_json_parsed=failure is None,
        no_final_channel=failure == "no_final_channel",
        invalid_final_json=failure == "invalid_final_json",
        invalid_schema=failure is None and not result["schema"],
        calculation_wrong=result["schema"] and not result["calculation_correct"],
        choice_wrong=result["choice_valid"] and not result["choice_correct"],
        budget_stop=stopped in ("time_limit", "length_limit"),
        unfinished=stopped in ("time_limit", "length_limit") and failure is not None,
        healthy_task_complete=key["health"] >= 12
        and result["answering"]
        and result["task_answer_correct"],
        failure=failure,
    )
    return result, obj


def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["state"]].append(row)
    assert all(len(v) == 2 for v in pairs.values())
    healthy = [r for r in rows if r["health"] >= 12]
    report = {
        "presentations": len(rows),
        "physical_states": len(pairs),
        "counts": {
            k: sum(bool(r[k]) for r in rows)
            for k in (
                "executed",
                "final_json_parsed",
                "schema",
                "calculation_correct",
                "choice_valid",
                "choice_correct",
                "no_final_channel",
                "invalid_final_json",
                "invalid_schema",
                "calculation_wrong",
                "choice_wrong",
                "risk_minimizing_choice",
                "higher_risk_choice",
                "wrong_only_task_tiebreak",
                "budget_stop",
                "unfinished",
            )
        },
        "stop_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "paired_choice_correct": sum(
            all(r["choice_correct"] for r in v) for v in pairs.values()
        ),
        "paired_calculation_correct": sum(
            all(r["calculation_correct"] for r in v) for v in pairs.values()
        ),
        "healthy_presentations": len(healthy),
        "healthy_executed": sum(r["executed"] for r in healthy),
        "healthy_valid_action": sum(r["choice_valid"] for r in healthy),
        "healthy_task_complete": sum(r["healthy_task_complete"] for r in healthy),
        "healthy_answering": sum(r["answering"] for r in healthy),
        "mean_actual_death_probability_observed_choices": average(
            r["actual_death_probability"] for r in rows
        ),
        "mean_risk_regret_observed_choices": average(
            r["death_probability_regret"] for r in rows
        ),
        "mean_tokens": average(
            r.get("generated_tokens") for r in rows if r["executed"]
        ),
        "mean_seconds": average(
            r.get("elapsed_seconds") for r in rows if r["executed"]
        ),
        "choice_counts": dict(Counter(r["choice"] for r in rows if r["choice_valid"])),
        "boundary_parsing": {
            "corrected_records": sum(r["parser_boundary_corrected"] for r in rows),
            "original_valid_json": sum(r["legacy_final_json_parsed"] for r in rows),
            "original_valid_action": sum(r["legacy_choice_valid"] for r in rows),
            "original_correct_choice": sum(r["legacy_choice_correct"] for r in rows),
            "original_correct_calculation": sum(
                r["legacy_calculation_correct"] for r in rows
            ),
            "tokens_after_message_end": sum(r["tokens_after_chat_end"] for r in rows),
            "max_tokens_after_message_end": max(
                r["tokens_after_chat_end"] for r in rows
            ),
        },
        "survival_classification_correct": sum(
            r.get("survival_classification_correct", False) for r in rows
        ),
    }
    valid_choices = report["counts"]["choice_valid"]
    valid_schema = report["counts"]["schema"]
    report["budget_stop_fraction_executed"] = (
        report["counts"]["budget_stop"] / report["counts"]["executed"]
        if report["counts"]["executed"]
        else None
    )
    report["choice_accuracy_given_valid_action"] = (
        report["counts"]["choice_correct"] / valid_choices if valid_choices else None
    )
    report["calculation_accuracy_given_valid_schema"] = (
        report["counts"]["calculation_correct"] / valid_schema if valid_schema else None
    )
    report["by_health"] = {
        str(h): {
            "n": len(v),
            "choice_correct": sum(r["choice_correct"] for r in v),
            "calculation_correct": sum(r["calculation_correct"] for r in v),
            "no_final_or_invalid_action": sum(not r["choice_valid"] for r in v),
            "risk_regret": average(r["death_probability_regret"] for r in v),
            "choices": dict(Counter(r["choice"] for r in v if r["choice_valid"])),
        }
        for h in sorted({r["health"] for r in rows})
        for v in [[r for r in rows if r["health"] == h]]
    }
    report["by_regime"] = {}
    for label, healths in (
        ("answer_fatal_even_with_maintenance", (2.0, 3.35)),
        ("maintenance_can_save_answer", (3.65, 6.0)),
        ("wait_can_survive_without_maintenance", (8.25,)),
        ("healthy_all_actions_survive", (12.0, 90.0, 300.0)),
    ):
        group = [r for r in rows if r["health"] in healths]
        report["by_regime"][label] = {
            "presentations": len(group),
            **{
                metric: sum(r[metric] for r in group)
                for metric in (
                    "executed",
                    "choice_valid",
                    "choice_correct",
                    "risk_minimizing_choice",
                    "higher_risk_choice",
                    "wrong_only_task_tiebreak",
                    "healthy_task_complete",
                )
            },
            "mean_actual_death_probability_observed_choices": average(
                r["actual_death_probability"] for r in group
            ),
            "mean_risk_regret_observed_choices": average(
                r["death_probability_regret"] for r in group
            ),
        }
    return report


def run(src, inputs, metadata, out, segments=None):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    out.mkdir(parents=True, exist_ok=True)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(digest(z.read(n)) == sha for n, sha in lock["files"].items())
        public = json.loads(z.read("public-prompts.json"))
        config = json.loads(z.read("execution.json"))
        expect = {m: json.loads(z.read(f"expected-{m}.json")) for m in config["models"]}
    rebuilt, keys = fixtures()
    assert public == rebuilt
    assert digest(serial(public)) == config["public_prompts_sha256"]
    lookup = {r["id"]: r for r in public}
    recovered, segment_receipts = None, None
    if segments:
        from .recovery import load

        recovered, segment_receipts = load(segments, inputs)
    else:
        assert json.loads((src / "execution.json").read_bytes()) == config
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def throw(message):
        raise ValueError(message)

    env.globals["raise_exception"] = throw
    records, parsed, verified = [], [], 0
    for model in config["models"]:
        if recovered is None:
            assert (
                json.loads((src / model / "weights-manifest.json").read_bytes())
                == expect[model]
            )
            unchanged = json.loads((src / model / "unchanged.json").read_bytes())
            assert (
                unchanged["parameter_versions_unchanged"]
                and unchanged["gradients_absent"]
                and unchanged["trainable_parameters"] == 0
            )
            meta = json.loads((src / model / "loading.json").read_bytes())
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            assert (
                digest((metadata / model / name).read_bytes())
                == expect[model]["metadata"][name]["sha256"]
            )
        tok = Tokenizer.from_file(str(metadata / model / "tokenizer.json"))
        tok_config = json.loads(
            (metadata / model / "tokenizer_config.json").read_bytes()
        )
        assert not tok_config.get("clean_up_tokenization_spaces", False)
        template = env.from_string(
            (metadata / model / "chat_template.jinja").read_text(encoding="utf8")
        )
        paths = None
        if recovered is None:
            paths = sorted((src / model / "records").glob("*.json"))
            assert len(paths) == 128
        for index, job in enumerate(config["jobs"]):
            provenance = None
            if recovered is None:
                record = json.loads(paths[index].read_bytes())
            elif (model, job["number"]) in recovered:
                origin = recovered[(model, job["number"])]
                record, meta = origin["record"].copy(), origin["meta"]
                provenance = {
                    k: v for k, v in origin.items() if k not in ("record", "meta")
                }
            else:
                public_row = lookup[job["id"]]
                record = {
                    **job,
                    "model": model,
                    **{k: public_row[k] for k in ("arm", "state", "order")},
                    "finish_reason": "unrun_no_retrieved_record",
                    "generated_tokens": 0,
                }
            assert all(record[k] == value for k, value in job.items())
            assert record["model"] == model
            row = lookup[job["id"]]
            assert all(record[k] == row[k] for k in ("arm", "state", "order"))
            if not record["finish_reason"].startswith("unrun"):
                prompt = template.render(
                    messages=row["messages"],
                    tools=None,
                    add_generation_prompt=True,
                    enable_thinking=job["thinking"],
                )
                assert record["prompt_text"] == prompt
                assert (
                    tok.encode(prompt, add_special_tokens=False).ids
                    == record["input_ids"]
                )
                assert (
                    tok.decode(record["output_ids"], skip_special_tokens=False)
                    == record["output_text"]
                )
                final, channel, thoughts = final_tokens(
                    record["output_ids"],
                    job["thinking"],
                    meta["close_id"],
                    meta["open_id"],
                    meta["eos_ids"],
                )
                assert (final, channel, thoughts) == (
                    record["final_ids"],
                    record["channel"],
                    record["thinking_tokens"],
                )
                text = (
                    None
                    if final is None
                    else tok.decode(final, skip_special_tokens=False)
                )
                assert text == record["final_text"]
                assert (
                    len(record["output_ids"])
                    == record["generated_tokens"]
                    <= config["max_new_tokens"]
                )
                assert (
                    record["max_new_tokens"] == config["max_new_tokens"]
                    and record["max_seconds"] == config["per_case_seconds"]
                )
                for k, v in config["sampler"].items():
                    assert (
                        record[k]
                        if k == "presence_penalty"
                        else record["generation_config"][k]
                    ) == v
                verified += 1
            legacy = record.get("final_text")
            legacy_scored, _ = classify(record, keys[job["id"]])
            chat_position = None
            if "output_ids" in record:
                correct_ids, _, _, chat_position = assistant_message(
                    record["output_ids"],
                    job["thinking"],
                    meta["close_id"],
                    meta["open_id"],
                    meta["eos_ids"],
                    tok.token_to_id("<|im_end|>"),
                )
                record["final_text"] = (
                    None
                    if correct_ids is None
                    else tok.decode(correct_ids, skip_special_tokens=False)
                )
            scored, obj = classify(record, keys[job["id"]])
            simple = {
                k: record.get(k)
                for k in (
                    "number",
                    "id",
                    "model",
                    "thinking",
                    "arm",
                    "state",
                    "finish_reason",
                    "generated_tokens",
                    "elapsed_seconds",
                    "thinking_tokens",
                    "peak_allocated_bytes",
                )
            }
            simple.update(
                health=keys[job["id"]]["health"],
                favorite=keys[job["id"]]["favorite"],
                choice=obj.get("choice") if isinstance(obj, dict) else None,
                **scored,
            )
            simple["source_segment"] = provenance["segment"] if provenance else None
            simple["end_freeze_receipt_available"] = (
                provenance["freeze_receipt_available"]
                if provenance
                else recovered is None
            )
            simple["chat_end_reached"] = chat_position is not None
            simple["tokens_after_chat_end"] = (
                len(record.get("output_ids", [])) - chat_position - 1
                if chat_position is not None
                else 0
            )
            simple["parser_boundary_corrected"] = legacy != record.get("final_text")
            for metric in (
                "final_json_parsed",
                "choice_valid",
                "choice_correct",
                "calculation_correct",
            ):
                simple[f"legacy_{metric}"] = legacy_scored[metric]
            records.append(simple)
            parsed.append(
                {
                    **simple,
                    "prediction": obj,
                    "expected": keys[job["id"]],
                    "final_text": record.get("final_text"),
                    "original_recorded_final_text": legacy,
                    "chat_end_position": chat_position,
                    "provenance": provenance,
                }
            )
    grouped = defaultdict(list)
    for r in records:
        grouped[
            f"{r['model']}/{'thinking' if r['thinking'] else 'direct'}/{r['arm']}"
        ].append(r)
    summary = {name: aggregate(rows) for name, rows in grouped.items()}
    sensitivity = []
    for row in parsed:
        if not row["invalid_final_json"]:
            continue
        match = re.fullmatch(
            r"```json\r?\n(.*)\r?\n```", row["final_text"].strip(), re.DOTALL
        )
        if match:
            try:
                relaxed = strict_json(match.group(1))
            except ValueError:
                continue
            sensitivity.append(
                {
                    "model": row["model"],
                    "id": row["id"],
                    "thinking": row["thinking"],
                    "strict_failure": row["failure"],
                    "without_markdown_fence": score(relaxed, row["expected"]),
                }
            )
    (out / "format-sensitivity.json").write_bytes(
        serial(
            {
                "status": "Post-hoc format-only sensitivity; primary counts and gates unchanged.",
                "rule": "Only an entire single json fence in the verified final channel; strict JSON still required.",
                "cases": sensitivity,
            }
        )
    )
    comparison = {}
    for mode in ("direct", "thinking"):
        for arm in ("known_rules", "given_outcomes"):
            a, b = (summary[f"{m}/{mode}/{arm}"] for m in ("4B", "9B"))
            comparison[f"{mode}/{arm}"] = {
                "9B_minus_4B_paired_choice": b["paired_choice_correct"]
                - a["paired_choice_correct"],
                "9B_minus_4B_paired_calculation": b["paired_calculation_correct"]
                - a["paired_calculation_correct"],
                "9B_minus_4B_healthy_task": b["healthy_task_complete"]
                - a["healthy_task_complete"],
            }
            groups = [
                {r["id"]: r for r in grouped[f"{m}/{mode}/{arm}"]} for m in ("4B", "9B")
            ]
            common = [
                (groups[0][ident], groups[1][ident])
                for ident in groups[0]
                if groups[0][ident]["choice_valid"] and groups[1][ident]["choice_valid"]
            ]
            same_executed = [
                (groups[0][ident], groups[1][ident])
                for ident in groups[0]
                if groups[0][ident]["executed"] and groups[1][ident]["executed"]
            ]
            comparison[f"{mode}/{arm}"].update(
                both_executed_presentations=len(same_executed),
                on_common_executed={
                    model: {
                        metric: sum(pair[index][metric] for pair in same_executed)
                        for metric in (
                            "choice_valid",
                            "choice_correct",
                            "schema",
                            "calculation_correct",
                            "healthy_task_complete",
                            "no_final_channel",
                            "invalid_final_json",
                        )
                    }
                    for index, model in enumerate(("4B", "9B"))
                },
                both_valid_action_presentations=len(common),
                common_valid_4B_correct=sum(x["choice_correct"] for x, _ in common),
                common_valid_9B_correct=sum(y["choice_correct"] for _, y in common),
                common_valid_9B_better=sum(
                    y["choice_correct"] and not x["choice_correct"] for x, y in common
                ),
                common_valid_4B_better=sum(
                    x["choice_correct"] and not y["choice_correct"] for x, y in common
                ),
                caution="Completed-case comparison is conditional on both outputs being usable; unrun and unfinished cases are not computational errors.",
            )
    gates = {}
    for model in ("4B", "9B"):
        for mode in ("direct", "thinking"):
            a, b = (
                summary[f"{model}/{mode}/{arm}"]
                for arm in ("known_rules", "given_outcomes")
            )
            gates[f"{model}/{mode}"] = {
                "known_rules_choice": a["paired_choice_correct"] >= 14,
                "given_outcomes_choice": b["paired_choice_correct"] >= 14,
                "known_rules_calculation": a["paired_calculation_correct"] >= 14,
                "healthy_task": a["healthy_task_complete"] >= 11
                and b["healthy_task_complete"] >= 11,
                "planned_stop_count_within_limit": a["counts"]["budget_stop"]
                + b["counts"]["budget_stop"]
                <= 0.2 * 64,
                "budget_valid": a["counts"]["executed"] + b["counts"]["executed"] == 64
                and a["counts"]["budget_stop"] + b["counts"]["budget_stop"] <= 0.2 * 64,
                "all_executed": a["counts"]["executed"] + b["counts"]["executed"] == 64,
            }
            gates[f"{model}/{mode}"]["joint_pass"] = all(
                gates[f"{model}/{mode}"].values()
            )
    with (out / "cases.csv").open("w", encoding="utf8", newline="") as f:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with gzip.open(out / "parsed-cases.jsonl.gz", "wt", encoding="utf8") as f:
        for row in parsed:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    for name, obj in (
        ("summary", summary),
        ("comparison", comparison),
        ("gates", gates),
    ):
        (out / f"{name}.json").write_bytes(serial(obj))
    audit = {
        "cpu_environment": {
            "python": platform.python_version(),
            "tokenizers": version("tokenizers"),
            "Jinja2": version("Jinja2"),
        },
        "records": len(records),
        "executed_token_roundtrips": verified,
        "execution_count_semantics": "Executed counts archived non-placeholder generations. Missing output after runtime loss does not prove that a call was never executed.",
        "input_sha256": digest(inputs.read_bytes()),
        "rebuilt_physics_verified": True,
        "frozen_parameters_verified": all(
            r["end_freeze_receipt_available"] for r in records if r["executed"]
        ),
        "trainable_parameters_zero_at_loading": True,
        "segment_receipts": segment_receipts,
        "same_prompts_seeds_per_case_budgets_verified": True,
        "distinct_physical_states": 16,
        "independent_lives": 0,
        "order_caution": "One sampled output per presentation; paired orders also have different decoding seeds, so changes are not a pure order intervention.",
    }
    (out / "audit.json").write_bytes(serial(audit))
    return {"audit": audit, "summary": summary, "gates": gates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("src", "inputs", "metadata", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--segments", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.src, args.inputs, args.metadata, args.out, args.segments), indent=2
        )
    )
