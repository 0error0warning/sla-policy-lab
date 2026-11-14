"""Replay raw token outputs and separate sorting, intersection and rule errors."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from experiments.frozen_capability.design import serial
from experiments.frozen_capability.scoring import assistant_message
from experiments.frozen_capability.wire import strict_json

from .contracts import projected_seconds
from .design import fixtures
from .score import score
from .structural_check import secondary_score


def classify(record, key):
    obj, failure = None, None
    if record is None:
        failure = "missing_record"
    elif record.get("final_text") is None:
        failure = "no_final_channel"
    else:
        try:
            obj = strict_json(record["final_text"])
        except (ValueError, TypeError):
            failure = "invalid_final_json"
    result = score(obj, key)
    result.update(
        executed=record is not None,
        failure=failure,
        final_json_parsed=failure is None,
        no_final_channel=failure == "no_final_channel",
        invalid_final_json=failure == "invalid_final_json",
        invalid_schema=failure is None and not result["protocol_valid"],
        calculation_wrong=result.get("numeric_schema", False)
        and not result.get("calculation_correct", False),
        choice_wrong=result["choice_valid"] and not result["choice_correct"],
        higher_risk_choice=result["choice_valid"] and result["risk_regret"] > 1e-12,
        wrong_only_task_tiebreak=result["choice_valid"]
        and result["risk_regret"] <= 1e-12
        and not result["choice_correct"],
    )
    return result, obj


def run(src, inputs, metadata, out):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    out.mkdir(parents=True, exist_ok=True)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(sha(z.read(n)) == h for n, h in lock["files"].items())
        public = json.loads(z.read("public-prompts.json"))
        cfg = json.loads(z.read("execution.json"))
        expected = {m: json.loads(z.read(f"expected-{m}.json")) for m in cfg["models"]}
        assert not any(
            "answer-key" in n or n.endswith(("design.py", "score.py"))
            for n in z.namelist()
        )
    rebuilt, keys = fixtures()
    assert public == rebuilt and sha(serial(public)) == cfg["public_prompts_sha256"]
    lookup = {p["id"]: p for p in public}
    assert json.loads((src / "execution.json").read_bytes()) == cfg
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def throw(message):
        raise ValueError(message)

    env.globals["raise_exception"] = throw
    rows, details, freezes, startup = [], [], {}, {}
    for model in cfg["models"]:
        meta_path = src / model / "loading.json"
        meta = json.loads(meta_path.read_bytes()) if meta_path.exists() else None
        freeze_path = src / model / "unchanged.json"
        freezes[model] = (
            json.loads(freeze_path.read_bytes()) if freeze_path.exists() else None
        )
        if freezes[model]:
            assert (
                freezes[model]["parameter_versions_unchanged"]
                and freezes[model]["gradients_absent"]
                and freezes[model]["trainable_parameters"] == 0
            )
        manifest = src / model / "weights-manifest.json"
        if manifest.exists():
            assert json.loads(manifest.read_bytes()) == expected[model]
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            assert (
                sha((metadata / model / name).read_bytes())
                == expected[model]["metadata"][name]["sha256"]
            )
        tok = Tokenizer.from_file(str(metadata / model / "tokenizer.json"))
        template = env.from_string(
            (metadata / model / "chat_template.jinja").read_text(encoding="utf8")
        )
        startup[model] = {}
        for name in (
            "canary-False.json",
            "canary-True.json",
            "speed-preflight.json",
            "cache-preflight.json",
            "preflight-summary.json",
        ):
            path = src / model / name
            if not path.exists():
                continue
            value = json.loads(path.read_bytes())
            if name.startswith("canary"):
                assert meta is not None
                fin, chan, thought, boundary = assistant_message(
                    value["output_ids"],
                    "True" in name,
                    meta["close_id"],
                    meta["open_id"],
                    meta["eos_ids"],
                    meta["chat_end_id"],
                )
                decoded = (
                    None if fin is None else tok.decode(fin, skip_special_tokens=False)
                )
                assert (fin, chan, thought, boundary, decoded) == (
                    value["final_ids"],
                    value["channel"],
                    value["thinking_tokens"],
                    value["message_end_position"],
                    value["final_text"],
                )
                try:
                    valid = strict_json(decoded) == {"ready": True}
                except (ValueError, TypeError):
                    valid = False
                startup[model][name] = {
                    "valid": valid,
                    "stop": value["finish_reason"],
                    "tokens": value["generated_tokens"],
                }
            elif name == "speed-preflight.json":
                startup[model][name] = {
                    "tokens": value["generated_tokens"],
                    "seconds": value["elapsed_seconds"],
                    "projected_seconds": projected_seconds(
                        64,
                        value["first_64_seconds"],
                        448,
                        value["elapsed_seconds"] - value["first_64_seconds"],
                    ),
                }
            else:
                startup[model][name] = value
        for job in cfg["jobs"]:
            prompt = lookup[job["id"]]
            path = src / model / "records" / f"{job['number']:03d}.json"
            record = json.loads(path.read_bytes()) if path.exists() else None
            if record is not None:
                assert meta is not None and all(record[k] == v for k, v in job.items())
                assert record["model"] == model
                assert all(
                    record[k] == prompt[k] for k in ("arm", "form", "state", "order")
                )
                text = template.render(
                    messages=prompt["messages"],
                    tools=None,
                    add_generation_prompt=True,
                    enable_thinking=job["thinking"],
                )
                assert text == record["prompt_text"]
                assert (
                    tok.encode(text, add_special_tokens=False).ids
                    == record["input_ids"]
                )
                assert (
                    tok.decode(record["output_ids"], skip_special_tokens=False)
                    == record["output_text"]
                )
                fin, chan, thought, boundary = assistant_message(
                    record["output_ids"],
                    job["thinking"],
                    meta["close_id"],
                    meta["open_id"],
                    meta["eos_ids"],
                    meta["chat_end_id"],
                )
                decoded = (
                    None if fin is None else tok.decode(fin, skip_special_tokens=False)
                )
                assert (fin, chan, thought, boundary, decoded) == (
                    record["final_ids"],
                    record["channel"],
                    record["thinking_tokens"],
                    record["message_end_position"],
                    record["final_text"],
                )
                assert (
                    len(record["output_ids"])
                    == record["generated_tokens"]
                    <= cfg["max_new_tokens"]
                )
                assert (
                    record["max_new_tokens"] == cfg["max_new_tokens"]
                    and record["max_seconds"] == cfg["per_case_seconds"]
                )
                for k, v in cfg["sampler"].items():
                    assert (
                        record[k]
                        if k == "presence_penalty"
                        else record["generation_config"][k]
                    ) == v
                assert (
                    meta["chat_end_id"] in record["generation_config"]["eos_token_id"]
                )
            scored, obj = classify(record, keys[job["id"]])
            secondary = secondary_score(obj, keys[job["id"]])
            simple = {
                **job,
                "model": model,
                "arm": prompt["arm"],
                "form": prompt["form"],
                "state": prompt["state"],
                "health": keys[job["id"]].get("health"),
                "choice": obj.get("choice") if isinstance(obj, dict) else None,
                **scored,
            }
            for name in (
                "finish_reason",
                "generated_tokens",
                "elapsed_seconds",
                "thinking_tokens",
                "peak_allocated_bytes",
            ):
                simple[name] = record.get(name) if record else None
            if secondary is not None:
                simple["secondary_actions_encoding"] = secondary["encoding"]
                for metric in (
                    "numeric_schema",
                    "calculation_correct",
                    "after_action_correct",
                    "endpoints_correct",
                    "joint_success",
                    "choice_consistent_with_own_risks",
                ):
                    simple["secondary_" + metric] = (secondary["score"] or {}).get(
                        metric, False
                    )
            rows.append(simple)
            details.append(
                {
                    **simple,
                    "parsed": obj,
                    "secondary_structural_check": secondary,
                    "reference": keys[job["id"]],
                    "raw_path": f"output/{model}/records/{job['number']:03d}.json"
                    if record
                    else None,
                }
            )
    metrics = (
        "executed",
        "final_json_parsed",
        "protocol_valid",
        "choice_correct",
        "completed_task",
        "min_set_correct",
        "task_set_correct",
        "intersection_consistent",
        "choice_consistent_with_sets",
        "sets_consistent_with_own_risks",
        "choice_consistent_with_own_risks",
        "calculation_correct",
        "after_action_correct",
        "endpoints_correct",
        "calculation_wrong",
        "choice_wrong",
        "higher_risk_choice",
        "wrong_only_task_tiebreak",
        "no_final_channel",
        "invalid_final_json",
        "invalid_schema",
        "joint_success",
        "secondary_calculation_correct",
        "secondary_after_action_correct",
        "secondary_endpoints_correct",
        "secondary_joint_success",
    )
    summary = []
    for model in cfg["models"]:
        for arm, form, thinking in (
            ("table", "choice_only", False),
            ("table", "sets", False),
            ("body", "sets", False),
            ("body", "sets", True),
        ):
            group = [
                r
                for r in rows
                if (r["model"], r["arm"], r["form"], r["thinking"])
                == (model, arm, form, thinking)
            ]
            executed = [r for r in group if r["executed"]]
            healthy = [r for r in group if r["health"] == 90]
            counts = {m: sum(bool(r.get(m)) for r in group) for m in metrics}
            summary.append(
                {
                    "model": model,
                    "arm": arm,
                    "form": form,
                    "thinking": thinking,
                    "planned": len(group),
                    "counts": counts,
                    "stops": dict(Counter(r["finish_reason"] for r in group)),
                    "healthy_completed": sum(r["completed_task"] for r in healthy),
                    "healthy_planned": len(healthy),
                    "mean_seconds": mean(r["elapsed_seconds"] for r in executed)
                    if executed
                    else None,
                    "mean_tokens": mean(r["generated_tokens"] for r in executed)
                    if executed
                    else None,
                    "max_gpu_allocated_bytes": max(
                        (r["peak_allocated_bytes"] for r in executed), default=None
                    ),
                    "passed": bool(group)
                    and counts["joint_success"] == len(group)
                    and all(
                        r["finish_reason"] not in ("time_limit", None) for r in group
                    ),
                }
            )
    columns = sorted({k for row in rows for k in row})
    with (out / "cases.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(out / "parsed-cases.jsonl.gz", "wt", encoding="utf8") as f:
        for value in details:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "summary.json").write_bytes(serial(summary))
    audit = {
        "planned": len(rows),
        "archived": sum(r["executed"] for r in rows),
        "all_records_verified": True,
        "all_tokens_and_templates_verified": True,
        "freeze_receipts": freezes,
        "startup": startup,
        "preflight_pass_file_exists": (src / "preflight-passed.json").exists(),
        "source_input_sha256": sha(inputs.read_bytes()),
    }
    if any(r["executed"] for r in rows):
        assert audit["preflight_pass_file_exists"]
    (out / "audit.json").write_bytes(serial(audit))
    return {"counts": audit["archived"], "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("src", "inputs", "metadata", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.src, args.inputs, args.metadata, args.out), indent=2))
