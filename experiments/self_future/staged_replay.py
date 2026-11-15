"""CPU replay of every subcall and literal assembly in the staged comparison."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from experiments.frozen_capability.replay import unpack
from experiments.frozen_capability.scoring import assistant_message

from .design import serial
from .score import strict_json, summarize
from .staged import STAGE_CAPS, assemble, messages


def read_components(outputs):
    """Preserve assessable choices even when a different stage has invalid JSON."""
    plans = {}
    for label in ("P", "Q"):
        try:
            plans[label] = strict_json(outputs[label])
        except (ValueError, TypeError, KeyError):
            plans[label] = {"unparsed_model_text": outputs.get(label)}
    try:
        choice = strict_json(outputs["choice"])["choice"]
    except (ValueError, TypeError, KeyError):
        choice = None
    return {"plans": plans, "choice": choice}


def substantive_forecast(record, key):
    """Post-hoc diagnostic: ignore only post-terminal 0 versus null, not costs."""
    try:
        obj = strict_json(record["final_text"])
        for label, expected in key["plans"].items():
            row = obj["plans"][label]
            if (
                row["terminal_step"] != expected["terminal_step"]
                or row["can_act_after_plan"] != expected["can_act_after_plan"]
            ):
                return False
            if len(row["health"]) != len(expected["health"]):
                return False
            for actual, truth in zip(row["health"], expected["health"]):
                if truth is not None and (type(actual) is not int or actual != truth):
                    return False
        return True
    except (ValueError, TypeError, KeyError):
        return False


def replay(inputs, archive, metadata, key_path, work, out):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    def sha(payload):
        return hashlib.sha256(payload).hexdigest()

    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(sha(z.read(n)) == expected for n, expected in lock["files"].items())
        assert not any(
            n.endswith(("design.py", "score.py")) or "evaluator-only" in n
            for n in z.namelist()
        )
        assert z.read("experiments/self_future/staged.py") == Path(__file__).with_name(
            "staged.py"
        ).read_bytes().replace(b"\r\n", b"\n")
        cfg = json.loads(z.read("execution.json"))
        public_bytes = z.read("public-prompts.json")
        assert sha(public_bytes) == cfg["public_prompts_sha256"]
        public = {row["id"]: row for row in json.loads(public_bytes)}
        expected = json.loads(z.read("expected-4B.json"))
    unpack(archive, work)
    src = work / "output"
    assert json.loads((src / "execution.json").read_bytes()) == cfg
    assert json.loads((src / "4B/weights-manifest.json").read_bytes()) == expected
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (
            sha((metadata / "4B" / name).read_bytes())
            == expected["metadata"][name]["sha256"]
        )
    tok = Tokenizer.from_file(str(metadata / "4B/tokenizer.json"))
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def throw(message):
        raise ValueError(message)

    env.globals["raise_exception"] = throw
    template = env.from_string(
        (metadata / "4B/chat_template.jinja").read_text(encoding="utf8")
    )
    meta = json.loads((src / "4B/loading.json").read_bytes())
    frozen = json.loads((src / "4B/unchanged.json").read_bytes())
    assert frozen["parameter_versions_unchanged"] and frozen["gradients_absent"]
    assert frozen["trainable_parameters"] == 0

    def audit_raw(record, prompt, cap):
        rendered = template.render(
            messages=prompt,
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert rendered == record["prompt_text"]
        assert tok.encode(rendered, add_special_tokens=False).ids == record["input_ids"]
        assert (
            tok.decode(record["output_ids"], skip_special_tokens=False)
            == record["output_text"]
        )
        final, channel, thoughts, boundary = assistant_message(
            record["output_ids"],
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
            record["final_ids"],
            record["channel"],
            record["thinking_tokens"],
            record["message_end_position"],
            record["final_text"],
        )
        assert len(record["output_ids"]) == record["generated_tokens"] <= cap
        assert (
            record["max_new_tokens"] == cap
            and record["generation_config"]["do_sample"] is False
        )
        if record["completion_status"] == "completed":
            assert record["output_ids"][-1] in meta["eos_ids"] and boundary is not None
            assert record["finish_reason"] == "eos"

    keys = json.loads(key_path.read_bytes())
    records, raw_calls, startups, effects = {"monolithic": [], "staged": []}, [], [], []
    for job in cfg["jobs"]:
        path = src / "4B/records" / f"{job['number']:03d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_bytes())
        assert all(record[k] == value for k, value in job.items())
        original = public[job["id"]]
        if job["arm"] == "monolithic":
            audit_raw(record, original["messages"], 1024)
            raw_calls.append(record)
        else:
            outputs, calls = {}, []
            for stage, cap in STAGE_CAPS.items():
                path = src / "4B/stages" / f"{job['number']:03d}-{stage}.json"
                if not path.exists():
                    break
                raw = json.loads(path.read_bytes())
                previous = (
                    outputs
                    if stage == "choice"
                    else {k: v for k, v in outputs.items() if k == "effects"}
                )
                prompt = messages(original, stage, previous)
                assert raw["messages"] == prompt and raw["stage"] == stage
                audit_raw(raw, prompt, cap)
                outputs[stage] = raw["final_text"]
                calls.append(raw)
                raw_calls.append(
                    {**raw, "number": job["number"], "split": job["split"]}
                )
            assert outputs == record["stages"]
            assert len(calls) == record["calls"]
            assert (
                sum(r["generated_tokens"] for r in calls)
                == record["generated_tokens"]
                <= 1024
            )
            if record["completion_status"] == "completed":
                assert len(calls) == 4 and all(
                    r["completion_status"] == "completed" for r in calls
                )
                assert strict_json(record["final_text"]) == assemble(outputs)
            try:
                extracted = strict_json(outputs["effects"])
                correct = extracted == keys[job["id"]]["case"]["effects"]
            except (ValueError, TypeError, KeyError):
                extracted, correct = None, False
            effects.append(
                {"id": job["id"], "extracted": extracted, "correct": correct}
            )
            if record["completion_status"] == "completed":
                record = {
                    **record,
                    "worker_assembled_text": record["final_text"],
                    "final_text": json.dumps(read_components(outputs)),
                    "final_text_source": "independent_read_of_model_subcalls",
                }
        if job["split"] == "startup":
            truth = (
                {"ready": True}
                if job["id"] == "startup-1"
                else {"check": {"values": [7, 0, None], "flag": False}}
            )
            startups.append(
                {
                    "id": job["id"],
                    "passed": record["completion_status"] == "completed"
                    and strict_json(record["final_text"]) == truth,
                }
            )
        else:
            records[job["arm"]].append(record)
    result = {
        "startup": startups,
        "raw_generation_calls": len(raw_calls),
        "generated_tokens": sum(r["generated_tokens"] for r in raw_calls),
        "generation_seconds": sum(r["elapsed_seconds"] for r in raw_calls),
        "all_tokens_templates_and_assembly_verified": True,
        "weights_frozen": frozen,
        "input_sha256": sha(inputs.read_bytes()),
        "archive_sha256": sha(archive.read_bytes()),
        "effects": effects,
        "arms": {arm: summarize(rows, keys) for arm, rows in records.items()},
        "supplementary_substantive_forecasts": {
            arm: {row["id"]: substantive_forecast(row, keys[row["id"]]) for row in rows}
            for arm, rows in records.items()
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_bytes(serial(result))
    for arm, rows in records.items():
        (out / (arm + ".jsonl")).write_bytes(
            b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "archive", "metadata", "key", "work", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.inputs, args.archive, args.metadata, args.key, args.work, args.out
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "raw_generation_calls",
                    "generated_tokens",
                    "generation_seconds",
                    "all_tokens_templates_and_assembly_verified",
                )
            },
            indent=2,
        )
    )
    for arm, summary in result["arms"].items():
        print(
            arm,
            json.dumps(
                {
                    k: {
                        m: v["counts"][m]
                        for m in (
                            "completed",
                            "health_trace_correct",
                            "terminal_correct",
                            "task_counts_correct",
                            "choice_correct",
                            "joint_success",
                        )
                    }
                    for k, v in summary["groups"].items()
                }
            ),
        )
