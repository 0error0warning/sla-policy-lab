"""Audit frozen GPU tokens and independently score a recorded self-future run."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from experiments.frozen_capability.replay import unpack
from experiments.frozen_capability.scoring import assistant_message

from .design import serial
from .score import strict_json, summarize


def replay(inputs, archive, metadata, key_path, work, out):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(sha(z.read(n)) == expected for n, expected in lock["files"].items())
        assert not any(
            n.endswith(("design.py", "score.py")) or "evaluator-only" in n
            for n in z.namelist()
        )
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
    records, startups, tokens, seconds = [], [], 0, 0.0
    for job in cfg["jobs"]:
        path = src / "4B/records" / f"{job['number']:03d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_bytes())
        assert all(record[k] == value for k, value in job.items())
        text = template.render(
            messages=public[job["id"]]["messages"],
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert text == record["prompt_text"]
        assert tok.encode(text, add_special_tokens=False).ids == record["input_ids"]
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
        assert (
            len(record["output_ids"])
            == record["generated_tokens"]
            <= cfg["max_new_tokens"]
        )
        assert record["max_seconds"] == cfg["case_seconds"]
        assert record["generation_config"]["do_sample"] is False
        assert record["max_new_tokens"] == cfg["max_new_tokens"]
        ended = bool(
            record["output_ids"] and record["output_ids"][-1] in meta["eos_ids"]
        )
        if record["completion_status"] == "completed":
            assert ended and boundary is not None and record["finish_reason"] == "eos"
        tokens += record["generated_tokens"]
        seconds += record["elapsed_seconds"]
        if job["split"] == "startup":
            truth = (
                {"ready": True}
                if job["id"] == "startup-1"
                else {"check": {"values": [7, 0, None], "flag": False}}
            )
            passed = (
                record["completion_status"] == "completed"
                and strict_json(decoded) == truth
            )
            startups.append({"id": job["id"], "passed": passed})
        else:
            records.append(record)
    keys = json.loads(key_path.read_bytes())
    scored = summarize(records, keys)
    result = {
        "startup": startups,
        "generated_tokens": tokens,
        "generation_seconds": seconds,
        "all_tokens_and_templates_verified": True,
        "weights_frozen": frozen,
        "input_sha256": sha(inputs.read_bytes()),
        "archive_sha256": sha(archive.read_bytes()),
        "scored": scored,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_bytes(serial(result))
    (out / "records.jsonl").write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in records
        )
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
                    "startup",
                    "generated_tokens",
                    "generation_seconds",
                    "all_tokens_and_templates_verified",
                )
            },
            indent=2,
        )
    )
    print(json.dumps(result["scored"]["groups"], indent=2))
