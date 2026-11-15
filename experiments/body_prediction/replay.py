"""Verify native token streams and score the pinned cross-family GPU comparison."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from experiments.frozen_capability.replay import unpack
from experiments.frozen_capability.scoring import assistant_message
from experiments.self_future.design import serial
from experiments.self_future.score import summarize

from .scoring import summarize_steps
from .wire import startup_matches, without_json_fence


def replay(inputs, archive, metadata, fixtures, work, out):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    with ZipFile(inputs) as z:
        lock = json.loads(z.read("run-lock.json"))
        assert set(z.namelist()) == set(lock["files"]) | {"run-lock.json"}
        assert all(sha(z.read(n)) == v for n, v in lock["files"].items())
        assert not any(
            "evaluator" in n
            or n.endswith(("/design.py", "/scoring.py"))
            and "body_prediction" in n
            for n in z.namelist()
        )
        cfg = json.loads(z.read("execution.json"))
        public = z.read("public-prompts.json")
        assert public == (fixtures / "public-prompts.json").read_bytes()
        assert sha(public) == cfg["public_prompts_sha256"]
        assert (
            sha((fixtures / "manifest.json").read_bytes())
            == cfg["fixture_manifest_sha256"]
        )
        expected = {m: json.loads(z.read(f"expected-{m}.json")) for m in cfg["models"]}
    key = json.loads((fixtures / "evaluator-only-key.json").read_bytes())
    lookup = {r["id"]: r for r in json.loads(public)}
    unpack(archive, work)
    src = work / "output"
    assert json.loads((src / "execution.json").read_bytes()) == cfg
    result = {
        "models": {},
        "input_sha256": sha(inputs.read_bytes()),
        "archive_sha256": sha(archive.read_bytes()),
    }
    out.mkdir(parents=True, exist_ok=False)
    for short in cfg["models"]:
        folder = src / short
        assert (
            json.loads((folder / "weights-manifest.json").read_bytes())
            == expected[short]
        )
        meta = json.loads((folder / "loading.json").read_bytes())
        frozen = json.loads((folder / "unchanged.json").read_bytes())
        assert (
            meta["loading_empty"]
            and frozen["parameter_versions_unchanged"]
            and frozen["gradients_absent"]
        )
        assert frozen["trainable_parameters"] == 0
        for name in ("tokenizer.json", "tokenizer_config.json"):
            assert (
                sha((metadata / short / name).read_bytes())
                == expected[short]["metadata"][name]["sha256"]
            )
        tok = Tokenizer.from_file(str(metadata / short / "tokenizer.json"))
        tok_config = json.loads(
            (metadata / short / "tokenizer_config.json").read_bytes()
        )
        template_path = metadata / short / "chat_template.jinja"
        if "chat_template.jinja" in expected[short]["metadata"]:
            assert (
                sha(template_path.read_bytes())
                == expected[short]["metadata"]["chat_template.jinja"]["sha256"]
            )
            template_text = template_path.read_text(encoding="utf8")
        else:
            template_text = tok_config["chat_template"]
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def throw(text):
            raise ValueError(text)

        env.globals["raise_exception"] = throw
        template = env.from_string(template_text)
        records, startups = [], []
        for job in cfg["jobs"]:
            file = folder / "records" / f"{job['number']:03d}.json"
            if not file.exists():
                continue
            r = json.loads(file.read_bytes())
            assert r["model"] == short and all(r[k] == v for k, v in job.items())
            prompt = template.render(
                messages=lookup[r["id"]]["messages"],
                tools=None,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert prompt == r["prompt_text"]
            assert tok.encode(prompt, add_special_tokens=False).ids == r["input_ids"]
            assert (
                tok.decode(r["output_ids"], skip_special_tokens=False)
                == r["output_text"]
            )
            final, channel, thoughts, end = assistant_message(
                r["output_ids"],
                False,
                meta["close_id"],
                meta["open_id"],
                meta["eos_ids"],
                meta["chat_end_id"],
            )
            if meta["open_id"] is None:
                end = next(
                    (
                        i
                        for i, token in enumerate(r["output_ids"])
                        if token in meta["eos_ids"]
                    ),
                    None,
                )
                final = r["output_ids"] if end is None else r["output_ids"][:end]
                channel, thoughts = "final_channel", 0
            text = (
                None if final is None else tok.decode(final, skip_special_tokens=False)
            )
            assert (final, channel, thoughts, end, text) == (
                r["final_ids"],
                r["channel"],
                r["thinking_tokens"],
                r["message_end_position"],
                r["final_text"],
            )
            cap, seconds = cfg["budgets"][r["task"]]
            assert r["generated_tokens"] == len(r["output_ids"]) <= cap
            assert (
                r["max_new_tokens"] == cap
                and r["max_seconds"] == seconds
                and not r["generation_config"]["do_sample"]
            )
            if r["completion_status"] == "completed":
                assert (
                    r["finish_reason"] == "eos"
                    and end is not None
                    and r["output_ids"][-1] in meta["eos_ids"]
                )
            records.append(r)
            if r["task"] == "startup":
                truth = (
                    {"ready": True}
                    if r["id"] == "startup-1"
                    else {"check": {"values": [7, 0, None], "flag": False}}
                )
                startups.append(startup_matches(r, truth))
        unfenced = [
            {**r, "final_text": without_json_fence(r["final_text"])} for r in records
        ]
        result["models"][short] = {
            "startup": startups,
            "all_tokens_and_templates_verified": True,
            "frozen": frozen,
            "raw_calls": len(records),
            "generated_tokens": sum(r["generated_tokens"] for r in records),
            "generation_seconds": sum(r["elapsed_seconds"] for r in records),
            "single_json_fence_count": sum(
                a["final_text"] != b["final_text"] for a, b in zip(records, unfenced)
            ),
            "plan": summarize(
                [r for r in records if r["task"] == "plan"], key["plans"]
            ),
            "step": summarize_steps(
                [r for r in records if r["task"] == "step"], key["steps"]
            ),
            "plan_unfenced": summarize(
                [r for r in unfenced if r["task"] == "plan"], key["plans"]
            ),
            "step_unfenced": summarize_steps(
                [r for r in unfenced if r["task"] == "step"], key["steps"]
            ),
        }
        (out / (short + ".jsonl")).write_bytes(
            b"".join(json.dumps(r).encode() + b"\n" for r in records)
        )
    (out / "analysis.json").write_bytes(serial(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "archive", "metadata", "fixtures", "work", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    result = replay(a.inputs, a.archive, a.metadata, a.fixtures, a.work, a.out)
    for model, v in result["models"].items():
        print(
            model,
            json.dumps(
                {
                    "startup": v["startup"],
                    "calls": v["raw_calls"],
                    "tokens": v["generated_tokens"],
                    "step": v["step"]["groups"]["all"],
                    "step_gate": v["step"]["readiness_gate"],
                    "plans": {
                        k: {
                            m: g["counts"][m]
                            for m in (
                                "completed",
                                "predictions_correct",
                                "choice_correct",
                                "joint_success",
                            )
                        }
                        for k, g in v["plan"]["groups"].items()
                    },
                }
            ),
        )
