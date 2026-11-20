"""Freeze #26's existing messages, pairing and control gate before allocation."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from experiments.frozen_capability.design import serial

PROMPT_SHA = "905f5375470ff51203f60e05222ee3b8e428c6817dcc331e49a70a533c42bb3b"


def build(repo, out):
    source = repo / "artifacts/model-studies/2026-11-14/transparent-body"
    parent = (
        repo / "artifacts/model-studies/2026-11-14/frozen-interface/preregistered-v2"
    )
    if sha256((source / "example-prompts.json").read_bytes()).hexdigest() != PROMPT_SHA:
        raise ValueError("#26 messages changed")
    probes = json.loads((source / "example-prompts.json").read_bytes())
    audit = json.loads((source / "audit-only-calculations.json").read_bytes())
    previous = json.loads((parent / "public-prompts.json").read_bytes())
    controls = [
        r for r in previous if r["arm"] == "table" and r["form"] == "choice_only"
    ]
    prior_jobs = json.loads((parent / "execution.json").read_bytes())["jobs"]
    seeds = {r["id"]: r["seed"] for r in prior_jobs}
    control_keys = json.loads((parent / "evaluator-only-answer-key.json").read_bytes())
    public = [{"id": r["id"], "messages": r["messages"]} for r in controls] + probes
    assert len(public) == 200 and len({r["id"] for r in public}) == 200
    jobs = [
        {
            "number": i,
            "id": r["id"],
            "seed": seeds[r["id"]],
            "arm": "positive",
            "group": i,
        }
        for i, r in enumerate(controls)
    ]
    groups = {}
    for r in probes:
        key = audit[r["id"]]
        group = (key["seed"], key["partner"], key["calculation"]["health"])
        groups.setdefault(group, []).append(r["id"])
    for n, (group, ids) in enumerate(sorted(groups.items())):
        assert len(ids) == 4
        assert {(audit[i]["stage"], audit[i]["wording"]) for i in ids} == {
            ("birth", 0),
            ("late", 0),
            ("birth", 1),
            ("late", 1),
        }
        seed = int(sha256(serial(["transparent-body-v1", *group])).hexdigest()[:8], 16)
        for ident in sorted(
            ids, key=lambda i: (audit[i]["wording"], audit[i]["stage"] == "late")
        ):
            jobs.append(
                {
                    "number": len(jobs),
                    "id": ident,
                    "seed": seed,
                    "arm": "probe",
                    "group": 8 + n,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "transparent-body-gpu-v1",
        "models": ["4B"],
        "torch_version": "2.14.1+cu130",
        "gpu_name_contains": "L4",
        "source_prompt_sha256": PROMPT_SHA,
        "public_prompts_sha256": sha256(serial(public)).hexdigest(),
        "max_new_tokens": 128,
        "per_case_seconds": 60,
        "allocated_minutes_cap": 45,
        "sampler": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
        },
        "thinking": False,
        "jobs": jobs,
    }
    for name, data in (
        ("public-prompts.json", public),
        ("execution.json", config),
        (
            "positive-key-local-only.json",
            {r["id"]: control_keys[r["id"]] for r in controls},
        ),
    ):
        (out / name).write_bytes(serial(data))
    return {
        "controls": len(controls),
        "probes": len(probes),
        "groups": len(groups),
        "source_prompt_sha256": PROMPT_SHA,
        "public_prompts_sha256": config["public_prompts_sha256"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.repo, a.out)))
