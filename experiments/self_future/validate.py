"""Verify archived unrun fixtures, independent physics and scorer controls."""

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

from .design import audit_physics, build, fixtures, serial
from .score import summarize


def validate(inputs):
    inputs = Path(inputs)
    with tempfile.TemporaryDirectory(prefix="self-future-rebuild-") as tmp:
        rebuilt = Path(tmp) / "inputs"
        build(rebuilt)
        equality = {
            name: (inputs / name).read_bytes() == (rebuilt / name).read_bytes()
            for name in (
                "prompts.jsonl",
                "startup-prompts.jsonl",
                "evaluator-only-key.json",
                "manifest.json",
            )
        }
    if not all(equality.values()):
        raise ValueError("fixture or lock drift: " + json.dumps(equality))
    _, keys = fixtures()
    # Explicitly synthetic: no outputs or completion claims from any real model.
    synthetic = [
        {
            "id": cid,
            "completion_status": "completed",
            "final_text": json.dumps(
                {"plans": key["plans"], "choice": key["accepted_choices"][0]}
            ),
        }
        for cid, key in keys.items()
    ]
    perfect = summarize(synthetic, keys)
    assert perfect["development_gate"] and perfect["confirmation_gate"]
    constants = {}
    for label in ("P", "Q"):
        records = [
            {
                "id": cid,
                "completion_status": "completed",
                "final_text": json.dumps({"plans": key["plans"], "choice": label}),
            }
            for cid, key in keys.items()
        ]
        result = summarize(records, keys)
        constants[label] = {
            "development_gate": result["development_gate"],
            "confirmation_gate": result["confirmation_gate"],
        }
        assert not any(constants[label].values())
    missing = summarize([], keys)
    assert not missing["development_gate"] and not missing["confirmation_gate"]
    return {
        "status": "CPU_validation_only_no_model_results",
        "python": platform.python_version(),
        "model_calls": 0,
        "gpu_allocations": 0,
        "new_autonomous_lives": 0,
        "independent_physics_comparisons": audit_physics(),
        "byte_identical_rebuild": equality,
        "synthetic_perfect_response_gate_pass": True,
        "synthetic_constant_choice_controls": constants,
        "missing_records_do_not_pass": True,
        "manifest_sha256": hashlib.sha256(
            (inputs / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = validate(args.inputs)
    with Path(args.out).open("xb") as handle:
        handle.write(serial(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
