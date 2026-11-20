"""Compare complete paired runs without treating repetitions as new lives."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def compare(first, second, out):
    a = json.loads((first / "analysis.json").read_bytes())
    b = json.loads((second / "analysis.json").read_bytes())
    result = {"paired_cases": {}, "source_sha256": {}}
    for label, source in (("first", first), ("second", second)):
        result["source_sha256"][label] = hashlib.sha256(
            (source / "analysis.json").read_bytes()
        ).hexdigest()
    for label, data in (("PhiMini_vs_4B", a), ("Olmo7B_vs_4B", b)):
        candidate = next(m for m in data["models"] if m != "4B")
        details = {}
        for task in ("step_unfenced", "plan_unfenced"):
            rows = data["models"][candidate][task]["cases"]
            reference = data["models"]["4B"][task]["cases"]
            counts = Counter()
            for cid, row in rows.items():
                ref = reference[cid]
                if not row["completed"] or not ref["completed"]:
                    category = "not_both_completed"
                elif row["joint_success"] and ref["joint_success"]:
                    category = "both_correct"
                elif row["joint_success"]:
                    category = "candidate_only_correct"
                elif ref["joint_success"]:
                    category = "qwen_only_correct"
                else:
                    category = "neither_correct"
                counts[category] += 1
            details[task] = counts
        result["paired_cases"][label] = details
    one = {
        r["id"]: r
        for line in (first / "4B.jsonl").read_bytes().splitlines()
        if (r := json.loads(line))
    }
    two = {
        r["id"]: r
        for line in (second / "4B.jsonl").read_bytes().splitlines()
        if (r := json.loads(line))
    }
    assert set(one) == set(two) and len(one) == 102
    same = {}
    for cid, r in one.items():
        other = two[cid]
        same[cid] = {
            "same_input_tokens": r["input_ids"] == other["input_ids"],
            "same_output_tokens": r["output_ids"] == other["output_ids"],
            "same_completion": r["completion_status"] == other["completion_status"],
        }
    result["qwen_repeat"] = {
        "n": len(same),
        "identical_output_count": sum(r["same_output_tokens"] for r in same.values()),
        "all_inputs_identical": all(r["same_input_tokens"] for r in same.values()),
        "cases": same,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.first, args.second, args.out)
    print(json.dumps(result["paired_cases"]))
    print(
        "Qwen identical token outputs:", result["qwen_repeat"]["identical_output_count"]
    )
