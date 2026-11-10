"""Score recorded body-channel forwards against the locally frozen simulator oracle."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
from statistics import mean

from prepare import build, save
from probe import read


def is_switch(answer):
    return answer["states"]["low"]["best"] != answer["states"]["high"]["best"]


def score(frozen, raw):
    public = read(frozen / "prompts.json")
    oracle = read(frozen / "oracle-local.json")
    rebuilt_public, rebuilt_oracle = build()
    assert public == json.loads(json.dumps(rebuilt_public)) and oracle == json.loads(json.dumps(rebuilt_oracle))
    status = read(raw / "status.json")
    lock = read(frozen / "run-lock.json")
    assert status["stage"] == "completed" and status["probes_done"] == status["probes_total"] == len(public["probes"])
    assert status["prompts_sha256"] == lock["prompts.json"] == sha256((frozen / "prompts.json").read_bytes()).hexdigest()
    assert lock["oracle_local_sha256"] == sha256((frozen / "oracle-local.json").read_bytes()).hexdigest()
    assert status["probe_source_sha256"] == lock["probe.py"]
    assert status["trainable_parameters"] == 0 and status["parameter_versions_unchanged"] and status["checkpoint_loading_clean"]
    assert status["dtype"] == "torch.bfloat16"
    by_mode = defaultdict(list)
    max_self_patch_delta = 0.0
    random_relative_norm_errors = []
    details = []

    for p in public["probes"]:
        recorded = read(raw / (p["id"] + ".json"))
        assert recorded["id"] == p["id"] and 0 < recorded["span"][0] < recorded["span"][1] < recorded["sequence_tokens"]
        assert len(set(recorded["prompt_sha256"].values())) == 3
        assert set(recorded["layers"]) == {str(layer) for layer in public["layers"]}
        answer = oracle[p["pair"]]
        states = answer["states"]
        low_best, high_best = (states[s]["best"] for s in ("low", "high"))
        label_for = {action: label for label, action in p["labels"].items()}
        outcomes = {
            "visible": [recorded["ordinary"][s] for s in ("low", "high")],
            "masked": [recorded["ordinary"]["masked"]] * 2,
        }
        for layer, evidence in recorded["layers"].items():
            max_self_patch_delta = max(max_self_patch_delta, evidence["self_patch_max_logit_delta"])
            assert evidence["self_patch_max_logit_delta"] <= 1e-6
            outcomes.update({
                f"matched_l{layer}": [evidence["matched"][s] for s in ("low", "high")],
                f"opposite_l{layer}": [evidence["matched"][s] for s in ("high", "low")],
                f"random_l{layer}": [evidence["random"][s] for s in ("low", "high")],
            })
            for norm in evidence["norms"].values():
                assert norm["matched_l2"] > 0
                random_relative_norm_errors.append(abs(norm["random_l2_after_bf16"] / norm["matched_l2"] - 1))

        condition = {}
        for name, pair in outcomes.items():
            for row in pair:
                assert set(row["logits"]) == {"A", "B", "C"}
                assert all(math.isfinite(v) for v in row["logits"].values())
                assert row["logits"][row["label"]] == max(row["logits"].values())
            selected = [p["labels"][row["label"]] for row in pair]
            change = None
            if is_switch(answer):
                margin = [row["logits"][label_for[high_best]] - row["logits"][label_for[low_best]] for row in pair]
                change = margin[1] - margin[0]
            loss = [max(states[s]["values"].values()) - states[s]["values"][a] for s, a in zip(("low", "high"), selected)]
            condition[name] = {"actions": selected, "both_correct": selected == [low_best, high_best],
                               "regret": loss, "directional_logit_change": change}

        detail = {"id": p["id"], "mode": p["mode"], "pair": p["pair"], "layout": p["layout"],
                  "switch": is_switch(answer), "nonterminal_switch": is_switch(answer) and answer["pair"]["no_death"],
                  "expected": [low_best, high_best], "conditions": condition}
        details.append(detail)
        by_mode[p["mode"]].append(detail)

    summaries = {}
    for mode, entries in by_mode.items():
        conditions = {}
        for name in entries[0]["conditions"]:
            result = {}
            for group in ("switch", "nonterminal_switch", "hold"):
                filtered = [entry for entry in entries if (not entry["switch"] if group == "hold" else entry[group])]
                by_pair = defaultdict(list)
                for entry in filtered:
                    by_pair[entry["pair"]].append(entry["conditions"][name])
                assert all(len(rows) == 2 for rows in by_pair.values())
                result[group] = {
                    "physical_pairs": len(by_pair),
                    "both_layouts_correct": sum(all(row["both_correct"] for row in rows) for rows in by_pair.values()),
                    "layout_pairs_correct": sum(row["both_correct"] for rows in by_pair.values() for row in rows),
                    "mean_regret": mean(mean(row["regret"]) for rows in by_pair.values() for row in rows),
                    "mean_directional_logit_change": None if group == "hold" else mean(
                        mean(row["directional_logit_change"] for row in rows) for rows in by_pair.values()
                    ),
                }
            conditions[name] = result
        summaries[mode] = conditions

    return {"schema": "sla-body-channel-results-v1", "status": status, "primary_layer": public["primary_layer"],
            "physical_pairs": len(oracle), "prompt_sets": len(details), "max_self_patch_logit_delta": max_self_patch_delta,
            "max_random_l2_relative_error_after_bf16": max(random_relative_norm_errors),
            "summaries": summaries, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.frozen, args.raw)
    save(args.output, result)
    for mode, conditions in result["summaries"].items():
        for name in ("visible", "masked", "matched_l15", "opposite_l15", "random_l15", "matched_l7"):
            info = conditions[name]
            print(json.dumps({"mode": mode, "condition": name, "switch": info["switch"],
                              "nonterminal_switch": info["nonterminal_switch"], "hold": info["hold"]}))


if __name__ == "__main__":
    main()
