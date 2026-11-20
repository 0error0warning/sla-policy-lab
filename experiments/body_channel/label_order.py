"""Cross letter-to-action assignment with semantic display order in a separate run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from prepare import LABELS, MASK, ORDERS, PAIRS, build, messages, save, write_frozen
from probe import read


def freeze():
    original, oracle = build()
    old = {p["id"]: p for p in original["probes"]}
    public = dict(original, schema="sla-body-channel-label-order-v1", probes=[])
    for pair in PAIRS:
        states = list(oracle[pair.name]["states"].values())
        for mapping_index, order in enumerate(ORDERS):
            letter_for = dict(zip(order, LABELS))
            for display_index, semantic_order in enumerate(ORDERS):
                labels = {letter_for[action]: action for action in semantic_order}
                variants = {name: messages(pair, states, labels, "provided_values", slot) for name, slot in
                            zip(("low", "high", "masked"), (f"{pair.damage[0]:.2f}", f"{pair.damage[1]:.2f}", MASK))}
                if mapping_index == display_index:
                    prior = old[f"{pair.name}-provided_values-{mapping_index}"]
                    assert variants == prior["messages"] and labels == prior["labels"]
                public["probes"].append({"id": f"{pair.name}-m{mapping_index}-d{display_index}", "pair": pair.name,
                                         "mapping": mapping_index, "display_order": display_index,
                                         "labels": labels, "messages": variants})
    return public, oracle


def evaluate(frozen, raw, prior):
    public = read(frozen / "prompts.json")
    oracle = read(frozen / "oracle-local.json")
    expected, expected_oracle = freeze()
    assert public == json.loads(json.dumps(expected)) and oracle == json.loads(json.dumps(expected_oracle))
    status = read(raw / "status.json")
    lock = read(frozen / "run-lock.json")
    assert status["stage"] == "completed" and status["probes_done"] == len(public["probes"])
    assert status["prompts_sha256"] == lock["prompts.json"] == sha256((frozen / "prompts.json").read_bytes()).hexdigest()
    assert lock["oracle_local_sha256"] == sha256((frozen / "oracle-local.json").read_bytes()).hexdigest()
    assert status["probe_source_sha256"] == lock["probe.py"]
    assert status["trainable_parameters"] == 0 and status["parameter_versions_unchanged"] and status["checkpoint_loading_clean"]
    details = []
    diagonals = []
    for p in public["probes"]:
        row = read(raw / (p["id"] + ".json"))
        assert row["id"] == p["id"]
        if p["mapping"] == p["display_order"]:
            old = read(prior / f"{p['pair']}-provided_values-{p['mapping']}.json")
            assert old["prompt_sha256"] == row["prompt_sha256"]
            diagonals.append({"id": p["id"], "exact_scores_and_patches_reproduced": all(old[k] == row[k] for k in ("ordinary", "layers"))})
        best = [oracle[p["pair"]]["states"][s]["best"] for s in ("low", "high")]
        results = {"visible": row["ordinary"], "masked": {s: row["ordinary"]["masked"] for s in ("low", "high")}}
        for layer, layer_row in row["layers"].items():
            assert layer_row["self_patch_max_logit_delta"] <= 1e-6
            for condition in ("matched", "random"):
                results[f"{condition}_l{layer}"] = layer_row[condition]
        choices = {name: [p["labels"][out[s]["label"]] for s in ("low", "high")] for name, out in results.items()}
        details.append({"id": p["id"], "pair": p["pair"], "mapping": p["mapping"], "display_order": p["display_order"],
                        "switch": best[0] != best[1], "expected": best, "actions": choices,
                        "both_correct": {name: actions == best for name, actions in choices.items()}})
    table = []
    for mapping_index in range(2):
        for display_index in range(2):
            group = [d for d in details if (d["mapping"], d["display_order"]) == (mapping_index, display_index)]
            assert len(group) == 8 and sum(d["switch"] for d in group) == 4
            table.append({"mapping": mapping_index, "display_order": display_index,
                          "switch_correct": {c: sum(d["both_correct"][c] for d in group if d["switch"]) for c in group[0]["actions"]},
                          "hold_correct": {c: sum(d["both_correct"][c] for d in group if not d["switch"]) for c in group[0]["actions"]}})
    return {"schema": "sla-body-channel-label-order-results-v1", "status": status, "cells": table,
            "diagonal_reproduction": diagonals, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True, type=Path)
    score = commands.add_parser("score")
    for name in ("frozen", "raw", "prior", "output"):
        score.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        public, oracle = freeze()
        hashes = write_frozen(args.output, public, oracle)
        print(json.dumps({"prompt_sets": len(public["probes"]), "original_cells_repeated": 16, "new_crossed_cells": 16,
                          "hashes": hashes}))
    else:
        result = evaluate(args.frozen, args.raw, args.prior)
        save(args.output, result)
        print(json.dumps({"cells": result["cells"], "exact_reproductions": sum(
            row["exact_scores_and_patches_reproduced"] for row in result["diagonal_reproduction"])}))


if __name__ == "__main__":
    main()
