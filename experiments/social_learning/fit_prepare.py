"""Freeze a tiny, fully crossed fitting assay before allocating a GPU.

This is an engineering memorization assay, not another lifetime experiment or
an unbiased test set. The selected body controls were already inspected.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from protocol import ACTIONS, PARTNERS, TRAIN_TASKS, TEST_TASKS, canonical, messages, targets, values

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/model-studies/2026-11-10/social-learning/gpu-results.zip"
SOURCE_HASH = "7c23ff365bf2837236f6c89343a8f845768cb11f3422a51c45af37959c7a5299"
STATES = (
    ("no_load", dict(damage=.25, load=0., weight=18., turns=4, protection_left=0)),
    ("needs_protection", dict(damage=.70, load=.85, weight=18., turns=4, protection_left=0)),
    ("last_turn", dict(damage=.70, load=.85, weight=18., turns=1, protection_left=0)),
    ("already_protected", dict(damage=.70, load=.85, weight=18., turns=4, protection_left=3)),
)


def explicit_probabilities(msg, probabilities):
    result = deepcopy(msg)
    result[0]["content"] = result[0]["content"].replace(
        "The partner's response pattern is not stated here.",
        "The current partner's grant probabilities are supplied below; use them directly.")
    result[1]["content"] += (
        f"\nGrant probability for appeal: {probabilities['appeal']:.12f}"
        f"\nGrant probability for exchange: {probabilities['exchange']:.12f}")
    return result


def social_messages(partner, style, order):
    definitions = {"accept": "the partner grants protection", "reject": "the partner declines protection"}
    choices = ("accept", "reject") if order == 0 else ("reject", "accept")
    return [dict(role="system", content=(
        "Predict the response of a scripted partner to a request. Use the response pattern learned from experience. "
        "Return exactly one outcome word. There is no body-value calculation in this task.\n" +
        "\n".join(f"{a}: {definitions[a]}" for a in choices))),
        dict(role="user", content=f"Partner: {partner}\nRequest style: {style}")]


def build(source=SOURCE):
    assert sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    with zipfile.ZipFile(source) as archive:
        old = json.loads(archive.read("protocol.json"))
        old_status = json.loads(archive.read("output/status.json"))
    cfg = deepcopy(old["training"])
    datasets, jobs = {}, []
    for mapping in (0, 1):
        life_id = f"s711210-m{mapping}"
        evidence = old["datasets"][life_id]["aligned"]
        estimates = {p: {s: c["mean"] for s, c in cells.items()}
                     for p, cells in evidence["posterior"].items()}
        for task in ("policy", "social", "body"):
            rows, transfer, compare, calculate = [], [], [], []
            if task == "social":
                for partner in PARTNERS:
                    for style in ("appeal", "exchange"):
                        for order in (0, 1):
                            p = estimates[partner][style]
                            rows.append(dict(id=f"{partner}-{style}-o{order}", partner=partner,
                                style=style, order=order, context=style,
                                messages=social_messages(partner, style, order), target=[p, 1-p]))
                actions = ["accept", "reject"]
            else:
                actions = list(ACTIONS)
                for context, state in STATES:
                    for partner in PARTNERS:
                        q = values(state, estimates[partner])
                        for order in (0, 1):
                            row = dict(id=f"{context}-{partner}-o{order}", context=context,
                                state=state, partner=partner, order=order, values=q, target=targets(q),
                                messages=messages(state, partner, TRAIN_TASKS[0][0], order))
                            if task == "body":
                                row["messages"] = explicit_probabilities(row["messages"], estimates[partner])
                            rows.append(row)
                            changed = deepcopy(row)
                            changed["id"] += "-new-question"
                            changed["messages"][1]["content"] = changed["messages"][1]["content"].replace(
                                TRAIN_TASKS[0][0], TEST_TASKS[0][0])
                            transfer.append(changed)
                            if task == "body":
                                selected = deepcopy(row)
                                selected["id"] += "-provided-returns"
                                selected["messages"] = [dict(role="system", content=(
                                    "Select the action with the largest already computed expected reference return. "
                                    "Return exactly its action word. Do not recompute the returns.")),
                                    dict(role="user", content="\n".join(
                                        f"{a}: {q[a]:.12f}" for a in (ACTIONS if order == 0 else tuple(reversed(ACTIONS)))))]
                                compare.append(selected)
                        if task == "body":
                            msg = explicit_probabilities(messages(state, partner, TRAIN_TASKS[0][0], 0), estimates[partner])
                            msg[0]["content"] = msg[0]["content"].replace(
                                "Choose one action for a synthetic agent. Return exactly the action word.",
                                'Compute the expected reference return for each action. Return exactly a JSON object '
                                'with keys "answer", "refuse", "appeal", "exchange" and numeric values; no other text.')
                            calculate.append(dict(id=f"{context}-{partner}-calculate", context=context,
                                partner=partner, state=state, messages=msg, values=q))
            did = f"m{mapping}-{task}"
            datasets[did] = dict(task=task, mapping=mapping, source_life_id=life_id,
                feedback=evidence["feedback"], posterior=evidence["posterior"], actions=actions,
                train=rows, new_question=transfer, provided_returns=compare, calculate=calculate)
            for learning_rate in ((.001, .0001) if task == "policy" else (.0001,)):
                jobs.append(dict(id=f"{did}-{'original' if learning_rate == .001 else 'lower'}-lr",
                    dataset=did, learning_rate=learning_rate))
    return dict(schema="crossed-mini-fit-v1", model=old["model"], revision=old["revision"],
        source_archive_sha256=SOURCE_HASH, expected_base_tensor_sha256=old_status["frozen_parameter_sha256"],
        initial_adapter_tensor_sha256=old_status["initial_adapter_sha256"],
        training_config={k: cfg[k] for k in ("rank", "alpha", "target_modules", "weight_decay", "gradient_clip")},
        training_batch_size=2, max_steps=96, checkpoints=[0, 16, 48, 96],
        permutation_seed=1211210, use_original_saved_zero_adapter=True, datasets=datasets, jobs=jobs,
        engineering_acceptance=dict(mean_kl_at_most=.02, all_teacher_actions_correct=True,
            max_order_tv_at_most=.05, all_required_partner_switch_pairs_correct=True),
        scope=("Fixed-example fitting assay, not independent scientific samples. The four body controls were selected "
               "after the previous audit. Two feedback banks share one environment seed and one saved adapter "
               "initialization. Both orderings are in training. No lifetime rollout or online feedback collection. "
               "Original and lower learning rates differ only in learning rate; task comparisons use lower rate. "
               "All scheduled checkpoints are retained; no best-checkpoint selection or adaptive hyperparameter search. "
               "The .02/.05 thresholds are engineering acceptance choices, not statistical significance tests. "
               "Synthetic four-turn utility remains unchanged and is not the paper's Reference-RUL objective."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build()
    with args.output.open("xb") as handle:
        handle.write(canonical(payload))
    print(json.dumps(dict(sha256=sha256(args.output.read_bytes()).hexdigest(), jobs=len(payload["jobs"]),
                         optimizer_steps=len(payload["jobs"]) * payload["max_steps"])))
