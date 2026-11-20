"""Freeze matched, one-field probes of the saved social-policy adapters."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from protocol import ACTIONS, PARTNERS, REPO, TEST_TASKS, canonical, digest, messages, probabilities, targets, values

DATA = REPO / "artifacts/model-studies/2026-11-10/social-learning"
GROUPS = ("train_original", "partner_only", "order_only", "question_only")


def build(archive_path=DATA / "gpu-results.zip"):
    archive_bytes = archive_path.read_bytes()
    stimuli, datasets, jobs = {}, {}, []
    with zipfile.ZipFile(archive_path) as archive:
        source = json.loads(archive.read("protocol.json"))

        def stimulus(prompt):
            sid = digest(prompt)
            item = {"id": sid, "messages": prompt}
            assert sid not in stimuli or stimuli[sid] == item
            stimuli[sid] = item
            return sid

        old_ids = {row["id"]: stimulus(row["messages"]) for row in source["tests"]}
        frozen_ids, frozen_singles = list(old_ids.values()), list(old_ids.values())
        frozen_references = {old_ids[row["id"]]: "output/frozen/" + row["id"] + ".json" for row in source["tests"]}
        for life in source["lives"]:
            lid = life["id"]
            for arm in ("aligned", "shuffled"):
                run_id = lid + "-" + arm
                training = source["datasets"][lid][arm]
                groups = {group: [] for group in GROUPS}
                for i, row in enumerate(training["rows"]):
                    original = row["messages"]
                    system, user = (m["content"] for m in original)
                    order = int(system.index("answer:") > system.index("exchange:"))
                    question = user.split("New task: ")[1]
                    for group in GROUPS:
                        partner = PARTNERS[1 - PARTNERS.index(row["partner"])] if group == "partner_only" else row["partner"]
                        shown_order = 1 - order if group == "order_only" else order
                        shown_question = TEST_TASKS[(i // 2) % len(TEST_TASKS)][0] if group == "question_only" else question
                        prompt = messages(row["state"], partner, shown_question, shown_order)
                        if group == "train_original":
                            assert prompt == original
                        p = {style: counts["mean"] for style, counts in training["posterior"][partner].items()}
                        q = values(row["state"], p)
                        target = targets(q)
                        if group != "partner_only":
                            assert all(abs(a - b) < 1e-12 for a, b in zip(target, row["target"]))
                        groups[group].append(dict(id=group + ":" + row["id"], source_id=row["id"],
                            stimulus_id=stimulus(prompt), state=row["state"], partner=partner,
                            question=shown_question, order=shown_order, target=target, values=q))
                # Partner/order changes share the same off-diagonal prompt set.
                # They remain different paired contrasts, not independent samples.
                assert {r["stimulus_id"] for r in groups["partner_only"]} == {r["stimulus_id"] for r in groups["order_only"]}
                seen = list(dict.fromkeys(r["stimulus_id"] for group in GROUPS for r in groups[group]))
                assert len(seen) == 96
                old_tests = []
                for row in source["tests"]:
                    p = {style: counts["mean"] for style, counts in training["posterior"][row["partner"]].items()}
                    q = values(row["state"], p)
                    old_tests.append(dict(id=row["id"], stimulus_id=old_ids[row["id"]], state=row["state"],
                        partner=row["partner"], order=row["order"], target=targets(q), values=q,
                        true_values=values(row["state"], probabilities(life["mapping"])[row["partner"]])))
                datasets[run_id] = dict(life=lid, arm=arm, groups=groups, original_tests=old_tests)
                metadata = json.loads(archive.read("output/" + run_id + "/training.json"))
                jobs.append(dict(id=run_id, adapter_member="output/" + run_id + "/adapter.safetensors",
                    adapter_tensor_sha256=metadata["final_adapter_sha256"],
                    stimulus_ids=seen + list(old_ids.values()),
                    batch_check_ids=list(old_ids.values()) + [r["stimulus_id"] for r in groups["train_original"][:4]],
                    original_references={old_ids[row["id"]]: "output/" + run_id + "-tests/" + row["id"] + ".json" for row in source["tests"]}))
                frozen_ids.extend(seen)
                frozen_singles.extend(r["stimulus_id"] for r in groups["train_original"][:4])
        jobs.insert(0, dict(id="frozen", adapter_member=None, adapter_tensor_sha256=None,
            stimulus_ids=list(dict.fromkeys(frozen_ids)), batch_check_ids=list(dict.fromkeys(frozen_singles)),
            original_references=frozen_references))
        # Counterbalanced source lives can share early body/prompt snapshots.
        assert set(stimuli) == set(jobs[0]["stimulus_ids"])
        assert 0 < len(stimuli) <= 416
        assert all(len(j["stimulus_ids"]) == 128 and len(j["batch_check_ids"]) == 36 for j in jobs[1:])
    public = dict(schema="saved-lora-readonly-diagnostic-v1", model=source["model"], revision=source["revision"],
        source_archive_sha256=sha256(archive_bytes).hexdigest(), training_config=source["training"],
        inference_batch_size=1, batch_check_size=4, stimuli=stimuli, jobs=jobs,
        scope="No optimizer, no training, no new life or continuation; one-field changes of saved training prompts.")
    key = dict(schema=public["schema"], public_sha256=digest(public), datasets=datasets,
        groups=GROUPS, independent_source_seed_blocks=2,
        note="128 paired case records per adapter use 96 distinct training-related prompts. Partner/order contrasts share off-diagonal prompts.")
    return public, key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    public, key = build()
    (args.output_dir / "public.json").write_bytes(canonical(public))
    (args.output_dir / "key-local.json").write_bytes(canonical(key))
    print(json.dumps({"public_sha256": digest(public), "unique_prompts": len(public["stimuli"]),
        "model_prompt_forwards": sum(len(j["stimulus_ids"]) for j in public["jobs"]),
        "batch_prompt_rechecks": sum(len(j["batch_check_ids"]) for j in public["jobs"])}))
