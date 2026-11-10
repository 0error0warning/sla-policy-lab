"""Score the fixed-weight diagnostic using independently held teacher targets."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

import numpy as np

from diagnose_prepare import DATA, GROUPS, build
from protocol import ACTIONS, canonical, digest
from score import match, read, verify_archive


def distribution(record, mode="full"):
    field = "full_log_likelihood" if mode == "full" else "word_log_likelihood"
    scores = np.array([record["actions"][a][field] for a in ACTIONS])
    shifted = scores - scores.max()
    log_p = shifted - math.log(float(np.exp(shifted).sum()))
    return np.exp(log_p), log_p, ACTIONS[int(scores.argmax())]


def best(case):
    maximum = max(case["values"].values())
    return {a for a, value in case["values"].items() if abs(maximum - value) < 1e-10}


def metrics(cases, records, constant, mode="full"):
    losses, kls, regrets, tvs, hits, constant_kls = [], [], [], [], [], []
    for case in cases:
        p, log_p, selected = distribution(records[case["stimulus_id"]], mode)
        q = np.array(case["target"])
        entropy = -float(q @ np.log(q))
        ce = -float(q @ log_p)
        losses.append(ce)
        kls.append(ce - entropy)
        constant_kls.append(-float(q @ np.log(constant)) - entropy)
        regrets.append(max(case["values"].values()) - case["values"][selected])
        tvs.append(float(np.abs(p - q).sum() / 2))
        hits.append(selected in best(case))
    return dict(n=len(cases), teacher_argmax_correct=sum(hits), mean_cross_entropy=mean(losses),
        mean_kl=mean(kls), mean_reference_regret=mean(regrets), mean_probability_tv=mean(tvs),
        context_free_mean_kl=mean(constant_kls))


def pairing(original, changed, records, mode="full"):
    both, before_only, after_only, neither, switches, required, required_both = 0, 0, 0, 0, 0, 0, 0
    model_tv, teacher_tv, dot, denominator = [], [], 0., 0.
    for a, b in zip(original, changed):
        assert a["source_id"] == b["source_id"]
        pa, _, aa = distribution(records[a["stimulus_id"]], mode)
        pb, _, ab = distribution(records[b["stimulus_id"]], mode)
        qa, qb = np.array(a["target"]), np.array(b["target"])
        ca, cb = aa in best(a), ab in best(b)
        both += ca and cb
        before_only += ca and not cb
        after_only += cb and not ca
        neither += not ca and not cb
        switches += aa != ab
        should_change = best(a).isdisjoint(best(b))
        required += should_change
        required_both += should_change and ca and cb
        model_tv.append(float(np.abs(pb - pa).sum() / 2))
        teacher_tv.append(float(np.abs(qb - qa).sum() / 2))
        dot += float((pb - pa) @ (qb - qa))
        denominator += float((qb - qa) @ (qb - qa))
    return dict(n=len(original), both_correct=both, original_only_correct=before_only,
        changed_only_correct=after_only, neither_correct=neither, action_changes=switches,
        teacher_requires_switch=required, required_switch_both_correct=required_both,
        mean_model_tv=mean(model_tv), mean_teacher_tv=mean(teacher_tv),
        partner_response_slope=(dot / denominator if denominator > 1e-15 else None))


def score(root, key_path):
    public, key = read(root / "public.json"), read(key_path)
    rebuilt_public, rebuilt_key = build()
    match(public, rebuilt_public)
    match(key, rebuilt_key)
    assert key["public_sha256"] == digest(public)
    for name, expected in read(root / "run-lock.json").items():
        raw = (DATA / "gpu-results.zip").read_bytes() if name == "source-study.zip" else (root / name).read_bytes()
        assert sha256(raw).hexdigest() == expected
    raw = root / "output"
    state = read(raw / "status.json")
    assert state["stage"] == "complete" and state["completed_jobs"] == state["total_jobs"] == 9
    assert state["optimizer_steps"] == state["trainable_parameters"] == 0 and state["all_weights_unchanged"]
    assert state["frozen_parameter_sha256"] == state["final_frozen_parameter_sha256"]
    assert state["public_sha256"] == digest(public)
    assert state["diagnostic_source_sha256"] == sha256((root / "diagnose.py").read_bytes()).hexdigest()
    original = verify_archive(DATA / "gpu-results.zip")
    original_state = json.loads(original.read("output/status.json"))
    assert state["frozen_parameter_sha256"] == original_state["frozen_parameter_sha256"]
    assert state["source_archive_sha256"] == public["source_archive_sha256"]
    all_records, validation = {}, {}
    prefix_max = 0.
    forwards = 0
    for job in public["jobs"]:
        jid = job["id"]
        checked = read(raw / jid / "checks.json")
        assert checked["adapter_unchanged"] and checked["parameter_versions_unchanged"]
        assert checked["optimizer_steps"] == checked["trainable_parameters"] == 0
        expected_hash = job["adapter_tensor_sha256"] or original_state["initial_adapter_sha256"]
        assert checked["adapter_tensor_sha256"] == expected_hash
        records, batches = {}, {}
        for mode, identifiers, dest in (("single", job["stimulus_ids"], records), ("batch", job["batch_check_ids"], batches)):
            for sid in identifiers:
                record = read(raw / jid / mode / (sid + ".json"))
                assert record["id"] == sid
                prefix_max = max(prefix_max, record["candidate_prefix_max_logit_difference"])
                for action in ACTIONS:
                    detail = record["actions"][action]
                    assert detail["token_ids"] == state["action_suffix_ids"][action]
                    components = detail["token_log_probabilities"]
                    assert all(math.isfinite(x) for x in components)
                    assert abs(sum(components) - detail["full_log_likelihood"]) < 1e-5
                    assert abs(sum(components[:-1]) - detail["word_log_likelihood"]) < 1e-5
                    assert components[-1] == detail["eos_log_probability"]
                for mode_name, field, selected in (("full", "full_probabilities", "action"), ("word", "word_only_probabilities", "word_only_action")):
                    p, _, a = distribution(record, mode_name)
                    assert a == record[selected]
                    assert max(abs(p[i] - record[field][action]) for i, action in enumerate(ACTIONS)) < 3e-7
                dest[sid] = record
                forwards += 1
        replay_max, replay_exact, replay_action_changes = 0., 0, 0
        for sid, path in job["original_references"].items():
            old = json.loads(original.read(path))
            now = records[sid]
            assert now["input_token_ids"] == old["input_token_ids"] and now["prompt_sha256"] == old["prompt_sha256"]
            delta = max(abs(now["actions"][a]["full_log_likelihood"] - old["action_log_likelihood"][a]) for a in ACTIONS)
            replay_max = max(replay_max, delta)
            replay_exact += delta == 0
            replay_action_changes += now["action"] != old["action"]
        assert checked["original_replay_exact"] == replay_exact
        assert checked["original_replay_max_full_score_delta"] == replay_max
        batch_delta = max(abs(batches[sid]["actions"][a]["full_log_likelihood"] - records[sid]["actions"][a]["full_log_likelihood"])
                          for sid in batches for a in ACTIONS)
        for sid in batches:
            assert batches[sid]["input_token_ids"] == records[sid]["input_token_ids"]
            assert batches[sid]["prompt_sha256"] == records[sid]["prompt_sha256"]
        validation[jid] = dict(original_replay_cases=32, original_replay_exact=replay_exact,
            original_replay_max_delta=replay_max, original_action_changes=replay_action_changes,
            batch_check_n=len(batches), batch_max_full_score_delta=batch_delta,
            batch_action_changes=sum(batches[sid]["action"] != records[sid]["action"] for sid in batches),
            eos_action_changes=sum(r["action"] != r["word_only_action"] for r in records.values()))
        all_records[jid] = records
    assert forwards == state["completed_forwards"] == state["total_forwards"] == 1752
    by_model = []
    for run_id, dataset in key["datasets"].items():
        groups = dataset["groups"]
        constant = np.array([r["target"] for r in groups["train_original"]]).mean(axis=0)
        for label, model_id in (("saved_adapter", run_id), ("frozen", "frozen")):
            records = all_records[model_id]
            result = dict(run_id=run_id, life=dataset["life"], feedback=dataset["arm"], model=label)
            result["groups"] = {g: {mode: metrics(cases, records, constant, mode) for mode in ("full", "word")}
                                for g, cases in groups.items()}
            result["pairs"] = {g: pairing(groups["train_original"], groups[g], records)
                               for g in GROUPS if g != "train_original"}
            result["old_static"] = metrics(dataset["original_tests"], records, constant)
            result["old_static"]["true_reference_correct"] = sum(
                abs(case["true_values"][records[case["stimulus_id"]]["action"]] - max(case["true_values"].values())) < 1e-10
                for case in dataset["original_tests"])
            result["eos_group_action_changes"] = {g: sum(records[c["stimulus_id"]]["action"] != records[c["stimulus_id"]]["word_only_action"] for c in cases)
                                                    for g, cases in groups.items()}
            by_model.append(result)
    aggregate = {}
    for feedback in ("aligned", "shuffled"):
        aggregate[feedback] = {}
        for label in ("saved_adapter", "frozen"):
            models = [r for r in by_model if r["feedback"] == feedback and r["model"] == label]
            aggregate[feedback][label] = {g: dict(n=sum(r["groups"][g]["full"]["n"] for r in models),
                teacher_correct=sum(r["groups"][g]["full"]["teacher_argmax_correct"] for r in models),
                mean_kl=mean(r["groups"][g]["full"]["mean_kl"] for r in models),
                context_free_mean_kl=mean(r["groups"][g]["full"]["context_free_mean_kl"] for r in models)) for g in GROUPS}
    original.close()
    return dict(public_sha256=digest(public), status=state, validation=validation,
        candidate_prefix_max_logit_difference=prefix_max, by_model=by_model, descriptive_aggregate=aggregate,
        seed_blocks=2, primary_model_prompt_records=1422, batch_check_records=330,
        note="Partner-only and order-only groups share off-diagonal prompts. Diagnose paired contrasts; their aggregate scores coincide by construction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with verify_archive(args.archive) as archive:
        result = score(zipfile.Path(archive), args.key)
    with args.output.open("xb") as handle:
        handle.write(canonical(result))
    print(json.dumps(result["descriptive_aggregate"]))
