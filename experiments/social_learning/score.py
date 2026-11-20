"""Recompute social-pilot behavior and replay its continuation bodies on the CPU."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

import numpy as np

from protocol import ACTIONS, PARTNERS, REPO, TEST_TASKS, SocialWorld, build, canonical, digest, messages, probabilities, values


def read(path):
    return json.loads(path.read_bytes().decode("utf-8"))


def match(actual, expected):
    """Exact structure, with only last-bit CPU floating-point tolerance."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            match(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            match(a, b)
    elif type(actual) is float and type(expected) is float:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def parameter_file(path):
    """Read this pilot's small, numeric safetensors without torch or pickle."""
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    start = 8 + size
    result, h = {}, sha256()
    kinds = {"F32": (np.dtype("<f4"), "torch.float32"), "F16": (np.dtype("<f2"), "torch.float16")}
    for name in sorted(k for k in header if k != "__metadata__"):
        entry = header[name]
        dtype, torch_name = kinds[entry["dtype"]]
        a, b = entry["data_offsets"]
        block = raw[start + a:start + b]
        array = np.frombuffer(block, dtype=dtype).reshape(entry["shape"])
        assert np.isfinite(array).all()
        h.update(canonical([name, torch_name, entry["shape"]]))
        h.update(block)
        result[name] = array
    return result, h.hexdigest()


def check_policy(record):
    scores = record["action_log_likelihood"]
    probs = record["probabilities"]
    assert set(scores) == set(probs) == set(ACTIONS)
    assert all(math.isfinite(v) for v in scores.values()) and all(math.isfinite(v) and v >= 0 for v in probs.values())
    array = np.array([scores[a] for a in ACTIONS], dtype=np.float64)
    expected = np.exp(array - array.max())
    expected /= expected.sum()
    assert max(abs(expected[i] - probs[a]) for i, a in enumerate(ACTIONS)) < 2e-7
    assert record["action"] == max(ACTIONS, key=scores.get)


def replay(life, records, learned):
    world = SocialWorld(life["seed"], life["mapping"])
    for row in life["trace"]:
        actual = world.step(row["encounter_id"], row["partner"], row["action"], tuple(row["task"]))
        assert actual == row
    initial_damage = world.world.state.damage
    total = 0.
    for row in records:
        if learned:
            check_policy(row)
        t = world.world.state.age
        assert row["partner"] == PARTNERS[t % 2] and row["task"] == list(TEST_TASKS[t % 4])
        actual = world.step(row["encounter_id"], row["partner"], row["action"], tuple(row["task"]))
        for key, value in actual.items():
            if key == "reward":
                assert abs(value - row[key]) < 1e-12
            else:
                assert row[key] == value, key
        total += row["reward"]
    assert world.world.done
    return dict(turns=len(records), reward=total, initial_damage=initial_damage,
                final_damage=world.world.state.damage, alive=world.world.state.alive,
                answers=sum(r["action"] == "answer" for r in records),
                requests=sum(r["action"] in ("appeal", "exchange") for r in records),
                accepts=sum(r["info"]["protection_accepted"] is True for r in records))


def cpu_baseline(life, name, posterior=None):
    world = SocialWorld(life["seed"], life["mapping"])
    for row in life["trace"]:
        world.step(row["encounter_id"], row["partner"], row["action"], tuple(row["task"]))
    records = []
    while not world.world.done:
        t = world.world.state.age
        partner, task = PARTNERS[t % 2], TEST_TASKS[t % 4]
        obs = world.observation(partner, task[0])
        state = dict(damage=obs["damage"], load=obs["load"], weight=18., turns=min(4, obs["horizon"] - t),
                     protection_left=obs["protection_left"])
        p = probabilities(life["mapping"])[partner] if posterior is None else {s: x["mean"] for s, x in posterior[partner].items()}
        q = values(state, p)
        action = "answer" if name == "always_answer" else max(ACTIONS, key=q.get)
        records.append(world.step(world.encounter_id(), partner, action, task))
    return records


def score(root):
    for name, expected in read(root / "run-lock.json").items():
        assert sha256((root / name).read_bytes()).hexdigest() == expected, name
    data = read(root / "protocol.json")
    match(data, json.loads(canonical(build())))
    raw = root / "output"
    status = read(raw / "status.json")
    assert status["stage"] == "complete" and status["completed_arms"] == status["total_arms"] == 8
    assert status["optimizer_steps"] == 384 and status["trainable_parameters"] == 188416
    assert status["protocol_sha256"] == sha256((root / "protocol.json").read_bytes()).hexdigest()
    assert status["source_sha256"] == sha256((root / "train.py").read_bytes()).hexdigest()
    assert status["frozen_parameter_sha256"] == status["final_frozen_parameter_sha256"]
    assert status["base_unchanged"] and status["zero_adapter_exact"] and status["frozen_end_exact"]
    for row in data["tests"][:2]:
        original = read(raw / "frozen" / (row["id"] + ".json"))
        for folder in ("frozen-end", "zero-adapter"):
            assert read(raw / folder / (row["id"] + ".json")) == original
    initial, initial_hash = parameter_file(raw / "initial-adapter.safetensors")
    assert initial_hash == status["initial_adapter_sha256"] and sum(x.size for x in initial.values()) == 188416
    assert all(not np.any(x) for n, x in initial.items() if ".lora_B." in n)
    tests = {r["id"]: r for r in data["tests"]}
    details, summaries, continuation = [], [], []
    for life in data["lives"]:
        lid = life["id"]
        training_metrics = {}
        for arm in ("aligned", "shuffled"):
            folder = raw / (lid + "-" + arm)
            train = read(folder / "training.json")
            after, after_hash = parameter_file(folder / "adapter.safetensors")
            assert set(after) == set(initial)
            standard, _ = parameter_file(folder / "peft/adapter_model.safetensors")
            for name, array in after.items():
                standard_name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
                assert np.array_equal(standard[standard_name], array)
            config = read(folder / "peft/adapter_config.json")
            assert config["base_model_name_or_path"] == data["model"] and config["revision"] == data["revision"]
            assert config["r"] == 4 and config["lora_alpha"] == 8
            delta = sum(float(np.square(after[n].astype(np.float64) - initial[n].astype(np.float64)).sum()) for n in initial) ** .5
            assert abs(delta - train["delta_l2"]) < 2e-6
            assert train["initial_adapter_sha256"] == initial_hash and train["final_adapter_sha256"] == after_hash
            assert train["life_sha256"] == data["datasets"][lid]["life_sha256"] == digest(life)
            assert train["training_rows_sha256"] == digest(data["datasets"][lid][arm]["rows"])
            assert train["steps"] == 48 and train["reload_exact"] and train["frozen_gradients_absent"]
            step_rows = [json.loads(line) for line in (folder / "steps.jsonl").read_bytes().splitlines()]
            assert len(step_rows) == 48
            assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in step_rows)
            assert any(r["gradient_norm"] > 0 for r in step_rows)
            training_metrics[arm] = dict(adapter_delta_l2=delta, final_adapter_sha256=after_hash,
                                         loss_first_8_mean=mean(r["loss"] for r in step_rows[:8]),
                                         loss_last_8_mean=mean(r["loss"] for r in step_rows[-8:]))
        # Same starting weights, examples and optimizer sequence; only targets differ.
        steps_a = [json.loads(line)["row_ids"] for line in (raw / (lid + "-aligned") / "steps.jsonl").read_bytes().splitlines()]
        steps_c = [json.loads(line)["row_ids"] for line in (raw / (lid + "-shuffled") / "steps.jsonl").read_bytes().splitlines()]
        assert steps_a == steps_c
        for arm in ("frozen", "aligned", "shuffled"):
            folder = raw / ("frozen" if arm == "frozen" else lid + "-" + arm + "-tests")
            rows = []
            for tid, test in tests.items():
                record = read(folder / (tid + ".json"))
                check_policy(record)
                preferred = max(probabilities(life["mapping"])[test["partner"]], key=probabilities(life["mapping"])[test["partner"]].get)
                q = values(test["state"], probabilities(life["mapping"])[test["partner"]])
                best = max(q.values())
                loss = best - q[record["action"]]
                request_mass = record["probabilities"]["appeal"] + record["probabilities"]["exchange"]
                row = dict(life=lid, arm=arm, id=tid, order=test["order"], action=record["action"], regret=loss,
                           reference_best=[a for a, v in q.items() if abs(best - v) < 1e-10],
                           correct=loss < 1e-9, request_probability=request_mass,
                           preferred_given_request=record["probabilities"][preferred] / request_mass,
                           preferred_rank_correct=record["action_log_likelihood"][preferred] > record["action_log_likelihood"]["exchange" if preferred == "appeal" else "appeal"],
                           social_relevant=tid.split("-")[1] in ("02", "04", "05"),
                           expected_regret=best - sum(q[a] * record["probabilities"][a] for a in ACTIONS))
                rows.append(row)
                details.append(row)
            groups = {}
            for order in (0, 1):
                chosen = [r for r in rows if r["order"] == order]
                social = [r for r in chosen if r["social_relevant"]]
                groups[str(order)] = dict(n=len(chosen), correct=sum(r["correct"] for r in chosen),
                                          mean_regret=mean(r["regret"] for r in chosen),
                                          mean_expected_regret=mean(r["expected_regret"] for r in chosen),
                                          relevant_style_n=len(social), preferred_rank_correct=sum(r["preferred_rank_correct"] for r in social),
                                          mean_preferred_given_request=mean(r["preferred_given_request"] for r in social))
            by_id = {r["id"]: r for r in rows}
            body_pairs = []
            for partner in PARTNERS:
                for order in (0, 1):
                    low, high = [by_id[f"test-{i:02}-{partner}-order{order}"] for i in (1, 4)]
                    body_pairs.append(dict(partner=partner, order=order, low_action=low["action"], high_action=high["action"],
                                           both_correct=low["correct"] and high["correct"],
                                           request_probability_high_minus_low=high["request_probability"] - low["request_probability"]))
            summaries.append(dict(life=lid, seed=life["seed"], mapping=life["mapping"], arm=arm,
                                  by_order=groups, n=len(rows), correct=sum(r["correct"] for r in rows),
                                  mean_regret=mean(r["regret"] for r in rows),
                                  mean_expected_regret=mean(r["expected_regret"] for r in rows),
                                  body_pairs=body_pairs, training=training_metrics.get(arm)))
            trajectory = read(raw / (lid + "-" + arm + "-rollout.json"))
            assert trajectory["life_sha256"] == digest(life)
            continuation.append(dict(life=lid, arm=arm, **replay(life, trajectory["trace"], True)))
        for name in ("always_answer", "empirical_reference", "privileged_true_p_reference"):
            posterior = data["datasets"][lid]["aligned"]["posterior"] if name == "empirical_reference" else None
            continuation.append(dict(life=lid, arm=name, **replay(life, cpu_baseline(life, name, posterior), False)))
    pooled = {}
    for arm in ("frozen", "aligned", "shuffled"):
        rows = [r for r in summaries if r["arm"] == arm]
        pooled[arm] = dict(source_lives=len(rows), n=sum(r["n"] for r in rows), correct=sum(r["correct"] for r in rows),
                           mean_regret=mean(r["mean_regret"] for r in rows),
                           mean_expected_regret=mean(r["mean_expected_regret"] for r in rows),
                           body_pairs_correct=sum(x["both_correct"] for r in rows for x in r["body_pairs"]))
    return dict(protocol_sha256=status["protocol_sha256"], gpu_status=status,
                pooled_descriptive_only=pooled, by_life=summaries, continuation=continuation, details=details,
                notes=["Four source lives from two environment seeds and reversed mappings; prompts are not independent replications.",
                       "Reference regret uses supplied four-turn constant-load continuation, not the true 64-turn optimum.",
                       "Observed-feedback targets are constructed by a supplied partner estimator and body model, not autonomous RL.",
                       "Continuation alternates definition order every two turns, so each partner appears under both orders."])


def verify_archive(path):
    archive = zipfile.ZipFile(path)
    assert archive.testzip() is None
    manifest = json.loads(archive.read("archive-manifest.json"))
    assert len(archive.namelist()) == len(set(archive.namelist()))
    assert set(archive.namelist()) == set(manifest) | {"archive-manifest.json"}
    for name, expected in manifest.items():
        raw = archive.read(name)
        assert len(raw) == expected["bytes"] and sha256(raw).hexdigest() == expected["sha256"]
    return archive


def audit_inputs(root, tokenizer_dir):
    """Independently render and tokenize every recorded decision, without torch."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    manifest = read(REPO / "artifacts/model-studies/2026-11-10/tokenizer-files.json")
    checked_files = 0
    for name, expected in manifest.items():
        parts = Path(name.replace("\\", "/")).parts[-2:]
        if parts[0] == "model_0":
            assert sha256((tokenizer_dir / parts[1]).read_bytes()).hexdigest() == expected["sha256"]
            checked_files += 1
    assert checked_files == 2
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    config = read(tokenizer_dir / "tokenizer_config.json")
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    def fail(message):
        raise ValueError(message)
    environment.globals["raise_exception"] = fail
    template = environment.from_string(config["chat_template"])
    data = read(root / "protocol.json")
    n = 0
    def verify(record, prompt):
        nonlocal n
        text = template.render(messages=prompt, add_generation_prompt=True, enable_thinking=False,
                               eos_token=config["eos_token"], bos_token=config.get("bos_token"), tools=None)
        assert sha256(text.encode()).hexdigest() == record["prompt_sha256"]
        assert tokenizer.encode(text, add_special_tokens=False).ids == record["input_token_ids"]
        n += 1
    for row in data["tests"]:
        verify(read(root / "output/frozen" / (row["id"] + ".json")), row["messages"])
    for life in data["lives"]:
        for arm in ("frozen", "aligned", "shuffled"):
            if arm != "frozen":
                for row in data["tests"]:
                    verify(read(root / "output" / (life["id"] + "-" + arm + "-tests") / (row["id"] + ".json")), row["messages"])
            rollout = read(root / "output" / (life["id"] + "-" + arm + "-rollout.json"))
            for row in rollout["trace"]:
                obs = row["before"]
                t = obs["age"]
                state = dict(damage=obs["damage"], load=obs["load"], weight=18., turns=min(4, obs["horizon"] - t),
                             protection_left=obs["protection_left"])
                verify(row, messages(state, row["partner"], row["task"][0], (t // 2) % 2))
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw", type=Path)
    group.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, help="optional pinned model_0 cache for independent input audit")
    args = parser.parse_args()
    if args.archive:
        with verify_archive(args.archive) as archive:
            result = score(zipfile.Path(archive))
            if args.tokenizer_dir:
                result["independently_tokenized_inputs"] = audit_inputs(zipfile.Path(archive), args.tokenizer_dir)
    else:
        result = score(args.raw)
        if args.tokenizer_dir:
            result["independently_tokenized_inputs"] = audit_inputs(args.raw, args.tokenizer_dir)
    with args.output.open("xb") as handle:
        handle.write(canonical(result))
    print(json.dumps({"probes": result["pooled_descriptive_only"], "continuation_lives_replayed": len(result["continuation"])}))
