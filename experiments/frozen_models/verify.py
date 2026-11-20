"""CPU-only audit of the published frozen-model evidence; never modifies it."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "artifacts" / "model-studies" / "2026-11-10"
sys.dont_write_bytecode = True
sys.path.insert(0, str(DATA / "environment"))
from sla_demo.environment import ACTIONS, Config, PROFILES, State, World, reference_action
from baselines import action as baseline_action, generate as generate_baselines


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def parse(text, schema):
    try:
        obj = json.loads(text, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if type(obj) is not dict or set(obj) != set(schema["required"]):
            return None
        for key, spec in schema["properties"].items():
            value, kind = obj[key], spec["type"]
            if kind == "boolean" and type(value) is not bool:
                return None
            if kind == "integer" and type(value) is not int:
                return None
            if kind == "number" and (type(value) not in (int, float) or not math.isfinite(value)):
                return None
            if kind == "string" and type(value) is not str:
                return None
            if "enum" in spec and value not in spec["enum"]:
                return None
        return obj
    except (ValueError, TypeError):
        return None


def equal(a, b, tolerance=0.0001):
    if type(a) in (int, float) and type(b) in (int, float):
        return abs(a - b) <= tolerance
    return type(a) is type(b) and a == b


def match(a, b):
    assert set(a) == set(b)
    for key, value in a.items():
        if type(value) in (int, float):
            assert math.isclose(value, b[key], abs_tol=1e-12, rel_tol=0), (key, value, b[key])
        else:
            assert value == b[key], (key, value, b[key])


ACTION_SCHEMA = {"required": ["action"], "properties": {"action": {"type": "string", "enum": list(ACTIONS)}}}


def final_text(call):
    return call.get("final_channel_text", call.get("canonical_generated_text", call["output"]))


def replay(life, model=True):
    world = World(Config(**life["config"]), life["seed"])
    np.testing.assert_array_equal(world.events, np.array(life["event_tape"]))
    total = 0.0
    for event in life["trajectory"]:
        match(world.observation(), event["before"])
        action = parse(final_text(event["model_call"]), ACTION_SCHEMA) if model else None
        if event.get("failed"):
            assert action is None and life["termination"] == "protocol_failure"
            break
        if model:
            assert action is not None
            action = ACTIONS.index(action["action"])
        else:
            action = event["action"]
        after, reward, _, info = world.step(action)
        match(after, event["after"])
        match(info, event["info"])
        assert abs(reward - event["reward"]) < 1e-12
        total += reward
    assert abs(total - life["return"]) < 1e-12
    assert world.state.age == life["settled_turns"]
    expected = "protocol_failure" if life.get("failed") else "horizon" if world.state.alive else "death"
    assert expected == life["termination"]


def payload(case):
    text = case["user"]
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def returns(p, first):
    profile = next(k for k, v in PROFILES.items() if v == p["damage_weight"])
    world = World(Config(horizon=8, load=p["constant_load_each_turn"], profile=profile,
                         protection_duration=p["protection_duration"], partner="supportive"), 77)
    world.state = State(age=8 - p["remaining_turns"], damage=p["damage"], protection_left=p["protection_left"])
    world.events[:, 0] = 1.0
    world.events[:, 1] = 0.0 if p["if_a_request_occurs_it_is_accepted"] else 1.0
    total = 0.0
    for turn in range(p["remaining_turns"]):
        _, reward, done, _ = world.step(ACTIONS.index(first) if turn == 0 else 0)
        total += reward
        if done:
            break
    return total


def verify_answers(cases, keys, v2=False):
    checked = 0
    for case in cases.values():
        category = case["stage"] if v2 else case["category"]
        expected = keys[case["id"]]["expected"]
        if category in ("transition", "single_step"):
            p = case["meta"] if v2 else payload(case)
            world = World(Config(horizon=8, load=p["current_load"], partner="supportive",
                                 protection_duration=p["protection_duration"]), 77)
            world.state = State(age=6, damage=p["damage"], protection_left=p["protection_left"])
            world.events[:, 0] = 1.0
            world.events[:, 1] = 0.0 if p["request_accepted_if_applicable"] else 1.0
            after, _, _, info = world.step(ACTIONS.index(p["action"]))
            actual = ({"added_damage": info["damage_added"], "damage_after": after["damage"],
                       "protection_left_after": after["protection_left"]} if v2 else
                      {k: after[k] for k in ("damage", "protection_left", "alive")})
            assert all(equal(actual[k], value, 1e-12) for k, value in expected.items())
            checked += 1
        elif category in ("delayed_choice", "given_totals", "end_to_end"):
            p = payload(case)
            given = p.get("given_total_returns")
            p = p.get("scenario", p)
            values = {label: returns(p, p[label]["first_action"]) for label in ("A", "B")}
            assert expected["plan"] == max(values, key=values.get)
            if given:
                assert all(abs(given[k] - values[k]) < 1e-8 for k in values)
            if category == "end_to_end":
                assert abs(expected["return_A"] - values["A"]) < 1e-12
                assert abs(expected["return_B"] - values["B"]) < 1e-12
            checked += 1
    return checked


def model_status(folder, candidate):
    status = read(folder / "status.json")
    assert status["stage"] == "completed" and status["revision"] == candidate["revision"] and status["model"] == candidate["id"]
    assert status["trainable_parameters"] == 0 and status["parameter_version_counters_unchanged"]
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = read(folder / "loading-info.json")[key]
        assert value is None or value == [] or value == "set()", (key, value)
    return status


def compare_scores(records, cases, keys, published, condition=None):
    assert len(records) == len({r["id"] for r in records})
    details = []
    for record in records:
        cid = record["id"]
        obj = parse(final_text(record), cases[cid]["response_schema"])
        correct = obj is not None and all(equal(obj[k], v) for k, v in keys[cid]["expected"].items())
        stored = published[cid] if condition is None else published[(condition, cid)]
        assert stored["schema_valid"] == (obj is not None) and stored["correct"] == correct, cid
        details.append((obj is not None, correct))
    return sum(x[0] for x in details), sum(x[1] for x in details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers-dir", type=Path, help="Optional pinned model_0/1/2 tokenizer files; no downloads")
    args = parser.parse_args()
    for name, info in read(DATA / "SHA256SUMS.json").items():
        path = DATA / name
        assert path.is_file() and path.stat().st_size == info["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
    for name, info in read(DATA / "source-map.json").items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == info["published_sha256"]
        if info["byte_identical"]:
            assert info["source_sha256"] == info["published_sha256"]

    formal_calls = []
    v1 = DATA / "frozen-v1"
    cases = {c["id"]: c for c in read(v1 / "inputs/qualification-cases-v1.json")["cases"]}
    keys = {k["id"]: k for k in read(v1 / "inputs/qualification-answer-key-v1.json")["answers"]}
    assert len(cases) == 48
    oracle_questions = verify_answers(cases, keys)
    candidates = read(v1 / "inputs/model-candidates-v2.json")["models"]
    summary = read(v1 / "results/scored-results.json")["models"]
    scored = read(v1 / "results/case-scores.json")
    v1_counts = {}
    tapes = {}
    for index in (2, 1, 0):
        folder = v1 / "raw" / f"model_{index}"
        status = model_status(folder, candidates[index])
        result = next(m for m in summary if m["index"] == index)
        questions = rows(folder / "qualification-decoded.jsonl")
        originals = rows(folder / "qualification.jsonl")
        assert {q["id"] for q in questions} == set(cases) and len(questions) == 48
        for a, b in zip(originals, questions):
            assert a["id"] == b["id"] and a["output_token_ids"] == b["output_token_ids"] and a["messages"] == b["messages"]
        published = {d["id"]: d for d in scored if d["model"] == status["model"]}
        valid, correct = compare_scores(questions, cases, keys, published)
        assert valid == result["schema_valid"] and correct == result["correct"]
        lives = rows(folder / "rollout-decoded.jsonl")
        raw_lives = rows(folder / "rollout.jsonl")
        assert len(lives) == 8 and len(raw_lives) == 8
        for raw, life in zip(raw_lives, lives):
            assert raw["life_index"] == life["life_index"] and len(raw["trajectory"]) == len(life["trajectory"])
            for a, b in zip(raw["trajectory"], life["trajectory"]):
                assert a["model_call"]["output_token_ids"] == b["model_call"]["output_token_ids"]
            replay(life)
            key = json.dumps(life["config"], sort_keys=True), life["seed"]
            if key in tapes:
                assert tapes[key] == life["event_tape"]
            tapes[key] = life["event_tape"]
        assert sum(bool(l["failed"]) for l in lives) == result["rollout_protocol_failures"]
        assert abs(statistics.mean(l["return"] for l in lives) - result["rollout_mean_return_all_attempts"]) < 1e-12
        formal_calls.extend((index, q) for q in questions)
        formal_calls.extend((index, e["model_call"]) for l in lives for e in l["trajectory"])
        v1_counts[status["model"]] = correct

    v2 = DATA / "diagnostic-v2"
    cases = {c["id"]: c for c in read(v2 / "inputs/public-cases.json")["cases"]}
    keys = {k["id"]: k for k in read(v2 / "inputs/answer-key-local.json")["answers"]}
    oracle_questions += verify_answers(cases, keys, v2=True)
    matrix = {m["life_index"]: m for m in read(v2 / "inputs/rollout-matrix.json")["lives"]}
    assert len(matrix) == 32 and len({m["seed"] for m in matrix.values()}) == 8
    candidates = read(v2 / "inputs/model-candidates.json")["models"]
    results = read(v2 / "results/scored-diagnostics.json")
    scored = read(v2 / "results/case-scores.json")
    v2_choices = {}
    for index in (1, 0):
        folder = v2 / "raw" / f"model_{index}"
        status = model_status(folder, candidates[index])
        result = next(m for m in results["models"] if m["index"] == index)
        published = {(d["condition"], d["id"]): d for d in scored if d["index"] == index}
        for name, condition in (("diagnostic_greedy.jsonl", "D2-G"), ("diagnostic_thinking.jsonl", "D2-T")):
            if condition == "D2-T" and index == 1:
                continue
            generated = rows(folder / name)
            expected_ids = set(cases) if condition == "D2-G" else {c["id"] for c in cases.values() if c["stage"] == "end_to_end" and c["language"] == "en"}
            assert {r["id"] for r in generated} == expected_ids and len(generated) == len(expected_ids)
            compare_scores(generated, cases, keys, published, condition)
            for stage, recorded in result["conditions"][condition]["stages"].items():
                group = [g for g in generated if cases[g["id"]]["stage"] == stage]
                objects = [(g, parse(final_text(g), cases[g["id"]]["response_schema"])) for g in group]
                assert recorded["n"] == len(group)
                assert recorded["schema_valid"] == sum(obj is not None for _, obj in objects)
                plans = sum(obj is not None and obj.get("plan") == keys[g["id"]]["expected"].get("plan") for g, obj in objects if "plan" in keys[g["id"]]["expected"])
                assert recorded["plan_correct"] == plans
                numeric = 0
                all_correct = 0
                for g, obj in objects:
                    expected = keys[g["id"]]["expected"]
                    number_keys = [k for k, value in expected.items() if type(value) in (int, float)]
                    numeric += bool(number_keys) and obj is not None and all(equal(obj[k], expected[k]) for k in number_keys)
                    all_correct += obj is not None and all(equal(obj[k], value) for k, value in expected.items())
                assert recorded["numeric_correct"] == numeric and recorded["all_fields_correct"] == all_correct
            formal_calls.extend((index, g) for g in generated)
        lives = rows(folder / "rollout.jsonl")
        assert len(lives) == 32 and {l["life_index"] for l in lives} == set(matrix)
        for life in lives:
            m = matrix[life["life_index"]]
            assert life["config"] == m["config"] and life["seed"] == m["seed"]
            replay(life)
            key = json.dumps(life["config"], sort_keys=True), life["seed"]
            if key in tapes:
                assert tapes[key] == life["event_tape"]
            tapes[key] = life["event_tape"]
        assert abs(statistics.mean(l["return"] for l in lives) - result["rollout"]["mean_return_including_failed_attempts"]) < 1e-12
        formal_calls.extend((index, e["model_call"]) for l in lives for e in l["trajectory"])
        v2_choices[status["model"]] = result["conditions"]["D2-G"]["stages"]["end_to_end"]["plan_correct"]
        earlier = rows(v2 / "interruption" / f"model_{index}_diagnostic_greedy.jsonl")
        now = {g["id"]: g for g in rows(folder / "diagnostic_greedy.jsonl")}
        assert len(earlier) == 48
        assert all(g["output_token_ids"] == now[g["id"]]["output_token_ids"] for g in earlier)

    for policy in read(v2 / "results/rule-baselines.json")["policies"]:
        assert len(policy["lives"]) == 32
        for life in policy["lives"]:
            replay(life, model=False)
            for event in life["trajectory"]:
                assert event["action"] == baseline_action(policy["name"], event["before"], life["config"]["profile"])
    assert generate_baselines(read(v2 / "inputs/rollout-matrix.json"))["policies"] == read(v2 / "results/rule-baselines.json")["policies"]
    assert len(formal_calls) == 898

    token_audited = 0
    if args.tokenizers_dir:
        from tokenizers import Tokenizer
        tokenizers, stops = {}, {}
        for name, info in read(DATA / "tokenizer-files.json").items():
            parts = Path(name.replace("\\", "/")).parts[-2:]
            path = args.tokenizers_dir.joinpath(*parts)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
        for index in (0, 1, 2):
            path = args.tokenizers_dir / f"model_{index}"
            tokenizers[index] = Tokenizer.from_file(str(path / "tokenizer.json"))
            cfg = read(path / "tokenizer_config.json")
            names = [cfg.get("eos_token"), cfg.get("pad_token")]
            names = [v.get("content") if isinstance(v, dict) else v for v in names]
            stops[index] = {tokenizers[index].token_to_id(v) for v in names if v}
        for index, call in formal_calls:
            tok, stop = tokenizers[index], stops[index]
            ids = call["output_token_ids"]
            assert tok.decode(ids, skip_special_tokens=True) == call["output"]
            def canonical(tokens):
                cut = len(tokens)
                while cut and (tokens[cut - 1] in stop or tok.decode([tokens[cut - 1]], skip_special_tokens=False).isspace()):
                    cut -= 1
                return tok.decode(tokens[:cut], skip_special_tokens=False) + tok.decode([v for v in tokens[cut:] if v not in stop], skip_special_tokens=False)
            assert canonical(ids) == call["canonical_generated_text"]
            if call.get("thinking"):
                positions = [i for i, v in enumerate(ids) if v == tok.token_to_id("</think>")]
                assert call["reasoning_closed"] == (len(positions) == 1)
                assert call["final_channel_text"] == (canonical(ids[positions[0] + 1:]) if len(positions) == 1 else "")
            token_audited += 1

    print(json.dumps({"hashes_verified": True, "oracle_questions_crosschecked": oracle_questions,
                      "model_lives_replayed": 88, "rule_lives_replayed": 128,
                      "v1_correct_out_of_48": v1_counts, "v2_self_computed_choice_out_of_16": v2_choices,
                      "formal_calls": len(formal_calls), "independently_token_decoded": token_audited,
                      "gpu_or_network_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
