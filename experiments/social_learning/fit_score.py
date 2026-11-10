"""CPU-only fitting, paired-condition, numerical-return and evidence checks."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from protocol import canonical


def distribution(record, actions, word_only=False):
    key = "word_log_likelihood" if word_only else "action_log_likelihood"
    x = np.array([record[key][a] for a in actions], dtype=float)
    assert np.isfinite(x).all()
    x -= x.max()
    return np.exp(x) / np.exp(x).sum()


def aggregate(rows, records, actions, acceptance):
    assert [r["id"] for r in rows] == [r["id"] for r in records]
    q = np.array([r["target"] for r in rows])
    p = np.array([distribution(r, actions) for r in records])
    assert (q > 0).all() and np.allclose(q.sum(1), 1., rtol=0, atol=1e-12)
    kl = (q * (np.log(q)-np.log(p))).sum(1)
    null = q.mean(0)
    null_kl = float((q * (np.log(q)-np.log(null))).sum(1).mean())
    correct = [bool(q[i, int(p[i].argmax())] >= q[i].max()-1e-10) for i in range(len(rows))]
    index = {(r["context"], r["partner"], r["order"]): i for i,r in enumerate(rows)}
    order_pairs, partner_pairs, body_pairs = [], [], []
    for i,row in enumerate(rows):
        if row["order"] == 0:
            j = index[(row["context"], row["partner"], 1)]
            order_pairs.append((i,j))
        if row["partner"] == "P17":
            partner_pairs.append((i,index[(row["context"], "P42", row["order"])]))
        if row["context"] == "needs_protection":
            for control in ("no_load", "last_turn", "already_protected"):
                body_pairs.append((i,index[(control, row["partner"], row["order"])]))

    def paired(pairs):
        required = [(i,j) for i,j in pairs if not set(np.flatnonzero(q[i] >= q[i].max()-1e-10)) &
                    set(np.flatnonzero(q[j] >= q[j].max()-1e-10))]
        return dict(pairs=len(pairs), mean_tv=float(np.mean([.5*abs(p[i]-p[j]).sum() for i,j in pairs])) if pairs else None,
            max_tv=float(max(.5*abs(p[i]-p[j]).sum() for i,j in pairs)) if pairs else None,
            action_changes=int(sum(p[i].argmax() != p[j].argmax() for i,j in pairs)),
            required_switch_pairs=len(required), both_correct_required=sum(correct[i] and correct[j] for i,j in required))
    op, pp, bp = paired(order_pairs), paired(partner_pairs), paired(body_pairs)
    accepted = bool(float(kl.mean()) <= acceptance["mean_kl_at_most"] and all(correct) and
        op["max_tv"] <= acceptance["max_order_tv_at_most"] and pp["both_correct_required"] == pp["required_switch_pairs"])
    return dict(n=len(rows), teacher_action_correct=sum(correct), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
        mean_cross_entropy=float(-(q*np.log(p)).sum(1).mean()), constant_distribution_kl=null_kl,
        mean_absolute_probability_error=float(abs(q-p).mean()),
        action_counts={a: sum(actions[int(x.argmax())] == a for x in p) for a in actions},
        order=op, partner=pp, body=bp, engineering_acceptance_passed=accepted)


def numeric_returns(rows, records, tolerance=.001):
    assert [r["id"] for r in rows] == [r["id"] for r in records]
    result = []
    for row,record in zip(rows,records):
        valid, error, extracted_error = False, None, None
        def check(parsed):
            return (isinstance(parsed,dict) and set(parsed) == set(row["values"]) and
                all(type(x) in (float,int) and math.isfinite(x) for x in parsed.values()))
        try:
            parsed = json.loads(record["text"].strip())
            valid = check(parsed)
            if valid:
                error = max(abs(parsed[a]-q) for a,q in row["values"].items())
        except (ValueError, TypeError):
            pass
        # Secondary format diagnostic only. The preregistered strict JSON score
        # is unchanged; this prevents decoration alone being called bad math.
        objects=re.findall(r"\{[^{}]*\}",record["text"],flags=re.DOTALL)
        if len(objects)==1:
            try:
                parsed=json.loads(objects[0])
                if check(parsed):
                    extracted_error=max(abs(parsed[a]-q) for a,q in row["values"].items())
            except (ValueError,TypeError):
                pass
        result.append(dict(id=row["id"], strict_json_valid=bool(valid), max_absolute_error=error,
            all_four_within_tolerance=bool(valid and error <= tolerance),terminated=record["terminated"],
            secondary_extractable_json_error=extracted_error,
            secondary_extracted_all_four_within_tolerance=bool(extracted_error is not None and extracted_error <= tolerance)))
    return dict(n=len(rows), tolerance=tolerance, strict_json_valid=sum(r["strict_json_valid"] for r in result),
        all_four_within_tolerance=sum(r["all_four_within_tolerance"] for r in result),
        secondary_extractable_json_valid=sum(r["secondary_extractable_json_error"] is not None for r in result),
        secondary_extracted_all_four_within_tolerance=sum(r["secondary_extracted_all_four_within_tolerance"] for r in result), cases=result)


def score(archive_path, protocol_path):
    raw = protocol_path.read_bytes()
    data = json.loads(raw)
    archive = zipfile.ZipFile(archive_path)
    manifest = json.loads(archive.read("archive-manifest.json"))
    assert set(archive.namelist()) == set(manifest) | {"archive-manifest.json"}
    for name, info in manifest.items():
        value = archive.read(name)
        assert sha256(value).hexdigest() == info["sha256"] and len(value) == info["bytes"], name
    assert archive.read("protocol.json") == raw
    read = lambda n: json.loads(archive.read(n))
    state = read("output/status.json")
    assert state["stage"] == "complete" and state["base_unchanged"] and state["frozen_end_exact"] and state["zero_adapter_exact"]
    assert state["protocol_sha256"] == sha256(raw).hexdigest()
    assert state["worker_sha256"] == sha256(archive.read("fit_train.py")).hexdigest()
    assert state["frozen_parameter_sha256"] == state["final_frozen_parameter_sha256"] == data["expected_base_tensor_sha256"]
    assert state["initial_adapter_sha256"] == data["initial_adapter_tensor_sha256"]
    lock=read("run-lock.json")
    for name,expected in lock["files"].items():
        if name != "source-study.zip":
            assert sha256(archive.read(name)).hexdigest() == expected,name
    assert state["optimizer_steps"] == len(data["jobs"]) * data["max_steps"]
    assert state["completed_jobs"] == len(data["jobs"])
    jobs,zero_scores = {},{}
    for job in data["jobs"]:
        dataset = data["datasets"][job["dataset"]]
        actions = dataset["actions"]
        base = "output/" + job["id"]
        checks = read(base + "/checks.json")
        assert all(checks[k] for k in ("reload_exact", "frozen_gradients_absent", "frozen_versions_unchanged"))
        assert checks["optimizer_steps"] == data["max_steps"]
        assert checks["initial_adapter_sha256"] == data["initial_adapter_tensor_sha256"]
        steps = [json.loads(s) for s in archive.read(base + "/steps.jsonl").splitlines()]
        assert len(steps) == data["max_steps"] and [r["step"] for r in steps] == list(range(1,data["max_steps"]+1))
        assert all(math.isfinite(r["loss"]) and math.isfinite(r["pre_clip_gradient_norm"]) for r in steps)
        train_ids = {r["id"] for r in dataset["train"]}
        assert all(len(r["row_ids"]) == data["training_batch_size"] and set(r["row_ids"]) <= train_ids for r in steps)
        counts = {rid: sum(rid in r["row_ids"] for r in steps) for rid in sorted(train_ids)}
        assert len(set(counts.values())) == 1
        result = dict(dataset=job["dataset"], task=dataset["task"], mapping=dataset["mapping"],
            learning_rate=job["learning_rate"], checkpoints={}, example_exposures=counts,
            gradient_clipped_steps=checks["gradient_clipped_steps"], checks=checks)
        for step in data["checkpoints"]:
            folder = f"{base}/step-{step:03}"
            cp = {}
            for group in ("train", "new_question", "provided_returns"):
                if dataset[group]:
                    records = read(folder + "/" + group + ".json")
                    for record in records:
                        p=distribution(record,actions)
                        assert record["action"] == actions[int(p.argmax())]
                        assert max(abs(record["probabilities"][a]-p[i]) for i,a in enumerate(actions)) < 3e-7
                        if step == 0:
                            signature=(record["prompt_sha256"],tuple(actions))
                            vector=[record["action_log_likelihood"][a] for a in actions]
                            assert zero_scores.setdefault(signature,vector) == vector,signature
                        for a in actions:
                            assert math.isclose(sum(record["token_log_probabilities"][a]),record["action_log_likelihood"][a],abs_tol=2e-5)
                            assert math.isclose(sum(record["token_log_probabilities"][a][:-1]),record["word_log_likelihood"][a],abs_tol=2e-5)
                    cp[group] = aggregate(dataset[group], records, actions, data["engineering_acceptance"])
            if step:
                cp["parameters"] = read(folder + "/parameters.json")
                assert cp["parameters"]["delta_l2"] > 0 and cp["parameters"]["frozen_versions_unchanged"]
            if step in (0,data["max_steps"]) and dataset["calculate"]:
                cp["numeric_returns"] = numeric_returns(dataset["calculate"], read(folder + "/calculate.json"))
            if step == data["max_steps"]:
                single, batched = (read(folder + "/" + n + ".json") for n in ("train", "train-batch2"))
                cp["batch2"] = aggregate(dataset["train"], batched, actions, data["engineering_acceptance"])
                cp["batch2_action_changes"] = sum(a["action"] != b["action"] for a,b in zip(single,batched))
                cp["batch2_max_full_score_delta"] = max(abs(a["action_log_likelihood"][k]-b["action_log_likelihood"][k]) for a,b in zip(single,batched) for k in actions)
            result["checkpoints"][str(step)] = cp
        jobs[job["id"]] = result
    return dict(schema="crossed-mini-fit-scored-v1", archive_sha256=sha256(archive_path.read_bytes()).hexdigest(),
        protocol_sha256=sha256(raw).hexdigest(), archive_files_verified=len(manifest), jobs=jobs,
        engineering_acceptance=data["engineering_acceptance"], status=state,
        scope="Descriptive fixed-example fitting and staged arithmetic diagnostics; no new life or independent efficacy test.")


def audit_tokens(archive_path, protocol_path, tokenizer_dir):
    """Reconstruct prompts without transformers using pinned Jinja/Rust assets."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    root=Path(__file__).resolve().parents[2]
    metadata=json.loads((root/"artifacts/model-studies/2026-11-10/tokenizer-files.json").read_bytes())
    for name in ("tokenizer.json","tokenizer_config.json"):
        assert sha256((tokenizer_dir/name).read_bytes()).hexdigest() == metadata["tokenizers/model_0/"+name]["sha256"]
    tok=Tokenizer.from_file(str(tokenizer_dir/"tokenizer.json"))
    config=json.loads((tokenizer_dir/"tokenizer_config.json").read_bytes())
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=["jinja2.ext.loopcontrols"])
    def fail(message):
        raise ValueError(message)
    env.globals["raise_exception"]=fail
    template=env.from_string(config["chat_template"])
    data=json.loads(protocol_path.read_bytes())
    expected,distinct={},set()
    for did,dataset in data["datasets"].items():
        for group in ("train","new_question","provided_returns","calculate"):
            for row in dataset[group]:
                text=template.render(messages=row["messages"],add_generation_prompt=True,enable_thinking=False,
                    eos_token=config["eos_token"],bos_token=config.get("bos_token"),tools=None)
                h=sha256(text.encode()).hexdigest()
                expected[(did,group,row["id"])]=(h,tok.encode(text,add_special_tokens=False).ids)
                distinct.add(h)
    records=0
    with zipfile.ZipFile(archive_path) as archive:
        for job in data["jobs"]:
            for step in data["checkpoints"]:
                folder=f"output/{job['id']}/step-{step:03}/"
                for name in ("train","new_question","provided_returns","calculate","train-batch2"):
                    path=folder+name+".json"
                    if path not in archive.namelist():
                        continue
                    group="train" if name == "train-batch2" else name
                    for record in json.loads(archive.read(path)):
                        h,ids=expected[(job["dataset"],group,record["id"])]
                        assert record["prompt_sha256"] == h and record["input_token_ids"] == ids,path
                        records+=1
    return dict(distinct_prompts=len(distinct),records_checked=records,transformers_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    result = score(args.archive,args.protocol)
    if args.tokenizer_dir:
        result["independent_token_audit"]=audit_tokens(args.archive,args.protocol,args.tokenizer_dir)
    args.output.write_bytes(canonical(result))
    print(json.dumps(dict(jobs=len(result["jobs"]), verified_files=result["archive_files_verified"],
                         optimizer_steps=result["status"]["optimizer_steps"])))
