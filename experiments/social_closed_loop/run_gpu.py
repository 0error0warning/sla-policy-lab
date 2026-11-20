"""Run real Qwen prediction, periodic online LoRA, and matched replay.

This worker uses an explicitly supplied GPU; it cannot allocate resources or
discover credentials. Source/protocol/checkpoint hashes are retained in output.
"""

from contextlib import nullcontext
from hashlib import sha256
from importlib import metadata
import argparse
import json
from pathlib import Path
import time

import numpy as np

from core import (BodyValue, CONFIG, Evidence, FastReadout, MODEL, PARTNERS,
                  REVISION, canonical, digest, execute_event, features, make_rows,
                  rng, schedule, sigmoid, write)


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(items):
    import torch
    h = sha256()
    for name, tensor in sorted(items):
        h.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(root):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    output = root/"output"
    output.mkdir(exist_ok=False)
    config = json.loads((root/"protocol.json").read_bytes())
    assert config == CONFIG
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    status = dict(stage="download", completed_jobs=0, total_jobs=40,
                  optimizer_steps=0, gpu=torch.cuda.get_device_name(),
                  gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  protocol_sha256=file_hash(root/"protocol.json"),
                  versions={n:metadata.version(n) for n in ("torch", "transformers", "peft", "numpy",
                    "safetensors", "accelerate", "huggingface-hub", "flash-linear-attention", "fla-core")})

    def progress(**kwargs):
        status.update(kwargs, elapsed_seconds=time.monotonic()-started)
        write(output/"status.json", status)

    progress()
    expected = json.loads((root/"expected-weights.json").read_bytes())
    assert expected["id"] == MODEL and expected["revision"] == REVISION
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, token=False,
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
    shards = {p.name:dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in snapshot.glob("*.safetensors")}
    assert shards == expected["files"], "immutable checkpoint bytes differ"
    write(output/"weights-manifest.json", dict(id=MODEL, revision=REVISION, files=shards))
    progress(stage="load")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, output_loading_info=True, dtype=torch.bfloat16,
        device_map={"":0}, attn_implementation="sdpa")
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    base.requires_grad_(False)
    base.eval()
    torch.manual_seed(CONFIG["adapter_seed"])
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0., bias="none",
        target_modules=r"model\.language_model\.layers\.(0|1|2|3)\.mlp\.down_proj"))
    model.eval()
    trainable = {n:p for n, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 8 and sum(p.numel() for p in trainable.values()) == 188416
    assert all("lora_" in n and p.dtype == torch.float32 for n, p in trainable.items())
    fixed = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    fixed_versions = {n:p._version for n,p in fixed}
    base_hash = tensor_hash(fixed)
    initial = {n:p.detach().cpu().clone() for n,p in trainable.items()}
    save_file(initial, str(output/"initial-adapter.safetensors"))
    token_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("accept", "reject")]
    assert token_ids == [[10024], [45427]]
    head = base.lm_head.weight.detach()[torch.tensor([10024,45427], device="cuda")].float()
    difference = (head[0]-head[1]).detach()
    rows = make_rows()
    write(output/"rows.json", rows)
    prefixes = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) < 300
        for label, suffix in zip(("accept", "reject"), token_ids):
            assert tokenizer.encode(text+label, add_special_tokens=False) == ids+suffix
        prefixes.append(ids)
    cache = {}

    def batch(indices):
        key = tuple(indices)
        if key not in cache:
            seqs = [prefixes[i] for i in indices]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs),width), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, seq in enumerate(seqs):
                ids[j,:len(seq)] = torch.tensor(seq, device="cuda")
                mask[j,:len(seq)] = 1
            cache[key] = (ids, mask, torch.tensor([len(s)-1 for s in seqs], device="cuda"))
        return cache[key]

    def hidden(indices):
        ids, mask, last = batch(indices)
        value = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return value[torch.arange(len(indices), device="cuda"),last].float()

    def score(indices):
        return hidden(indices)@difference

    def observe(indices):
        with torch.no_grad():
            chunks = [score(indices[i:i+6]).cpu().numpy().astype(float) for i in range(0,len(indices),6)]
        return np.concatenate(chunks)

    def install(snapshot):
        with torch.no_grad():
            for n,p in trainable.items():
                p.grad = None
                p.copy_(snapshot[n])

    def snapshot_adapter():
        return {n:p.detach().cpu().contiguous().clone() for n,p in trainable.items()}

    with torch.no_grad(), model.disable_adapter():
        frozen_hidden = torch.cat([hidden(list(range(i,min(len(rows),i+6)))) for i in range(0,len(rows),6)])
        # Match the fixed six-row readout shape used in subsequent forwards;
        # a single 612-row GEMM has a different floating-point reduction order.
        frozen = torch.cat([frozen_hidden[i:i+6]@difference for i in range(0,len(rows),6)]).cpu().numpy().astype(float)
    x, centers = features(frozen_hidden.cpu().numpy().astype(float), rows)
    np.save(output/"frozen-hidden.npy", frozen_hidden.cpu().numpy(), allow_pickle=False)
    write(output/"frozen.json", dict(log_odds=frozen.tolist(), feature_centers=centers))
    del frozen_hidden
    assert np.array_equal(observe(list(range(6))), frozen[:6])
    body = BodyValue()
    np.savez_compressed(output/"body-value-grid.npz", grid=body.grid, reference=body.reference,
                        maintain=body.maintain, skip=body.skip)
    lookup = {(r["partner"],r["frame"],r["task"],r["wording"],r["order"],r["action"]):i
              for i,r in enumerate(rows) if r["split"] == "heldout"}
    train_ids = [i for i,r in enumerate(rows) if r["split"] == "train"]
    compact_probe_ids = train_ids+[i for i,r in enumerate(rows) if
        (r["split"] == "heldout" and r["task"] == 0) or r["split"] in ("paraphrase","content")]
    progress(stage="loops", trainable_parameters=sum(p.numel() for p in trainable.values()),
             base_tensor_sha256=base_hash, initial_adapter_sha256=tensor_hash(initial.items()),
             row_count=len(rows), training_row_count=len(train_ids))

    def probe(method, evidence, fast, turn):
        ids = list(range(len(rows))) if turn in (0,60) else compact_probe_ids
        if method == "hierarchical":
            probabilities = evidence.target([rows[i] for i in ids])
            logits = np.log(probabilities/(1-probabilities))
        elif method == "fast":
            logits = fast.predict()[ids]
        elif method == "frozen":
            logits = frozen[ids]
        else:
            logits = observe(ids)
        probabilities = sigmoid(logits)
        targets = evidence.target([rows[i] for i in ids])
        by_split = {}
        for split in ("train", "heldout", "paraphrase", "content"):
            pick = [j for j,i in enumerate(ids) if rows[i]["split"] == split]
            by_split[split] = float(np.abs(probabilities[pick]-targets[pick]).mean())
        return dict(turn=turn, row_ids=ids, log_odds=logits.tolist(), targets=targets.tolist(),
                    mean_absolute_error=by_split, evidence=evidence.export())

    def optimize(evidence, optimizer, turn, seed):
        logs = []
        # Six steps: one batch for each partner x each option order. Each batch
        # crosses both incoming framings and all three response-bearing acts.
        task = (turn//5-1)%2
        partner_order = list(rng(seed, "training-partner-order", turn).permutation(3))
        for order in range(2):
            for j in partner_order:
                partner = PARTNERS[j]
                ids = [i for i in train_ids if rows[i]["partner"] == partner and
                       rows[i]["task"] == task and rows[i]["order"] == order]
                assert len(ids) == 6
                target = torch.tensor(evidence.target([rows[i] for i in ids]), device="cuda", dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = score(ids)
                loss = F.binary_cross_entropy_with_logits(logits,target)
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True)
                optimizer.step()
                logs.append(dict(turn=turn, row_ids=ids, targets=target.cpu().tolist(),
                                 loss=float(loss), gradient_norm=float(norm)))
                progress(optimizer_steps=status["optimizer_steps"]+1)
        return logs

    teacher_traces = {}

    def run_life(seed, mapping, method, replay=None):
        mode = "replay" if replay is not None else "closed"
        name = f"{mode}-s{seed}-m{mapping}-{method}"
        dest = output/name
        dest.mkdir(exist_ok=False)
        install(initial)
        optimizer = torch.optim.AdamW(trainable.values(),lr=.0001,weight_decay=0.) if method == "lora" else None
        fast = FastReadout(rows,x,frozen)
        evidence, readout_evidence, health = Evidence(),Evidence(),350.
        trace, logs = [],[]
        probes = [probe(method,evidence,fast,0)]
        adapters = {}
        progress(job=name,turn=0)
        context = model.disable_adapter() if method != "lora" else nullcontext()
        with context:
            for item in schedule(seed):
                if health <= 0:
                    break
                key = tuple(item[k] for k in ("partner","frame","task","wording","order"))
                ids = [lookup[(*key,act)] for act in range(1,4)]
                if method == "hierarchical":
                    p = readout_evidence.moments(item["partner"],item["frame"])[0]
                elif method == "fast":
                    p = sigmoid(fast.predict()[ids])
                elif method == "frozen":
                    p = sigmoid(frozen[ids])
                else:
                    p = sigmoid(observe(ids))
                if replay is None:
                    record = execute_event(body,evidence,health,item,mapping,seed,p)
                else:
                    source = replay[item["turn"]-1]
                    assert source["health_before"] == health
                    assert evidence.export() == source["evidence_before"]
                    evidence.observe(item["partner"],item["frame"],source["action"],source["accepted"])
                    record = dict(source, replay_predicted_probabilities=p.tolist(),
                        replay_candidate_values=body.values(health,p).tolist(), source_action_used=True)
                    assert evidence.export() == source["evidence_after"]
                health = record["health_after"]
                trace.append(record)
                turn = item["turn"]
                record["learner_updated_after_event"] = False
                if turn%5 == 0:
                    readout_evidence.counts[:] = evidence.counts
                    readout_evidence.observations = evidence.observations
                    if method == "lora":
                        logs += optimize(evidence,optimizer,turn,seed)
                        record["learner_updated_after_event"] = True
                    elif method == "fast":
                        fast.update(evidence,(turn//5-1)%2)
                        record["learner_updated_after_event"] = True
                if turn in CONFIG["checkpoint_turns"]:
                    probes.append(probe(method,evidence,fast,turn))
                    if method == "lora":
                        saved = snapshot_adapter()
                        save_file(saved,str(dest/f"adapter-t{turn:03}.safetensors"))
                        adapters[str(turn)] = tensor_hash(saved.items())
                progress(job=name,turn=turn)
            if trace and probes[-1]["turn"] != trace[-1]["turn"]:
                probes.append(probe(method,evidence,fast,trace[-1]["turn"]))
        if method == "lora":
            saved = snapshot_adapter()
            save_file(saved,str(dest/"adapter-final.safetensors"))
            before_reload = observe(list(range(6))).tolist()
            install(load_file(str(dest/"adapter-final.safetensors")))
            assert observe(list(range(6))).tolist() == before_reload
            assert all(p.grad is None for _,p in fixed)
            delta = float(sum((saved[n].float()-initial[n].float()).square().sum() for n in saved).sqrt())
            assert delta > 0
            adapter_info = dict(delta_l2=delta, final_sha256=tensor_hash(saved.items()),
                                checkpoint_sha256=adapters, reload_exact=True)
        else:
            adapter_info = None
        result = dict(seed=seed,mapping=mapping,method=method,mode=mode,trace=trace,probes=probes,
            training_steps=logs,fast_weights={p:w.tolist() for p,w in fast.w.items()} if method == "fast" else None,
            fast_optimization=fast.optimization if method == "fast" else None,adapter=adapter_info,
            final_health=health,turns=len(trace),alive_at_end=bool(health>0),
            decision_maker=CONFIG["decision_maker"],protocol_sha256=digest(CONFIG))
        write(dest/"result.json",result)
        progress(completed_jobs=status["completed_jobs"]+1)
        return trace

    # Run all closed loops before replay. The baseline trace is reused without
    # changing its body, actions, observations or exposure count in replay arms.
    for seed in CONFIG["seeds"]:
        for mapping in CONFIG["mappings"]:
            for method in CONFIG["methods"]:
                trace = run_life(seed,mapping,method)
                if method == "hierarchical":
                    teacher_traces[seed,mapping] = trace
    for seed in CONFIG["replay_seeds"]:
        for mapping in CONFIG["mappings"]:
            for method in ("fast","lora"):
                run_life(seed,mapping,method,replay=teacher_traces[seed,mapping])
    install(initial)
    with model.disable_adapter():
        assert np.array_equal(observe(list(range(6))),frozen[:6])
    assert {n:p._version for n,p in fixed} == fixed_versions
    assert tensor_hash(fixed) == base_hash
    progress(stage="complete",base_unchanged=True,initial_adapter_recovered=True,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_reserved_bytes=torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    run(args.root)
