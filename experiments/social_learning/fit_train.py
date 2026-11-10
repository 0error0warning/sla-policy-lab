"""Run the pre-frozen crossed fitting assay on one explicitly supplied GPU.

No allocation, credentials, account selection, rollouts, or adaptive search here.
"""

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import time
import zipfile

from protocol import canonical
from train import file_hash, read, tensor_hash, write


def run(protocol_path, source_archive, output):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load as load_tensors, load_file, save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    data = read(protocol_path)
    assert data["schema"] == "crossed-mini-fit-v1"
    assert file_hash(source_archive) == data["source_archive_sha256"]
    archive = zipfile.ZipFile(source_archive)
    expected = json.loads(archive.read("expected-weights.json"))
    previous = json.loads(archive.read("output/status.json"))
    assert expected["id"] == data["model"] and expected["revision"] == data["revision"]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(stage="download", completed_jobs=0, total_jobs=len(data["jobs"]), optimizer_steps=0,
        total_optimizer_steps=len(data["jobs"]) * data["max_steps"], protocol_sha256=file_hash(protocol_path),
        worker_sha256=file_hash(Path(__file__)), source_archive_sha256=file_hash(source_archive),
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        versions={n: metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "huggingface-hub", "flash-linear-attention", "fla-core")})
    for name, version in status["versions"].items():
        assert version == previous["versions"][name], (name, version, previous["versions"][name])
    write(output / "status.json", status)
    snapshot = Path(snapshot_download(data["model"], revision=data["revision"], token=False,
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
    shards = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in snapshot.glob("*.safetensors")}
    assert shards == expected["files"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, output_loading_info=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    base.requires_grad_(False)
    cfg = data["training_config"]
    model = get_peft_model(base, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.,
                                           bias="none", target_modules=cfg["target_modules"]))
    model.eval()  # eval controls dropout, not gradient computation
    model.peft_config["default"].base_model_name_or_path = data["model"]
    model.peft_config["default"].revision = data["revision"]
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fixed = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    assert len(trainable) == 8 and sum(p.numel() for p in trainable.values()) == 188416
    assert all("lora_" in n and p.dtype == torch.float32 for n, p in trainable.items())
    fixed_versions = {n: p._version for n, p in fixed}
    base_hash = tensor_hash(fixed)
    assert base_hash == data["expected_base_tensor_sha256"]
    initial = load_tensors(archive.read("output/initial-adapter.safetensors"))
    assert set(initial) == set(trainable) and tensor_hash(initial.items()) == data["initial_adapter_tensor_sha256"]

    def reset():
        for p in trainable.values():
            p.grad = None
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(initial[n].to(device=p.device, dtype=p.dtype))
        assert tensor_hash(trainable.items()) == data["initial_adapter_tensor_sha256"]

    def encode(row, actions):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(text, add_special_tokens=False)
        assert len(prefix) < 800
        suffixes = [tokenizer.encode(a, add_special_tokens=False) + [tokenizer.eos_token_id] for a in actions]
        assert tokenizer.eos_token_id == 248046
        for a, suffix in zip(actions, suffixes):
            assert tokenizer.encode(text + a, add_special_tokens=False) == prefix + suffix[:-1]
        return dict(id=row["id"], prefix=prefix, actions=actions, suffixes=suffixes,
                    prompt_sha256=sha256(text.encode()).hexdigest())

    def forward(rows, details=False):
        actions = rows[0]["actions"]
        assert all(r["actions"] == actions for r in rows)
        sequences, meta = [], []
        for row in rows:
            for suffix in row["suffixes"]:
                sequences.append(row["prefix"] + suffix[:-1])
                meta.append((len(row["prefix"]), suffix))
        size = max(map(len, sequences))
        ids = torch.full((len(sequences), size), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
            mask[i, :len(seq)] = 1
        keep = size - min(n for n, _ in meta) + 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits.float()
        log_probs = logits.log_softmax(-1)
        full, words, tokens = [], [], []
        for i, (n, suffix) in enumerate(meta):
            pos = torch.arange(len(suffix), device="cuda") + n - 1 - (size - keep)
            parts = log_probs[i, pos, torch.tensor(suffix, device="cuda")]
            full.append(parts.sum())
            if details:
                words.append(parts[:-1].sum())
                tokens.append(parts.detach().cpu().tolist())
        scores = torch.stack(full).view(len(rows), len(actions))
        if not details:
            return scores
        records = []
        for i, row in enumerate(rows):
            records.append(dict(id=row["id"], prompt_sha256=row["prompt_sha256"], input_token_ids=row["prefix"],
                batch_size=len(rows), action_log_likelihood=dict(zip(actions, scores[i].cpu().tolist())),
                word_log_likelihood=dict(zip(actions, [float(w.item()) for w in words[i*len(actions):(i+1)*len(actions)]])),
                probabilities=dict(zip(actions, scores[i].softmax(-1).cpu().tolist())),
                suffix_ids=dict(zip(actions, row["suffixes"])),
                token_log_probabilities=dict(zip(actions, tokens[i*len(actions):(i+1)*len(actions)])),
                action=actions[int(scores[i].argmax().item())]))
        return scores, records

    encoded = {did: {group: [encode(r, dataset["actions"]) for r in dataset[group]]
                    for group in ("train", "new_question", "provided_returns", "calculate")}
               for did, dataset in data["datasets"].items()}
    reset()
    first = encoded[data["jobs"][0]["dataset"]]["train"][:1]
    with torch.no_grad(), model.disable_adapter():
        frozen_first = forward(first).cpu().tolist()
    with torch.no_grad():
        assert forward(first).cpu().tolist() == frozen_first
    status.update(stage="train", zero_adapter_exact=True, frozen_parameter_sha256=base_hash,
        trainable_parameters=sum(p.numel() for p in trainable.values()), trainable_names=list(trainable), weights=shards,
        initial_adapter_sha256=tensor_hash(initial.items()))
    write(output / "status.json", status)

    def evaluate(dest, did, step):
        dest.mkdir(parents=True, exist_ok=False)
        for group in ("train", "new_question", "provided_returns"):
            if not encoded[did][group]:
                continue
            records = []
            with torch.no_grad():
                for row in encoded[did][group]:
                    records.extend(forward([row], details=True)[1])
            write(dest / (group + ".json"), records)
        if step == data["max_steps"]:
            records = []
            with torch.no_grad():
                rows = encoded[did]["train"]
                for offset in range(0, len(rows), 2):
                    records.extend(forward(rows[offset:offset+2], details=True)[1])
            write(dest / "train-batch2.json", records)
        if step in (0, data["max_steps"]) and encoded[did]["calculate"]:
            records = []
            with torch.no_grad():
                for row in encoded[did]["calculate"]:
                    ids = torch.tensor([row["prefix"]], device="cuda")
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=192, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                    new = generated[0, ids.shape[1]:].cpu().tolist()
                    records.append(dict(id=row["id"], input_token_ids=row["prefix"], prompt_sha256=row["prompt_sha256"],
                        output_token_ids=new, text=tokenizer.decode(new, skip_special_tokens=True),
                        max_new_tokens=192, terminated=bool(new and new[-1] == tokenizer.eos_token_id)))
            write(dest / "calculate.json", records)

    for job in data["jobs"]:
        reset()
        did = job["dataset"]
        rows = data["datasets"][did]["train"]
        dest = output / job["id"]
        dest.mkdir(exist_ok=False)
        evaluate(dest / "step-000", did, 0)
        optimizer = torch.optim.AdamW(trainable.values(), lr=job["learning_rate"], weight_decay=cfg["weight_decay"])
        rng = np.random.default_rng(data["permutation_seed"])
        permutation, cursor, history = [], 0, []
        for step in range(1, data["max_steps"] + 1):
            if cursor >= len(permutation):
                permutation, cursor = rng.permutation(len(rows)).tolist(), 0
            selected = permutation[cursor:cursor + data["training_batch_size"]]
            cursor += data["training_batch_size"]
            optimizer.zero_grad(set_to_none=True)
            scores = forward([encoded[did]["train"][i] for i in selected])
            q = torch.tensor([rows[i]["target"] for i in selected], device="cuda", dtype=torch.float32)
            loss = -(q * scores.log_softmax(-1)).sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg["gradient_clip"], error_if_nonfinite=True)
            assert all(p.grad is None for _, p in fixed)
            optimizer.step()
            record = dict(step=step, row_ids=[rows[i]["id"] for i in selected], loss=float(loss.item()),
                          pre_clip_gradient_norm=float(norm.item()))
            history.append(record)
            with (dest / "steps.jsonl").open("ab") as handle:
                handle.write(canonical(record))
            status.update(stage="train", job=job["id"], job_step=step,
                optimizer_steps=status["optimizer_steps"] + 1, elapsed_seconds=time.monotonic()-started)
            write(output / "status.json", status)
            if step in data["checkpoints"]:
                cp = dest / f"step-{step:03}"
                evaluate(cp, did, step)
                after = {n: p.detach().cpu().clone() for n, p in trainable.items()}
                save_file(after, str(cp / "adapter.safetensors"))
                delta = float(sum((after[n].float()-initial[n].float()).square().sum().item() for n in after)**.5)
                write(cp / "parameters.json", dict(adapter_tensor_sha256=tensor_hash(after.items()),
                    initial_adapter_sha256=data["initial_adapter_tensor_sha256"], delta_l2=delta, step=step,
                    frozen_gradients_absent=True, frozen_versions_unchanged=all(p._version == fixed_versions[n] for n,p in fixed)))
                assert delta > 0 and all(p._version == fixed_versions[n] for n,p in fixed)
                del after
        model.save_pretrained(str(dest / "peft"), safe_serialization=True, save_embedding_layers=False)
        final_hash = tensor_hash(trainable.items())
        saved = load_file(str(dest / "peft/adapter_model.safetensors"))
        with torch.no_grad():
            for p in trainable.values():
                p.zero_()
        set_peft_model_state_dict(model, saved)
        assert tensor_hash(trainable.items()) == final_hash
        with torch.no_grad():
            reloaded = [forward([r]).cpu().tolist()[0] for r in encoded[did]["train"][:2]]
        original = read(dest / f"step-{data['max_steps']:03}" / "train.json")[:2]
        assert reloaded == [[r["action_log_likelihood"][a] for a in data["datasets"][did]["actions"]] for r in original]
        write(dest / "checks.json", dict(reload_exact=True, final_adapter_tensor_sha256=final_hash,
            initial_adapter_sha256=data["initial_adapter_tensor_sha256"], optimizer_steps=len(history),
            gradient_clipped_steps=sum(r["pre_clip_gradient_norm"] > cfg["gradient_clip"] for r in history),
            frozen_gradients_absent=True, frozen_versions_unchanged=True))
        status.update(completed_jobs=status["completed_jobs"]+1)
        write(output / "status.json", status)
        del optimizer, saved
    for p in trainable.values():
        p.grad = None
    with torch.no_grad(), model.disable_adapter():
        assert forward(first).cpu().tolist() == frozen_first
    assert all(p._version == fixed_versions[n] for n,p in fixed)
    assert tensor_hash(fixed) == base_hash
    status.update(stage="complete", base_unchanged=True, frozen_end_exact=True,
        final_frozen_parameter_sha256=base_hash, elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run required; preparation never allocates a GPU")
    run(args.protocol, args.source_archive, args.output)
