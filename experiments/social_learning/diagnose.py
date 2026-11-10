"""Read-only CUDA evaluation of existing adapters; no optimizer or parameter learning."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import time
import zipfile

from protocol import ACTIONS
from train import file_hash, read, tensor_hash, write


def run(public_path, source_archive, output):
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load as load_tensors
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    public = read(public_path)
    assert file_hash(source_archive) == public["source_archive_sha256"]
    archive = zipfile.ZipFile(source_archive)
    expected = json.loads(archive.read("expected-weights.json"))
    original_status = json.loads(archive.read("output/status.json"))
    assert expected["id"] == public["model"] and expected["revision"] == public["revision"]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(stage="download", completed_jobs=0, total_jobs=len(public["jobs"]),
        completed_forwards=0, total_forwards=sum(len(j["stimulus_ids"]) + len(j["batch_check_ids"]) for j in public["jobs"]),
        optimizer_steps=0, trainable_parameters=0, public_sha256=file_hash(public_path),
        diagnostic_source_sha256=file_hash(Path(__file__)), source_archive_sha256=file_hash(source_archive),
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        versions={n: metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "huggingface-hub", "flash-linear-attention", "fla-core")})
    for name, version in state["versions"].items():
        assert version == original_status["versions"][name], (name, version, original_status["versions"][name])
    write(output / "status.json", state)
    snapshot = Path(snapshot_download(public["model"], revision=public["revision"], token=False,
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
    shards = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in snapshot.glob("*.safetensors")}
    assert shards == expected["files"]
    state.update(stage="load", weights=shards)
    write(output / "status.json", state)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, token=False, local_files_only=True, trust_remote_code=False)
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, output_loading_info=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    base.requires_grad_(False)
    cfg = public["training_config"]
    model = get_peft_model(base, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"],
        lora_dropout=0., bias="none", target_modules=cfg["target_modules"]))
    model.eval()
    model.requires_grad_(False)
    adapters = {n: p for n, p in model.named_parameters() if "lora_" in n}
    fixed = [(n, p) for n, p in model.named_parameters() if "lora_" not in n]
    assert len(adapters) == 8 and sum(p.numel() for p in adapters.values()) == 188416
    assert all(p.dtype == torch.float32 for p in adapters.values())
    fixed_versions = {n: p._version for n, p in fixed}
    fixed_hash = tensor_hash(fixed)
    assert fixed_hash == original_status["frozen_parameter_sha256"]
    suffixes = [tokenizer.encode(action, add_special_tokens=False) + [tokenizer.eos_token_id] for action in ACTIONS]
    assert dict(zip(ACTIONS, suffixes)) == original_status["action_suffix_ids"]
    encoded = {}
    for sid, stimulus in public["stimuli"].items():
        text = tokenizer.apply_chat_template(stimulus["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) < 640
        for action, suffix in zip(ACTIONS, suffixes):
            assert tokenizer.encode(text + action, add_special_tokens=False) == ids + suffix[:-1]
        encoded[sid] = dict(id=sid, prefix=ids, prompt_sha256=sha256(text.encode()).hexdigest())

    def forward(rows):
        sequences, positions = [], []
        for row in rows:
            for suffix in suffixes:
                sequences.append(row["prefix"] + suffix[:-1])
                positions.append((len(row["prefix"]), suffix))
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
            mask[i, :len(seq)] = 1
        keep = length - min(p for p, _ in positions) + 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits.float()
        lp = logits.log_softmax(-1)
        results = []
        for j, row in enumerate(rows):
            full_scores, word_scores, details = [], [], {}
            prefix_position = len(row["prefix"]) - 1 - (length - keep)
            common = logits[j * 4:j * 4 + 4, prefix_position]
            prefix_difference = float((common - common[0]).abs().max().item())
            for k, action in enumerate(ACTIONS):
                i = j * 4 + k
                prefix, suffix = positions[i]
                wanted = torch.arange(len(suffix), device="cuda") + prefix - 1 - (length - keep)
                components = lp[i, wanted, torch.tensor(suffix, device="cuda")]
                full_scores.append(components.sum())
                word_scores.append(components[:-1].sum())
                details[action] = dict(token_ids=suffix, token_log_probabilities=components.cpu().tolist(),
                    full_log_likelihood=float(full_scores[-1].item()), word_log_likelihood=float(word_scores[-1].item()),
                    eos_log_probability=float(components[-1].item()))
            full, word = torch.stack(full_scores), torch.stack(word_scores)
            results.append(dict(id=row["id"], prompt_sha256=row["prompt_sha256"], input_token_ids=row["prefix"],
                batch_size=len(rows), actions=details, full_probabilities=dict(zip(ACTIONS, full.softmax(-1).cpu().tolist())),
                word_only_probabilities=dict(zip(ACTIONS, word.softmax(-1).cpu().tolist())),
                action=ACTIONS[int(full.argmax().item())], word_only_action=ACTIONS[int(word.argmax().item())],
                candidate_prefix_max_logit_difference=prefix_difference))
        return results

    state.update(stage="evaluate", frozen_parameter_sha256=fixed_hash, action_suffix_ids=dict(zip(ACTIONS, suffixes)))
    write(output / "status.json", state)
    for job in public["jobs"]:
        weights = load_tensors(archive.read(job["adapter_member"] or "output/initial-adapter.safetensors"))
        assert set(weights) == set(adapters)
        if job["adapter_member"]:
            assert tensor_hash(weights.items()) == job["adapter_tensor_sha256"]
        with torch.no_grad():
            for n, p in adapters.items():
                p.copy_(weights[n].to(device=p.device, dtype=p.dtype))
        model.requires_grad_(False)
        before_hash = tensor_hash(adapters.items())
        if job["adapter_member"]:
            assert before_hash == job["adapter_tensor_sha256"]
        versions = {n: p._version for n, p in model.named_parameters()}
        folders = [("single", job["stimulus_ids"], public["inference_batch_size"]),
                   ("batch", job["batch_check_ids"], public["batch_check_size"])]
        max_replay_difference, replay_exact = 0., 0
        context = model.disable_adapter() if job["id"] == "frozen" else nullcontext()
        with context, torch.inference_mode():
            for mode, identifiers, batch_size in folders:
                dest = output / job["id"] / mode
                dest.mkdir(parents=True, exist_ok=False)
                for offset in range(0, len(identifiers), batch_size):
                    selected = identifiers[offset:offset + batch_size]
                    records = forward([encoded[sid] for sid in selected])
                    for record in records:
                        sid = record["id"]
                        if mode == "single" and sid in job["original_references"]:
                            old = json.loads(archive.read(job["original_references"][sid]))
                            assert record["input_token_ids"] == old["input_token_ids"] and record["prompt_sha256"] == old["prompt_sha256"]
                            delta = max(abs(record["actions"][a]["full_log_likelihood"] - old["action_log_likelihood"][a]) for a in ACTIONS)
                            max_replay_difference = max(max_replay_difference, delta)
                            replay_exact += delta == 0.
                        write(dest / (sid + ".json"), record)
                    state.update(job=job["id"], mode=mode, completed_forwards=state["completed_forwards"] + len(records),
                        elapsed_seconds=time.monotonic() - started)
                    write(output / "status.json", state)
        model.requires_grad_(False)
        assert all(not p.requires_grad and p.grad is None and p._version == versions[n] for n, p in model.named_parameters())
        assert tensor_hash(adapters.items()) == before_hash
        write(output / job["id"] / "checks.json", dict(adapter_unchanged=True, adapter_tensor_sha256=before_hash,
            parameter_versions_unchanged=True, trainable_parameters=0, optimizer_steps=0,
            original_replay_cases=len(job["original_references"]), original_replay_exact=replay_exact,
            original_replay_max_full_score_delta=max_replay_difference))
        state.update(completed_jobs=state["completed_jobs"] + 1)
        write(output / "status.json", state)
        del weights
    assert all(p._version == fixed_versions[n] for n, p in fixed)
    final_hash = tensor_hash(fixed)
    assert final_hash == fixed_hash
    assert state["completed_forwards"] == state["total_forwards"]
    state.update(stage="complete", all_weights_unchanged=True, final_frozen_parameter_sha256=final_hash,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic() - started)
    write(output / "status.json", state)
    archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required for actual GPU inference")
    run(args.public, args.source_archive, args.output)
