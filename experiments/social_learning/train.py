"""Run the frozen, aligned and shuffled LoRA pilot on one explicitly supplied GPU.

No resource allocation, credential discovery, network APIs or account selection
are performed here. Model weights are downloaded at an immutable revision.
"""

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import time

from protocol import ACTIONS, PARTNERS, TEST_TASKS, SocialWorld, canonical, digest, messages


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical(value))
    tmp.replace(path)


def file_hash(path):
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(items):
    import torch
    h = sha256()
    for name, tensor in sorted(items):
        h.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        # Byte view supports bf16 without converting or rounding the values.
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(input_path, expected_path, output):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    data, expected = read(input_path), read(expected_path)
    cfg = data["training"]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert data["model"] == expected["id"] and data["revision"] == expected["revision"]
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    status = dict(stage="download", completed_arms=0, total_arms=8, optimizer_steps=0,
                  protocol_sha256=file_hash(input_path), source_sha256=file_hash(Path(__file__)),
                  gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  versions={n: metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "huggingface-hub", "flash-linear-attention", "fla-core")})
    write(output / "status.json", status)
    snapshot = Path(snapshot_download(data["model"], revision=data["revision"], token=False,
                                     allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
    shards = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in snapshot.glob("*.safetensors")}
    if shards != expected["files"]:
        raise RuntimeError("Checkpoint bytes differ from the frozen baseline")
    status.update(stage="load", weights=shards)
    write(output / "status.json", status)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, token=False)
    base, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, output_loading_info=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Checkpoint loading was not exact")
    base.requires_grad_(False)
    base.eval()
    torch.manual_seed(data["seeds"][0] + cfg["adapter_seed_offset"])
    model = get_peft_model(base, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.,
                                           bias="none", target_modules=cfg["target_modules"]))
    model.peft_config["default"].base_model_name_or_path = data["model"]
    model.peft_config["default"].revision = data["revision"]
    model.eval()  # Gradients remain enabled for training; all dropout is disabled.
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 8 and all("lora_" in n for n in trainable)
    assert sum(p.numel() for p in trainable.values()) == 188416
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_versions = {n: p._version for n, p in frozen}
    base_hash = tensor_hash(frozen)
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    initial_hash = tensor_hash(initial.items())
    save_file(initial, str(output / "initial-adapter.safetensors"))
    suffixes = [tokenizer.encode(a, add_special_tokens=False) + [tokenizer.eos_token_id] for a in ACTIONS]
    assert tokenizer.eos_token_id == 248046
    status.update(stage="tokenize", trainable_parameters=sum(p.numel() for p in trainable.values()),
                  trainable_names=list(trainable), initial_adapter_sha256=initial_hash,
                  frozen_parameter_sha256=base_hash, action_suffix_ids=dict(zip(ACTIONS, suffixes)))
    write(output / "status.json", status)

    def encode(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(text, add_special_tokens=False)
        assert len(prefix) < 640
        for action, suffix in zip(ACTIONS, suffixes):
            if tokenizer.encode(text + action, add_special_tokens=False) != prefix + suffix[:-1]:
                raise RuntimeError("Candidate boundaries retokenize the prompt")
        return {"prefix": prefix, "prompt_sha256": sha256(text.encode()).hexdigest(), "id": row["id"]}

    tests = [encode(r) for r in data["tests"]]

    def scores(rows):
        sequences, meta = [], []
        for row in rows:
            for suffix in suffixes:
                prefix = row["prefix"]
                sequences.append(prefix + suffix[:-1])
                meta.append((len(prefix), suffix))
        size = max(map(len, sequences))
        ids = torch.full((len(sequences), size), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(sequences):
            ids[i, :len(tokens)] = torch.tensor(tokens, device="cuda")
            mask[i, :len(tokens)] = 1
        keep = size - min(length for length, _ in meta) + 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits.float()
        log_probs = logits.log_softmax(-1)
        result = []
        for i, (prefix_length, suffix) in enumerate(meta):
            positions = torch.arange(len(suffix), device="cuda") + prefix_length - 1 - (size - keep)
            result.append(log_probs[i, positions, torch.tensor(suffix, device="cuda")].sum())
        return torch.stack(result).view(len(rows), len(ACTIONS))

    def evaluate(name, encoded=tests):
        dest = output / name
        dest.mkdir(exist_ok=False)
        records = []
        with torch.no_grad():
            for row in encoded:
                result = scores([row])[0]
                probs = result.softmax(-1).cpu().tolist()
                record = {"id": row["id"], "prompt_sha256": row["prompt_sha256"],
                          "input_token_ids": row["prefix"], "action_log_likelihood": dict(zip(ACTIONS, result.cpu().tolist())),
                          "probabilities": dict(zip(ACTIONS, probs)), "action": ACTIONS[int(result.argmax().item())]}
                write(dest / (row["id"] + ".json"), record)
                records.append(record)
        return records

    def rollout(life, name):
        world = SocialWorld(life["seed"], life["mapping"])
        for old in life["trace"]:
            replay = world.step(old["encounter_id"], old["partner"], old["action"], tuple(old["task"]))
            if replay != old:
                raise RuntimeError("Learning checkpoint cannot be replayed exactly")
        assert world.world.state.damage == life["checkpoint_body"]["damage"]
        records = []
        with torch.no_grad():
            while not world.world.done:
                t = world.world.state.age
                partner, task = PARTNERS[t % 2], TEST_TASKS[t % 4]
                obs = world.observation(partner, task[0])
                state = dict(damage=obs["damage"], load=obs["load"], weight=18.,
                             turns=min(4, obs["horizon"] - t), protection_left=obs["protection_left"])
                encoded = encode(dict(id=world.encounter_id(), messages=messages(state, partner, task[0], (t // 2) % 2)))
                logits = scores([encoded])[0]
                action = ACTIONS[int(logits.argmax().item())]
                record = world.step(world.encounter_id(), partner, action, task)
                record.update(prompt_sha256=encoded["prompt_sha256"], input_token_ids=encoded["prefix"],
                              probabilities=dict(zip(ACTIONS, logits.softmax(-1).cpu().tolist())),
                              action_log_likelihood=dict(zip(ACTIONS, logits.cpu().tolist())))
                records.append(record)
                write(output / (name + "-rollout.json"), dict(life_sha256=digest(life), trace=records))
        return records

    with model.disable_adapter():
        frozen_tests = evaluate("frozen")
        for life in data["lives"]:
            rollout(life, life["id"] + "-frozen")
    zero_tests = evaluate("zero-adapter", tests[:2])
    if any(a["action_log_likelihood"] != b["action_log_likelihood"] for a, b in zip(frozen_tests, zero_tests)):
        raise RuntimeError("Zero-initialized adapter does not reproduce the frozen policy")
    status.update(stage="train", zero_adapter_exact=True)
    write(output / "status.json", status)

    for life in data["lives"]:
        for arm in ("aligned", "shuffled"):
            run_id = life["id"] + "-" + arm
            dest = output / run_id
            dest.mkdir(exist_ok=False)
            with torch.no_grad():
                for n, p in trainable.items():
                    p.copy_(initial[n].to(device=p.device, dtype=p.dtype))
            assert tensor_hash(trainable.items()) == initial_hash
            rows = data["datasets"][life["id"]][arm]["rows"]
            encoded = [encode(r) for r in rows]
            optimizer = torch.optim.AdamW(trainable.values(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            rng = np.random.default_rng(life["seed"] + cfg["shuffle_seed_offset"])
            metrics = []
            for epoch in range(cfg["epochs"]):
                permutation = rng.permutation(len(rows))
                for offset in range(0, len(rows), cfg["batch_size"]):
                    selected = [int(i) for i in permutation[offset:offset + cfg["batch_size"]]]
                    optimizer.zero_grad(set_to_none=True)
                    logits = scores([encoded[i] for i in selected])
                    target = torch.tensor([rows[i]["target"] for i in selected], device="cuda", dtype=torch.float32)
                    loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    record = dict(epoch=epoch, row_ids=[rows[i]["id"] for i in selected],
                                  loss=float(loss.item()), gradient_norm=float(norm.item()))
                    metrics.append(record)
                    with (dest / "steps.jsonl").open("ab") as handle:
                        handle.write(canonical(record))
                    status.update(run_id=run_id, optimizer_steps=status["optimizer_steps"] + 1,
                                  arm_steps=len(metrics), elapsed_seconds=time.monotonic() - started)
                    write(output / "status.json", status)
            after = {n: p.detach().cpu().clone() for n, p in trainable.items()}
            delta = float(sum((after[n].float() - initial[n].float()).square().sum().item() for n in after) ** .5)
            assert delta > 0 and all(p.grad is None for _, p in frozen)
            save_file(after, str(dest / "adapter.safetensors"))
            model.save_pretrained(str(dest / "peft"), safe_serialization=True, save_embedding_layers=False)
            result = evaluate(run_id + "-tests")
            rollout(life, run_id)
            # Save/reload identity is checked without keeping old tensor references.
            saved = load_file(str(dest / "peft/adapter_model.safetensors"))
            with torch.no_grad():
                for n, p in trainable.items():
                    p.zero_()
            set_peft_model_state_dict(model, saved)
            assert tensor_hash(trainable.items()) == tensor_hash(after.items())
            with torch.no_grad():
                # Batch size can cause harmless BF16 kernel differences, so check
                # exact same single-row forwards as the original evaluation.
                reloaded = [scores([r])[0].detach().cpu().tolist() for r in tests[:2]]
            if reloaded != [list(r["action_log_likelihood"].values()) for r in result[:2]]:
                raise RuntimeError("Saved adapter did not recover evaluation scores")
            write(dest / "training.json", dict(life_sha256=data["datasets"][life["id"]]["life_sha256"],
                  training_rows_sha256=digest(rows), initial_adapter_sha256=initial_hash,
                  final_adapter_sha256=tensor_hash(after.items()), delta_l2=delta,
                  steps=len(metrics), first_loss=metrics[0]["loss"], last_loss=metrics[-1]["loss"],
                  reload_exact=True, frozen_gradients_absent=True))
            status.update(completed_arms=status["completed_arms"] + 1)
            write(output / "status.json", status)
            del optimizer, after, saved
    with model.disable_adapter():
        frozen_end = evaluate("frozen-end", tests[:2])
    assert all(a["action_log_likelihood"] == b["action_log_likelihood"] for a, b in zip(frozen_tests, frozen_end))
    assert all(p._version == frozen_versions[n] for n, p in frozen)
    final_base_hash = tensor_hash(frozen)
    assert final_base_hash == base_hash
    status.update(stage="complete", base_unchanged=True, frozen_end_exact=True,
                  final_frozen_parameter_sha256=final_base_hash,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  elapsed_seconds=time.monotonic() - started)
    write(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="actually download weights and use the supplied GPU")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; preparing data does not start GPU work")
    run(args.input, args.expected_weights, args.output)
