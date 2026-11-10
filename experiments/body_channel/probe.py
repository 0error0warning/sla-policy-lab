"""Run the fixed, untrained residual-patching probe on a CUDA GPU."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import time


def align(text, offsets, start_marker, end_marker):
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("The prompt must contain exactly one body field")
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    positions = [index for index, (a, b) in enumerate(offsets) if a < end and b > start]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("The body-field token span is not contiguous")
    if offsets[positions[0]][0] != start or offsets[positions[-1]][1] != end:
        raise ValueError("The body-field token span crosses a text boundary")
    return positions[0], positions[-1] + 1


def check_alignment(encoded):
    spans = {tuple(value[1]) for value in encoded.values()}
    lengths = {len(value[0]) for value in encoded.values()}
    if len(spans) != 1 or len(lengths) != 1:
        raise ValueError("Body variants do not align to the same token positions")
    start, end = spans.pop()
    masked = encoded["masked"][0]
    for tokens, _ in encoded.values():
        if tokens[:start] != masked[:start] or tokens[end:] != masked[end:]:
            raise ValueError("Token differences outside the body slot")
    if encoded["low"][0][start:end] == encoded["high"][0][start:end]:
        raise ValueError("The donor states tokenize identically")
    return start, end


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha(path):
    result = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(public_path, expected_weights_path, output):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    public = read(public_path)
    expected_weights = read(expected_weights_path)
    assert expected_weights["id"] == public["model"] and expected_weights["revision"] == public["revision"]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    state = {"stage": "download", "probes_done": 0, "probes_total": len(public["probes"]),
             "model": public["model"], "revision": public["revision"], "prompts_sha256": file_sha(public_path),
             "probe_source_sha256": file_sha(Path(__file__)), "gpu": torch.cuda.get_device_name(),
             "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
             "versions": {name: metadata.version(name) for name in ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors")}}
    write(output / "status.json", state)
    snapshot = Path(snapshot_download(public["model"], revision=public["revision"], token=False,
                                      allow_patterns=["*.safetensors", "*.json", "*.jinja", "vocab.*", "merges.*"]))
    shards = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha(path)} for path in snapshot.glob("*.safetensors")}
    if shards != expected_weights["files"]:
        raise RuntimeError("Loaded weight files differ from the frozen diagnostic checkpoint")
    state.update(stage="load", weights=shards)
    write(output / "status.json", state)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, token=False, trust_remote_code=False, local_files_only=True)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, output_loading_info=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise RuntimeError(f"Checkpoint loading reported {key}")
    model.requires_grad_(False)
    model.eval()
    parameter_versions = {name: p._version for name, p in model.named_parameters()}
    layers = model.model.language_model.layers
    assert len(layers) == 32
    layer_indices = tuple(public["layers"])
    labels = ["A", "B", "C"]
    label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    assert all(len(tokens) == 1 for tokens in label_tokens)
    label_ids = [tokens[0] for tokens in label_tokens]

    def forward(tokens, span, capture=(), injection=None):
        collected = {}
        start, end = span

        def hook(layer):
            def apply(module, args, value):
                residual = value[0] if isinstance(value, tuple) else value
                assert residual.ndim == 3 and residual.shape[:2] == tokens.shape
                assert layer not in collected
                slot = residual[:, start:end, :]
                collected[layer] = slot.clone()
                if injection is None or injection[0] != layer:
                    return None
                replacement = injection[1]
                assert slot.shape == replacement.shape and slot.dtype == replacement.dtype
                changed = residual.clone()
                changed[:, start:end, :] = replacement
                return (changed, *value[1:]) if isinstance(value, tuple) else changed
            return apply

        with ExitStack() as stack:
            for layer in capture:
                handle = layers[layer].register_forward_hook(hook(layer))
                stack.callback(handle.remove)
            with torch.inference_mode():
                final = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                              use_cache=False, logits_to_keep=1).logits[0, -1, :].float()
        assert set(collected) == set(capture)
        scores = final[label_ids].cpu().tolist()
        result = {"logits": dict(zip(labels, scores)), "label": labels[max(range(3), key=scores.__getitem__)],
                  "unrestricted_top_token_id": int(final.argmax().item())}
        return result, collected

    def perturb(baseline, donor, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(baseline.shape, generator=generator).to(baseline.device)
        scale = torch.linalg.vector_norm(donor.float() - baseline.float())
        replacement = (baseline.float() + noise * (scale / torch.linalg.vector_norm(noise))).to(baseline.dtype)
        realized = torch.linalg.vector_norm(replacement.float() - baseline.float())
        return replacement, float(scale.item()), float(realized.item())

    state.update(stage="forward", dtype=str(next(model.parameters()).dtype), label_token_ids=label_ids,
                 layer_types={layer: model.config.text_config.layer_types[layer] for layer in layer_indices},
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write(output / "status.json", state)
    started = time.monotonic()
    for probe in public["probes"]:
        encoded = {}
        prompt_hashes = {}
        for variant, messages in probe["messages"].items():
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert text.endswith("<think>\n\n</think>\n\n")
            tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            encoded[variant] = (tokens["input_ids"], align(text, tokens["offset_mapping"], public["body_start"], public["body_end"]))
            prompt_hashes[variant] = sha256(text.encode()).hexdigest()
        span = check_alignment(encoded)
        inputs = {variant: torch.tensor([tokens], device="cuda") for variant, (tokens, _) in encoded.items()}
        ordinary, observed = {}, {}
        for variant, tokens in inputs.items():
            ordinary[variant], observed[variant] = forward(tokens, span, layer_indices)
        results = {"id": probe["id"], "span": span, "prompt_sha256": prompt_hashes,
                   "sequence_tokens": inputs["masked"].shape[1], "ordinary": ordinary, "layers": {}}
        for layer in layer_indices:
            control, _ = forward(inputs["masked"], span, (layer,), (layer, observed["masked"][layer]))
            delta = max(abs(control["logits"][label] - ordinary["masked"]["logits"][label]) for label in labels)
            if delta > 1e-6:
                raise RuntimeError(f"Self-patch changed the result for {probe['id']} layer {layer}: {delta}")
            layer_result = {"self_patch_max_logit_delta": delta, "matched": {}, "random": {}, "norms": {}}
            seed = int.from_bytes(sha256(f"{probe['pair']}:{layer}".encode()).digest()[:4], "big")
            for variant in ("low", "high"):
                donor = observed[variant][layer]
                recipient = observed["masked"][layer]
                replacement, expected_norm, actual_norm = perturb(recipient, donor, seed)
                layer_result["matched"][variant], _ = forward(inputs["masked"], span, (layer,), (layer, donor))
                layer_result["random"][variant], _ = forward(inputs["masked"], span, (layer,), (layer, replacement))
                layer_result["norms"][variant] = {"matched_l2": expected_norm, "random_l2_after_bf16": actual_norm}
            results["layers"][layer] = layer_result
        write(output / f"{probe['id']}.json", results)
        state["probes_done"] += 1
        state["last_probe"] = probe["id"]
        write(output / "status.json", state)
    assert parameter_versions == {name: p._version for name, p in model.named_parameters()}
    state.update(stage="completed", parameter_versions_unchanged=True, forward_seconds=time.monotonic() - started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_loading_clean=True)
    write(output / "status.json", state)
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--expected-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.prompts, args.expected_weights, args.output)


if __name__ == "__main__":
    main()
