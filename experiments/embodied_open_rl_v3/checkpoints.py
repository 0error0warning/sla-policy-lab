"""Measure effective LoRA weight change on the CPU from archived adapter tensors."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .protocol import CONFIG, STAGES, canonical, write


def load_adapter(path, expected):
    data = path.read_bytes()
    assert len(data) == expected["file_bytes"]
    assert sha256(data).hexdigest() == expected["file_sha256"]
    header_end = 8 + int.from_bytes(data[:8], "little")
    header = json.loads(data[8:header_end])
    payload = memoryview(data)[header_end:]
    tensor_hash = sha256()
    tensors = {}
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32"
        start, end = info["data_offsets"]
        raw = payload[start:end]
        tensor = np.frombuffer(raw, dtype="<f4").reshape(info["shape"])
        assert np.isfinite(tensor).all()
        tensor_hash.update(canonical([name, "torch.float32", info["shape"]]))
        tensor_hash.update(raw)
        tensors[name] = tensor.astype(np.float64)
    assert tensor_hash.hexdigest() == expected["tensor_sha256"]
    pairs = {
        name: (value, tensors[name.replace(".lora_A.", ".lora_B.")])
        for name, value in tensors.items()
        if ".lora_A." in name
    }
    assert len(pairs) == 16 and len(tensors) == 2 * len(pairs)
    return pairs


def effective_inner_product(left, right):
    assert left.keys() == right.keys()
    value = 0.0
    for name, (left_a, left_b) in left.items():
        right_a, right_b = right[name]
        value += float(np.sum((left_b.T @ right_b) * (left_a @ right_a.T)))
    return value * (CONFIG["lora_alpha"] / CONFIG["lora_rank"]) ** 2


def effective_distance(left, right):
    squared = (
        effective_inner_product(left, left)
        + effective_inner_product(right, right)
        - 2.0 * effective_inner_product(left, right)
    )
    return float(np.sqrt(max(0.0, squared)))


def compare_checkpoints(results_zip, directory, life_ids, out):
    with ZipFile(results_zip) as archive:
        manifest = json.loads(archive.read("output/checkpoints-manifest.json"))
    birth_name = "initial-adapter.safetensors"
    birth = load_adapter(directory / birth_name, manifest[birth_name])
    results = {}
    for life in life_ids:
        snapshots = {}
        for stage in STAGES:
            name = f"{life}/adapter-{stage}.safetensors"
            snapshots[stage] = load_adapter(directory / name, manifest[name])
        results[life] = {
            "effective_l2_from_birth": {
                stage: effective_distance(birth, snapshots[stage]) for stage in STAGES
            },
            "effective_l2_between_stages": {
                f"{before}_to_{after}": effective_distance(snapshots[before], snapshots[after])
                for before, after in zip(STAGES, STAGES[1:])
            },
        }
    write(out, results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--life", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare_checkpoints(args.results_zip, args.directory, args.life, args.out)
