"""Versioned numeric-only checkpoints. No pickle, code loading, or network calls."""
import json
from pathlib import Path
import numpy as np
from .models import ModelSuite, MLP, LoRA, Steering, digest, HIDDEN, RANK
from .environment import PROFILES


def arrays(suite):
    values = {f"base_{i}": p for i, p in enumerate(suite.base.params)}
    for profile in PROFILES:
        values.update({f"{profile}_small_{i}": p for i, p in enumerate(suite.small[profile].params)})
        values[f"{profile}_lora_a"] = suite.lora[profile].a
        values[f"{profile}_lora_b"] = suite.lora[profile].b
        values[f"{profile}_direction"] = suite.steering[profile].direction
    return values


def fingerprint(suite):
    return digest(*suite.base.params, *[p for m in suite.small.values() for p in m.params],
                  *[p for m in suite.lora.values() for p in (m.a, m.b)],
                  *[m.direction for m in suite.steering.values()])


def save(suite, path):
    metadata = {"format": "sla-toy-npz-v1", "train_seed": suite.seed,
                "fingerprint": fingerprint(suite), "training": suite.report}
    # A file handle prevents NumPy from silently adding another .npz suffix.
    with Path(path).open("wb") as output:
        np.savez_compressed(output, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays(suite))


def load(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Toy checkpoint exceeds 2 MB limit")
    # Check uncompressed size as well; never deserialize Python object arrays.
    import zipfile
    with zipfile.ZipFile(path) as archive:
        if sum(f.file_size for f in archive.infolist()) > 2_000_000:
            raise ValueError("Expanded checkpoint exceeds 2 MB limit")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("format") != "sla-toy-npz-v1":
            raise ValueError("Unsupported checkpoint format")
        seed = metadata.get("train_seed")
        if type(seed) is not int or not 0 <= seed <= 2**31 - 10000:
            raise ValueError("Invalid checkpoint train seed")
        suite = ModelSuite.__new__(ModelSuite)
        suite.seed, suite.report = seed, metadata["training"]
        suite.base = MLP(seed)
        suite.small, suite.lora, suite.steering = {}, {}, {}
        for profile in PROFILES:
            suite.small[profile] = MLP(seed)
            suite.lora[profile] = LoRA(suite.base, seed, RANK)
            steering = Steering.__new__(Steering)
            steering.base = suite.base
            steering.direction = np.zeros(HIDDEN)
            steering.target = "request_protection" if profile == "preserve" else "answer"
            suite.steering[profile] = steering
        expected = arrays(suite)
        if set(data.files) != set(expected) | {"metadata"}:
            raise ValueError("Checkpoint array keys do not match model schema")
        for name, target in expected.items():
            source = data[name]
            if source.shape != target.shape or source.dtype != np.float64 or not np.isfinite(source).all():
                raise ValueError(f"Invalid checkpoint array: {name}")
            target[:] = source
    suite.fingerprint = fingerprint(suite)
    if suite.fingerprint != metadata.get("fingerprint"):
        raise ValueError("Checkpoint fingerprint mismatch")
    return suite
