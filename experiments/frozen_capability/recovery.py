"""Audit segmented recovery without manufacturing completion or freeze receipts."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def package(path):
    with ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}
    lock = json.loads(files.pop("run-lock.json"))
    assert set(files) == set(lock["files"])
    assert all(
        hashlib.sha256(data).hexdigest() == lock["files"][name]
        for name, data in files.items()
    )
    return files


def select_record(previous, incoming):
    """An executed answer is immutable; later budget placeholders cannot replace it."""
    if previous is None:
        return incoming
    before = "output_ids" in previous["record"]
    after = "output_ids" in incoming["record"]
    if before and after:
        raise ValueError("duplicate executed presentation; no best-of-retry selection")
    return previous if before else incoming


def load(manifest, original_inputs):
    manifest = Path(manifest)
    original = package(original_inputs)
    full = json.loads(original["execution.json"])
    specification = json.loads(manifest.read_bytes())
    chosen, receipts = {}, []
    hardware = None
    for segment in specification["segments"]:
        name = segment["name"]
        root = manifest.parent / segment["source"]
        inputs = manifest.parent / segment["inputs"]
        files = package(inputs)
        assert set(files) == set(original)
        assert all(files[n] == original[n] for n in files if n != "execution.json")
        config = json.loads(files["execution.json"])
        assert json.loads((root / "execution.json").read_bytes()) == config
        remote_lock = root.parent / "run-lock.json"
        if remote_lock.exists():
            with ZipFile(inputs) as archive:
                assert json.loads(remote_lock.read_bytes()) == json.loads(
                    archive.read("run-lock.json")
                )
        if not segment["partial"]:
            assert remote_lock.exists()
            receipt_path = root.parent / "launch-receipt.json"
            launch = json.loads(receipt_path.read_bytes())
            assert (
                launch["input_sha256"]
                == hashlib.sha256(inputs.read_bytes()).hexdigest()
            )
        assert set(config) <= set(full) | {"recovery"}
        variable = {"models", "jobs", "per_model_seconds", "recovery"}
        assert all(config[k] == v for k, v in full.items() if k not in variable)
        assert 0 < config["per_model_seconds"] <= full["per_model_seconds"]
        assert set(config["models"]) <= set(full["models"])
        if "recovery" in config:
            assert len(config["models"]) == 1
            m = config["models"][0]
            prior = sorted(
                n
                for (model, n), record in chosen.items()
                if model == m and "output_ids" in record["record"]
            )
            assert config["recovery"]["prior_preserved_jobs"] == prior
            remaining = [j for j in full["jobs"] if j["number"] not in prior]
            if config["recovery"].get("direct_only", False):
                remaining = [j for j in remaining if not j["thinking"]]
            assert config["jobs"] == remaining
        else:
            assert config == full
        current_hw = json.loads((root / "hardware.json").read_bytes())
        if hardware is None:
            hardware = current_hw
        else:
            assert current_hw == hardware, (
                name,
                "different recorded hardware/software",
            )
        completed = (root / "completed.json").exists()
        assert completed or segment["partial"], name
        jobs = {j["number"]: j for j in config["jobs"]}
        receipt = {
            "name": name,
            "source_partial": segment["partial"],
            "completed": completed,
            "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
            "models": {},
        }
        for model in config["models"]:
            directory = root / model
            paths = sorted((directory / "records").glob("*.json"))
            if not paths:
                continue
            assert json.loads(
                (directory / "weights-manifest.json").read_bytes()
            ) == json.loads(original[f"expected-{model}.json"])
            meta = json.loads((directory / "loading.json").read_bytes())
            assert meta["trainable_parameters"] == 0 and meta["loading_empty"]
            freeze_path = directory / "unchanged.json"
            freeze = (
                json.loads(freeze_path.read_bytes()) if freeze_path.exists() else None
            )
            if freeze:
                assert (
                    freeze["parameter_versions_unchanged"]
                    and freeze["gradients_absent"]
                    and freeze["trainable_parameters"] == 0
                )
            if not segment["partial"]:
                assert freeze is not None and len(paths) == len(jobs)
            used = 0
            for path in paths:
                data = path.read_bytes()
                record = json.loads(data)
                n = record["number"]
                assert record["model"] == model and all(
                    record[k] == v for k, v in jobs[n].items()
                )
                assert path.name == f"{n:03d}.json"
                used += "output_ids" in record
                key = (model, n)
                incoming = {
                    "record": record,
                    "meta": meta,
                    "segment": name,
                    "raw_sha256": hashlib.sha256(data).hexdigest(),
                    "raw_path": f"{segment['source']}/{model}/records/{path.name}",
                    "freeze_receipt_available": freeze is not None,
                }
                chosen[key] = select_record(chosen.get(key), incoming)
            receipt["models"][model] = {
                "records": len(paths),
                "executed": used,
                "freeze_receipt": freeze,
            }
        receipts.append(receipt)
    return chosen, receipts
