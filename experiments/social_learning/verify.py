"""Verify published hashes, adapter tensors, scores and body replay without a GPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from protocol import REPO
from score import match, read, score, verify_archive


DATA = REPO / "artifacts/model-studies/2026-11-10/social-learning"


def check_input(path):
    archive = zipfile.ZipFile(path)
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist()))
    lock = json.loads(archive.read("run-lock.json"))
    assert set(archive.namelist()) == set(lock) | {"run-lock.json"}
    for name, expected in lock.items():
        assert sha256(archive.read(name)).hexdigest() == expected
    return archive, lock


def verify(folder):
    publication = read(folder / "publication.json")
    assert {p.name for p in folder.iterdir() if p.is_file()} == set(publication["files"]) | {"publication.json"}
    for name, expected in publication["files"].items():
        raw = (folder / name).read_bytes()
        assert len(raw) == expected["bytes"] and sha256(raw).hexdigest() == expected["sha256"], name
    original_input, original_lock = check_input(folder / "failed-startup-inputs.zip")
    successful_input, successful_lock = check_input(folder / "successful-inputs.zip")
    try:
        assert original_lock["protocol.json"] == successful_lock["protocol.json"] == publication["protocol_sha256"]
        assert original_lock["train.py"] != successful_lock["train.py"]
        assert successful_lock["train.py"] == publication["successful_worker_sha256"]
        with verify_archive(folder / "failed-startup.zip") as failure:
            failed = zipfile.Path(failure)
            assert read(failed / "output/status.json")["stage"] == "load"
            assert read(failed / "output/status.json")["optimizer_steps"] == 0
            assert not any(n.startswith(("output/frozen/", "output/initial-adapter")) for n in failure.namelist())
            assert "incompatible version of torchao" in (failed / "worker.log").read_text(encoding="utf-8")
            assert sha256(failure.read("train.py")).hexdigest() == original_lock["train.py"]
            assert read(failed / "launch-receipt.json")["input_zip_sha256"] == sha256((folder / "failed-startup-inputs.zip").read_bytes()).hexdigest()
        with verify_archive(folder / "gpu-results.zip") as successful:
            root = zipfile.Path(successful)
            assert read(root / "launch-receipt.json")["input_zip_sha256"] == sha256((folder / "successful-inputs.zip").read_bytes()).hexdigest()
            for name in successful_lock:
                assert successful.read(name) == successful_input.read(name)
            actual = score(root)
        expected = read(folder / "scored.json")
        audited = expected.pop("independently_tokenized_inputs")
        assert audited == publication["independently_tokenized_inputs"] == 550
        # Optional tokenizer audit was run locally with the pinned tokenizer.
        # CI recomputes all other metrics and never downloads model/tokenizer files.
        match(actual, expected)
    finally:
        original_input.close()
        successful_input.close()
    return dict(published_file_hashes_verified=len(publication["files"]), adapter_arms_verified=8,
                model_continuations_replayed=12, rule_continuations_replayed=12,
                all_scores_reproduced=True, failed_startup_retained=True, gpu_or_network_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir)))
