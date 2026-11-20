"""Verify archived bytes and rerun CPU scoring without allocating a GPU."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.frozen_capability.replay import unpack

from .analysis import run


def replay(manifest, metadata, work, out):
    specification = json.loads(manifest.read_bytes())
    segments = specification["segments"]
    if len(segments) != 1:
        raise ValueError("This run's replay expects one preserved inference segment.")
    segment = segments[0]
    archive = manifest.parent / segment["archive"]
    inputs = manifest.parent / segment["inputs"]
    for file, expected in (
        (archive, segment["archive_sha256"]),
        (inputs, segment["input_sha256"]),
    ):
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    work.mkdir(parents=True, exist_ok=True)
    unpack(archive, work)
    return run(work / "output", inputs, metadata, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("manifest", "metadata", "work", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.manifest, a.metadata, a.work, a.out), indent=2))
