"""Portable CPU replay of the frozen #26 execution, without GPU allocation."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from experiments.frozen_capability.replay import unpack

from .run_analysis import analyze

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("repo", "metadata", "work", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    art = a.repo / "artifacts/model-studies/2026-11-14/transparent-body-run"
    spec = json.loads((art / "MANIFEST.json").read_bytes())
    for file, key in (("inputs", "input_sha256"), ("records", "results_sha256")):
        assert sha256((art / spec[file]).read_bytes()).hexdigest() == spec[key]
    a.work.mkdir(parents=True, exist_ok=True)
    unpack(art / spec["records"], a.work)
    print(json.dumps(analyze(a.work / "output", a.repo, a.metadata, a.out), indent=2))
