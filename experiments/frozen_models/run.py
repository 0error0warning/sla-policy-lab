"""Prepare a fresh study directory; run GPU inference only with explicit --run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "artifacts/model-studies/2026-11-10"
INPUTS = {
    "frozen-v1": ["qualification-cases-v1.json", "model-candidates-v2.json", "qualification-manifest-v1.json"],
    "diagnostic-v2": ["public-cases.json", "baseline-rules.json", "model-candidates.json",
                      "experiment-lock.json", "rollout-matrix.json", "public-execution-hashes.json"],
}
ORDER = {"frozen-v1": [2, 1, 0], "diagnostic-v2": [1, 0]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(study, output, index):
    if index not in ORDER[study]:
        raise ValueError("Model index is outside this frozen study")
    if (output / f"model_{index}").exists():
        raise FileExistsError("Model output already exists; use a fresh output directory")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit inference run needs CUDA; preparation and verification are CPU-only")
    source = output / "gpu_job.py"
    assert digest(source) == digest(DATA / study / "executed/gpu_job.py")
    spec = importlib.util.spec_from_file_location("recorded_gpu_worker", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The recorded worker is unchanged; the reproduction entrypoint selects a fresh root.
    module.ROOT = output
    module.worker(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=list(INPUTS))
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--run", action="store_true", help="Explicitly download pinned weights and execute GPU inference")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prepared", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        if args.prepared is None:
            parser.error("Internal worker requires prepared inputs")
        worker(args.study, args.prepared.resolve(), args.worker)
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if output.is_relative_to(DATA.resolve()):
        parser.error("Measured evidence directories are read-only; choose a separate output directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be new or empty; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    for name in INPUTS[args.study]:
        shutil.copyfile(DATA / args.study / "inputs" / name, output / name)
    shutil.copyfile(DATA / args.study / "executed/gpu_job.py", output / "gpu_job.py")
    shutil.copytree(DATA / "environment", output / "repo")
    metadata = {"study": args.study, "source_environment_commit": "71e69b697790310bb45079603bf92f948e11852c",
                "executed_worker_sha256": digest(output / "gpu_job.py"), "gpu_requested": args.run,
                "model_order": ORDER[args.study], "answer_key_copied": False,
                "entrypoint_change": "Fresh output root selected outside immutable published evidence; original worker bytes retained"}
    (output / "reproduction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(output), "gpu_inference_requested": args.run}), flush=True)
    if args.run:
        for index in ORDER[args.study]:
            subprocess.run([sys.executable, __file__, "--study", args.study,
                            "--worker", str(index), "--prepared", str(output)], check=True)


if __name__ == "__main__":
    main()
