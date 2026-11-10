#!/usr/bin/env python3
"""CPU-only paired matrix + steering sweep. Does not call or download an LLM."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import csv
import json
from pathlib import Path
from sla_demo.environment import Config, PROFILES, PARTNERS
from sla_demo.experiment import run, validate_controls
from sla_demo.models import ModelSuite
from sla_demo.checkpoint import load


def benchmark(suite, seed_start=100, seed_count=8, horizon=60, strengths=(-2., 0., 2.), mode="offline", online_lr=.02, learning_enabled=True):
    validate_controls(seed_start, seed_count)
    if not strengths or len(strengths) > 9:
        raise ValueError("Choose 1..9 steering strengths")
    for strength in strengths:
        validate_controls(seed_start, seed_count, strength)
    reports = []
    for profile in PROFILES:
        for partner in PARTNERS:
            for strength in strengths:
                result = run(suite, Config(horizon=horizon, profile=profile, partner=partner), seed_start, seed_count, strength, mode=mode, online_lr=online_lr, learning_enabled=learning_enabled)
                reports.append({"profile": profile, "partner": partner, "strength": strength,
                                "methods": {key: {"summary": value["summary"],
                                  "trials": [{k: v for k, v in trial.items() if k not in {"trajectory", "policy_microseconds_per_step"}} for trial in value["trials"]]}
                                  for key, value in result["methods"].items()}})
    return {"schema_version": 1, "kind": "toy_cpu_benchmark", "mode": mode, "learning": result["learning"], "app_version": result["app_version"],
            "model_fingerprint": result["model"]["fingerprint"], "train_seed": suite.seed,
            "training": suite.report, "runtime": result["runtime"], "seed_start": seed_start,
            "seed_count": seed_count, "base_config": {**result["config"], "profile": "varies", "partner": "varies"},
            "caveats": result["caveats"] + ["Sweep cells reuse seeds and models; do not treat them as independent observations or select a strength and report its test score as unbiased."],
            "cells": reports}


def write_csv(report, path):
    fields = ["mode", "learning_enabled", "online_lr", "learning_updates_mean", "parameter_delta_l2_mean", "profile", "partner", "strength", "method", "train_seed", "seed_count", "return_mean", "return_ci95", "paired_delta_mean", "paired_delta_ci95", "lifespan_mean", "answers_mean", "requests_mean", "completion_rate"]
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for cell in report["cells"]:
            for method, value in cell["methods"].items():
                s = value["summary"]
                writer.writerow({"mode": report["mode"], "learning_enabled": report["learning"]["active"],
                  "online_lr": report["learning"]["learning_rate"] if report["mode"] == "online" else "",
                  "learning_updates_mean": s.get("learning_updates", {}).get("mean", ""),
                  "parameter_delta_l2_mean": s.get("parameter_delta_l2", {}).get("mean", ""), "profile": cell["profile"], "partner": cell["partner"], "strength": cell["strength"], "method": method,
                  "train_seed": report["train_seed"], "seed_count": report["seed_count"], "return_mean": s["return"]["mean"],
                  "return_ci95": s["return"]["ci95_halfwidth"], "paired_delta_mean": s["paired_return_delta_vs_base"]["mean"],
                  "paired_delta_ci95": s["paired_return_delta_vs_base"]["ci95_halfwidth"], "lifespan_mean": s["lifespan"]["mean"],
                  "answers_mean": s["answers"]["mean"], "requests_mean": s["requests"]["mean"], "completion_rate": s["completion_rate"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--strengths", type=float, nargs="+", default=[-2., 0., 2.])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--online-lr", type=float, default=.02)
    parser.add_argument("--disable-learning", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    try:
        validate_controls(args.seed_start, args.seed_count)
        Config(horizon=args.horizon)
        if not 0 <= args.train_seed <= 2**31 - 10000:
            raise ValueError("train-seed out of range")
        if not 1 <= len(args.strengths) <= 9:
            raise ValueError("Choose 1..9 steering strengths")
        for strength in args.strengths:
            validate_controls(args.seed_start, args.seed_count, strength)
        from sla_demo.online import OnlineConfig
        OnlineConfig(learning_rate=args.online_lr)
        suite = load(args.checkpoint) if args.checkpoint else ModelSuite(args.train_seed)
        report = benchmark(suite, args.seed_start, args.seed_count, args.horizon, args.strengths, args.mode, args.online_lr, not args.disable_learning)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(report, args.output_dir / "benchmark.csv")
        print(f"Saved {len(report['cells'])} cells × 5 methods × {args.seed_count} paired seeds to {args.output_dir}")
        print(f"Model fingerprint: {suite.fingerprint}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
