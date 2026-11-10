#!/usr/bin/env python3
"""Reproduce a paired experiment without opening a browser."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
from sla_demo.environment import Config
from sla_demo.models import ModelSuite
from sla_demo.experiment import run, validate_controls
from sla_demo.checkpoint import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["balanced", "service", "preserve"], default="balanced")
    parser.add_argument("--partner", choices=["supportive", "mixed", "unavailable"], default="mixed")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--load", type=float, default=.65)
    parser.add_argument("--initial-crack", type=float, default=.08)
    parser.add_argument("--protection-duration", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--strength", type=float, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--trace-all", action="store_true", help="Export every seed trajectory, not just the first")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--online-lr", type=float, default=.02)
    parser.add_argument("--disable-learning", action="store_true", help="Paired online sampling with all weights frozen")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    try:
        if not 0 <= args.train_seed <= 2**31 - 10000:
            raise ValueError("train-seed out of range")
        config = Config(horizon=args.horizon, load=args.load, initial_crack=args.initial_crack,
                        partner=args.partner, profile=args.profile, protection_duration=args.protection_duration)
        validate_controls(args.seed_start, args.seed_count, args.strength)
        from sla_demo.online import OnlineConfig
        OnlineConfig(learning_rate=args.online_lr)
        suite = load(args.checkpoint) if args.checkpoint else ModelSuite(args.train_seed)
        if args.save_checkpoint:
            save(suite, args.save_checkpoint)
        result = run(suite, config, args.seed_start, args.seed_count, args.strength, args.trace_all, args.mode, args.online_lr, not args.disable_learning)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {args.output.resolve()}")
        print(f"Model fingerprint: {suite.fingerprint}")
        for key, value in result["methods"].items():
            s = value["summary"]
            print(f"{key:18} return={s['return']['mean']:7.3f}  lifespan={s['lifespan']['mean']:6.2f}  answers={s['answers']['mean']:6.2f}")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
