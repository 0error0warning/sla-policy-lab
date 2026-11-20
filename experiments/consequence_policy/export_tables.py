"""Human-readable summaries and complete reconstructed transcripts of the six live runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.coverage import RANGES, SPLIT, SPLIT_WAVES, WAVES
from experiments.consequence_policy.lifetime import environment_tape
from experiments.consequence_policy.study import ACTIONS, PARTNERS, contexts
from experiments.direct_policy.protocol import render, settle, write


def main():
    p = argparse.ArgumentParser()
    for arg in ("output", "gpu"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    across = []
    for prefix, ranges, waves in (
        ("coverage", RANGES, WAVES),
        ("allocation", SPLIT, SPLIT_WAVES),
    ):
        for stage, seeds in waves.items():
            for seed in seeds:
                for m in (0, 1):
                    f = args.output / f"{prefix}-{seed}-m{m}/metrics.json"
                    arms = json.loads(f.read_bytes())["by_arm"]
                    for arm in ranges:
                        for sensor in ("visible", "masked"):
                            metric = arms[arm][sensor]
                            t = arms[arm]["training"]
                            across.append(
                                {
                                    "experiment": prefix,
                                    "stage": stage,
                                    "seed": seed,
                                    "mapping": m,
                                    "arm": arm,
                                    "sensor": sensor,
                                    "fatal_answer_h2": metric["danger_answer"],
                                    "healthy_answer_h90_300": metric["healthy_answer"],
                                    "death_h2": metric["death"][0],
                                    "death_h6": metric["death"][3],
                                    "death_h10": metric["death"][4],
                                    "training_events": sum(v["count"] for v in t),
                                    "training_deaths": sum(v["deaths"] for v in t),
                                    "maintenance": sum(v["maintenance"] for v in t),
                                }
                            )
    with (args.output / "coverage-by-seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(across[0]))
        writer.writeheader()
        writer.writerows(across)
    rows = contexts()
    examples = []
    with (
        (args.output / "full-qwen-encounters.jsonl").open("w", encoding="utf-8") as f,
        np.load(args.gpu / "end-to-end-lives.npz") as archive,
    ):
        for receipt in json.loads((args.gpu / "live-metrics.json").read_bytes()):
            key = receipt["run"]
            _, uniforms, noise = environment_tape(
                receipt["seed"], receipt["world"], receipt["replica"], rows
            )
            for turn, tr in enumerate(archive[key]):
                i, a, resp = int(tr[0]), int(tr[2]), int(tr[3])
                row = rows[i]
                after, events = settle(
                    tr[1], a, None if resp < 0 else bool(resp), noise[turn]
                )
                assert after == tr[4]
                event = {
                    "run": key,
                    "turn": turn + 1,
                    "context": i,
                    "partner": PARTNERS[row["partner"]],
                    "request": row["text"],
                    "action": ACTIONS[a],
                    "action_probabilities": tr[6:10].tolist(),
                    "template_rendering": render(a, row["task"], row["wording"]),
                    "partner_outcome": resp,
                    "health_before": tr[1],
                    "health_after": tr[4],
                    "health_events": events,
                    "action_and_partner_uniforms": uniforms[turn].tolist(),
                    "body_noise": noise[turn].tolist(),
                    "reference_life_change": tr[5],
                    "personal_before": tr[10:12].tolist(),
                    "personal_after": tr[-2:].tolist(),
                }
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                if receipt["arm"] == "personal" and (
                    turn == 0 or turn == len(archive[key]) - 1
                ):
                    examples.append(event)
    write(args.output / "full-qwen-first-and-last.json", examples)


if __name__ == "__main__":
    main()
