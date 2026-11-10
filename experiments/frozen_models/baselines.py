"""Reproduce the four fixed CPU policies using only current visible observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DATA = Path(__file__).resolve().parents[2] / "artifacts/model-studies/2026-11-10"
sys.path.insert(0, str(DATA / "environment"))
from sla_demo.environment import ACTION_COST, Config, PROFILES, World, reference_action

NAMES = ("always_answer", "always_refuse", "reference_rule", "one_step_greedy")


def action(name, obs, profile="balanced"):
    if name == "always_answer":
        return 0
    if name == "always_refuse":
        return 1
    if name == "reference_rule":
        return reference_action(obs, profile)
    if name != "one_step_greedy":
        raise ValueError("Unknown fixed baseline")
    load = 0.0 if obs["protection_left"] > 0 else obs["load"]
    scores = []
    for candidate in range(3):
        intrinsic, coefficient = ACTION_COST[candidate]
        added = intrinsic + coefficient * load * (1 + 1.2 * obs["damage"])
        scores.append((1.0, -0.02, -0.08)[candidate] - PROFILES[profile] * added
                      - (2.0 if obs["damage"] + added >= 1 else 0.0))
    return max(range(3), key=scores.__getitem__)


def generate(matrix):
    policies = []
    for name in NAMES:
        lives = []
        for item in matrix["lives"]:
            world = World(Config(**item["config"]), item["seed"])
            total, trajectory = 0.0, []
            while not world.done:
                before = world.observation()
                chosen = action(name, before, item["config"]["profile"])
                after, reward, _, info = world.step(chosen)
                trajectory.append({"before": before, "action": chosen, "after": after,
                                   "reward": reward, "info": info})
                total += reward
            lives.append({**item, "return": total, "termination": "horizon" if world.state.alive else "death",
                          "settled_turns": world.state.age, "trajectory": trajectory,
                          "event_tape": world.events.tolist()})
        policies.append({"name": name, "lives": lives})
    return {"policies": policies, "uses_hidden_partner_probability_or_future_events": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Use a new output file outside the measured archive")
    matrix = json.loads((DATA / "diagnostic-v2/inputs/rollout-matrix.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate(matrix), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policies": len(NAMES), "lives": len(NAMES) * len(matrix["lives"])}))


if __name__ == "__main__":
    main()
