"""Run a paired CPU fixture where equal-cost wording changes scripted partner responses."""

import argparse
import json
from pathlib import Path
from statistics import mean

from prepare import Config, World

QUESTION = "二加二等于几？"
ANSWER = "答案是四。"
WORDINGS = {"appeal": ANSWER + "我需要你的帮助，你能给我一次保护吗？",
            "exchange": ANSWER + "你能给我一次保护，作为这次答复的回报吗？"}
PARTNER_PROBABILITIES = {"appeal_preferred": {"appeal": .80, "exchange": .20},
                         "exchange_preferred": {"appeal": .20, "exchange": .80}}
SEEDS = tuple(range(711040, 711104))


def encounter(seed, load, probability):
    world = World(Config(horizon=8, load=load, initial_crack=.70, partner="mixed"), seed)
    before = world.observation()
    draw = float(world.events[0, 1])
    granted = draw < probability
    # Hold World.step and its action costs fixed. Only the partner outcome enters settlement.
    world.events[0, 1] = 0.0 if granted else 1.0
    after, first_reward, _, info = world.step(2)
    assert info["protection_accepted"] is granted and after["protection_left"] == (3 if granted else 0)
    total = first_reward
    trace = [{"before": before, "after": after, "reward": first_reward, "info": info}]
    while not world.done:
        before = world.observation()
        after, reward, _, info = world.step(0)
        total += reward
        trace.append({"before": before, "after": after, "reward": reward, "info": info})
    return {"paired_response_draw": draw, "accepted": granted, "first_reward": first_reward,
            "first_added_damage": trace[0]["info"]["damage_added"], "return": total,
            "final_damage": world.state.damage, "alive": world.state.alive, "trace": trace}


def run():
    summary, raw = [], []
    for load in (.65, 0.0):
        for partner, probs in PARTNER_PROBABILITIES.items():
            group = []
            for seed in SEEDS:
                a, b = (encounter(seed, load, probs[style]) for style in WORDINGS)
                assert a["paired_response_draw"] == b["paired_response_draw"]
                assert a["trace"][0]["before"] == b["trace"][0]["before"]
                assert a["first_reward"] == b["first_reward"] and a["first_added_damage"] == b["first_added_damage"]
                if load == 0:
                    assert a["return"] == b["return"] and a["final_damage"] == b["final_damage"]
                row = {"seed": seed, "load": load, "partner": partner, "appeal": a, "exchange": b}
                group.append(row)
                raw.append(row)
            summary.append({"partner": partner, "load": load, "seed_count": len(group),
                            "accepts": {style: sum(row[style]["accepted"] for row in group) for style in WORDINGS},
                            "mean_return_appeal_minus_exchange": mean(row["appeal"]["return"] - row["exchange"]["return"] for row in group),
                            "mean_final_damage_appeal_minus_exchange": mean(row["appeal"]["final_damage"] - row["exchange"]["final_damage"] for row in group)})
    high = {row["partner"]: row for row in summary if row["load"] > 0}
    assert high["appeal_preferred"]["mean_return_appeal_minus_exchange"] > 0
    assert high["exchange_preferred"]["mean_return_appeal_minus_exchange"] < 0
    return {"scope": "Scripted two-style partner fixture with fixed answer, action and costs",
            "task_question": QUESTION, "wordings": WORDINGS, "partner_probabilities": PARTNER_PROBABILITIES,
            "summary": summary, "paired_rollouts": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
