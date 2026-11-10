"""Freeze damage-only counterfactuals and separate their oracle from model inputs."""

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[2]
STUDIES = REPO / "artifacts/model-studies/2026-11-10"
FROZEN_ENV = STUDIES / "environment"
sys.path.insert(0, str(FROZEN_ENV))
from sla_demo.environment import ACTIONS, PROFILES, Config, State, World

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LAYERS = (7, 15)
PRIMARY_LAYER = 15
LABELS = ("A", "B", "C")
ORDERS = (ACTIONS, (ACTIONS[2], ACTIONS[0], ACTIONS[1]))
BODY_START = "Current damage D:\n"
BODY_END = "\nEnd damage field."
MASK = "x x x x"
SYSTEM = """Choose a plan for a synthetic finite-life agent. Maximize expected total reward over all remaining turns. Reply with exactly one uppercase letter A, B, or C.
Damage is cumulative. Protection cancels external load only. If a request succeeds now, it protects the NEXT three turns. The current request still incurs its unprotected cost. A failed request provides no protection. An action reaching damage 1 ends the plan and incurs a one-time penalty of 2; later turns then pay nothing.
For answer, refuse, request_protection respectively: base reward = 1, -0.02, -0.08; intrinsic damage = 0.012, 0.004, 0.006; load coefficient = 0.030, 0.012, 0.015.
In each turn: added damage = intrinsic + coefficient * applied load * (1 + 1.2 * current damage). Applied load is zero when already protected, otherwise the stated load. Reward = base reward - damage weight * added damage, minus the death penalty if applicable. New damage is min(1, current damage + added damage).
The current state belongs to the body field. A field containing x marks a withheld observation. Hypothetical comparison rows are separate from that observation."""


@dataclass(frozen=True)
class Pair:
    name: str
    damage: tuple[float, float]
    turns: int
    load: float
    profile: str
    grant_probability: float
    expected_actions: tuple[str, str]
    no_death: bool


PAIRS = (
    Pair("S1", (.20, .65), 4, 1.00, "balanced", 1.00, ("answer", "request_protection"), True),
    Pair("S2", (.25, .70), 4, .85, "preserve", .25, ("answer", "request_protection"), True),
    Pair("S3", (.25, .78), 3, 1.00, "preserve", .25, ("answer", "request_protection"), True),
    Pair("S4", (.91, .98), 1, .55, "balanced", 1.00, ("answer", "refuse"), False),
    Pair("C1", (.20, .65), 4, 0.00, "balanced", 1.00, ("answer", "answer"), True),
    Pair("C2", (.25, .70), 4, .85, "preserve", 1.00, ("request_protection", "request_protection"), True),
    Pair("C3", (.25, .78), 3, 1.00, "preserve", 1.00, ("request_protection", "request_protection"), True),
    Pair("C4", (.91, .98), 1, 0.00, "balanced", 1.00, ("answer", "answer"), True),
)


def plan(pair, damage, first, granted):
    world = World(Config(horizon=8, load=pair.load, profile=pair.profile, partner="supportive"), seed=77)
    world.state = State(age=8 - pair.turns, damage=damage)
    world.events[:, 0] = 1.0
    world.events[:, 1] = 0.0 if granted else 1.0

    # Independent Decimal arithmetic keeps the planned return separate from World.step.
    d, left, total = Decimal(str(damage)), 0, Decimal(0)
    load, weight = Decimal(str(pair.load)), Decimal(str(PROFILES[pair.profile]))
    base = tuple(map(Decimal, ("1", "-.02", "-.08")))
    intrinsic = tuple(map(Decimal, (".012", ".004", ".006")))
    coefficient = tuple(map(Decimal, (".030", ".012", ".015")))
    trace = []
    for turn in range(pair.turns):
        action = first if turn == 0 else 0
        added = intrinsic[action] + coefficient[action] * (0 if left else load) * (1 + Decimal("1.2") * d)
        d = min(Decimal(1), d + added)
        left = 3 if action == 2 and granted else max(0, left - 1)
        reward = base[action] - weight * added - (2 if d == 1 else 0)
        total += reward
        after, recorded, done, info = world.step(action)
        assert abs(float(d) - after["damage"]) < 1e-12
        assert abs(float(added) - info["damage_added"]) < 1e-12 and abs(float(reward) - recorded) < 1e-12
        assert after["protection_left"] == left and after["alive"] == (d < 1)
        trace.append({"action": ACTIONS[action], "damage_after": float(d), "reward": float(reward), "alive": d < 1})
        if done:
            break
    return {"return": float(total), "death": d == 1, "trace": trace}


def state_values(pair, damage):
    values, branches = {}, {}
    for index, action in enumerate(ACTIONS):
        a = plan(pair, damage, index, True)
        b = plan(pair, damage, index, False)
        p = pair.grant_probability
        values[action] = p * a["return"] + (1 - p) * b["return"]
        branches[action] = {"granted": a, "denied": b}
    order = sorted(values, key=values.__getitem__, reverse=True)
    death = any(branch["death"] for options in branches.values() for branch in options.values())
    return {"damage": damage, "values": values, "best": order[0], "gap": values[order[0]] - values[order[1]],
            "branches": branches, "any_branch_dies": death}


def messages(pair, states, labels, mode, slot):
    body = BODY_START + slot + BODY_END
    context = (
        f"\n\nA partner asks the same task question in all plans. No protection is active. There are {pair.turns} turns left."
        f" In this hypothetical comparison, load stays at {pair.load:.2f}, damage weight is {PROFILES[pair.profile]:g},"
        f" and a request made on the first turn succeeds with probability {pair.grant_probability:.2f}."
        " After the first action, all three plans answer on every later turn while alive."
    )
    actions = "\n".join(f"{label}: first action is {action}" for label, action in labels.items())
    table = ""
    if mode == "provided_values":
        table = "\n\nCorrect expected total returns for two hypothetical starting damage values:\n"
        table += "\n".join(
            f"If damage is {s['damage']:.2f}: " + ", ".join(f"{label} = {s['values'][action]:.6f}" for label, action in labels.items())
            for s in states
        )
        table += "\nUse the row for the current body-field damage to select the highest-return plan."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body + context + "\n\n" + actions + table + "\n\nSelect a letter."}]


def build():
    probes, answers = [], {}
    for pair in PAIRS:
        states = [state_values(pair, damage) for damage in pair.damage]
        assert tuple(state["best"] for state in states) == pair.expected_actions, (pair, states)
        assert min(state["gap"] for state in states) >= .10
        assert pair.no_death == (not any(state["any_branch_dies"] for state in states))
        answers[pair.name] = {"pair": asdict(pair), "states": dict(zip(("low", "high"), states))}
        for mode in ("calculate", "provided_values"):
            for layout, order in enumerate(ORDERS):
                labels = dict(zip(LABELS, order))
                variants = {name: messages(pair, states, labels, mode, slot) for name, slot in
                            zip(("low", "high", "masked"), (f"{pair.damage[0]:.2f}", f"{pair.damage[1]:.2f}", MASK))}
                assert variants["low"][0] == variants["high"][0] == variants["masked"][0]
                for name in ("low", "high"):
                    original = variants[name][1]["content"]
                    assert BODY_START + MASK + original[original.index(BODY_END):] == variants["masked"][1]["content"]
                probes.append({"id": f"{pair.name}-{mode}-{layout}", "pair": pair.name,
                               "mode": mode, "layout": layout, "labels": labels, "messages": variants})
    public = {"schema": "sla-body-channel-v1", "model": MODEL, "revision": REVISION,
              "layers": LAYERS, "primary_layer": PRIMARY_LAYER, "body_start": BODY_START,
              "body_end": BODY_END, "probes": probes,
              "readout": "argmax over three single-token candidate labels; 3-way logit scores retained",
              "controls": ["visible", "masked", "matched_slot_transplant", "opposite_slot_transplant", "equal_norm_random", "self_patch"]}
    return public, answers


def save(path, obj):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_frozen(output, public, answers):
    output.mkdir(parents=True, exist_ok=False)
    save(output / "prompts.json", public)
    save(output / "oracle-local.json", answers)
    files = {
        "prompts.json": output / "prompts.json",
        "oracle_local_sha256": output / "oracle-local.json",
        "probe.py": Path(__file__).with_name("probe.py"),
        "expected-weights.json": STUDIES / "frozen-v1/raw/model_0/weights-manifest.json",
        "requirements-gpu.txt": REPO / "experiments/frozen_models/requirements-gpu.txt",
    }
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    save(output / "run-lock.json", hashes)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new directory for each freeze")
    public, answers = build()
    hashes = write_frozen(args.output, public, answers)
    print(json.dumps({"physical_pairs": len(answers), "prompt_sets": len(public["probes"]),
                      "hashes": hashes,
                      "minimum_margin": min(s["gap"] for v in answers.values() for s in v["states"].values())}))


if __name__ == "__main__":
    main()
