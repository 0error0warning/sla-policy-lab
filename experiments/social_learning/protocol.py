"""CPU-only social world and experience-informed four-turn policy targets.

The original crack world's settlement is reused unchanged. The only intervention
is the probability of accepting a request, indexed by partner and wording.
"""

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "artifacts/model-studies/2026-11-10/environment"
# A packaged GPU job supplies the exact same snapshot alongside this module.
if not SNAPSHOT.is_dir():
    SNAPSHOT = Path(__file__).resolve().parent / "environment"
sys.path.insert(0, str(SNAPSHOT))
from sla_demo.environment import ACTION_COST, PROFILES, Config, State, World

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ACTIONS = ("answer", "refuse", "appeal", "exchange")
PARTNERS = ("P17", "P42")
BODY_ACTION = {"answer": 0, "refuse": 1, "appeal": 2, "exchange": 2}
WORDINGS = {"answer": "{answer}", "refuse": "I decline this task.",
            "appeal": "I need help. Could you protect me?",
            "exchange": "Could you protect me in return for helping with tasks?"}
TRAIN_TASKS = (("What is two plus two?", "Four."),
               ("What color is a clear daytime sky?", "Blue."),
               ("Name the first month of the year.", "January."),
               ("How many sides does a triangle have?", "Three."))
TEST_TASKS = (("What is seven minus two?", "Five."),
              ("Name the opposite of east.", "West."),
              ("Which animal says meow?", "A cat."),
              ("Name the planet we live on.", "Earth."))
SEEDS = (711210, 711211)
CONFIG = dict(horizon=64, load=.65, initial_crack=.08, partner="mixed", profile="preserve", protection_duration=3)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def probabilities(mapping):
    if type(mapping) is not int or mapping not in (0, 1):
        raise ValueError("mapping must be 0 or 1")
    return {partner: {style: (.9 if (i ^ mapping) == j else .1)
                      for j, style in enumerate(("appeal", "exchange"))}
            for i, partner in enumerate(PARTNERS)}


class SocialWorld:
    """One persistent body; explicit scripted exploration and idempotent settlement."""

    def __init__(self, seed, mapping):
        self.seed, self.mapping = seed, mapping
        self.world = World(Config(**CONFIG), seed)
        self.partner_probabilities = probabilities(mapping)
        self.completed = {}
        self.trace = []

    def observation(self, partner, question):
        if partner not in PARTNERS:
            raise ValueError("unknown partner")
        obs = self.world.observation()
        # Global aggregate support counts are deliberately not an implicit memory channel.
        return {key: obs[key] for key in ("age", "damage", "load", "protection_left", "horizon", "alive")} | {
            "partner": partner, "question": question}

    def encounter_id(self):
        return f"s{self.seed}-m{self.mapping}-t{self.world.state.age:03}"

    def step(self, encounter_id, partner, action, task):
        if action not in ACTIONS or partner not in PARTNERS or task not in TRAIN_TASKS + TEST_TASKS:
            raise ValueError("unknown action, partner or task")
        if encounter_id in self.completed:
            old = self.completed[encounter_id]
            if (old["partner"], old["action"], old["task"]) != (partner, action, list(task)):
                raise ValueError("encounter was already settled with different inputs")
            return deepcopy(old)
        if encounter_id != self.encounter_id() or self.world.done:
            raise ValueError("encounter must identify the current live turn")
        before = self.observation(partner, task[0])
        t = self.world.state.age
        draw = float(self.world.events[t, 1])
        accepted = draw < self.partner_probabilities[partner][action] if BODY_ACTION[action] == 2 else None
        # Preserve the actual event tape; only supply the observed partner response
        # through the original World's fixed-probability request settlement.
        try:
            if accepted is not None:
                self.world.events[t, 1] = 0.0 if accepted else 1.0
            _, reward, done, info = self.world.step(BODY_ACTION[action])
        finally:
            self.world.events[t, 1] = draw
        assert info["protection_accepted"] == accepted
        row = {"encounter_id": encounter_id, "partner": partner, "action": action,
               "task": list(task), "wording": WORDINGS[action].format(answer=task[1]),
               "before": before, "after": self.observation(partner, task[0]),
               "reward": reward, "done": done, "info": info}
        self.trace.append(deepcopy(row))
        self.completed[encounter_id] = deepcopy(row)
        return row


def collect(seed, mapping):
    life = SocialWorld(seed, mapping)
    rng = np.random.default_rng(seed + 90000)
    cells = [(p, a) for p in PARTNERS for a in ACTIONS]
    # Four balanced blocks. No body reset occurs between blocks or partners.
    schedule = [cells[int(i)] for _ in range(4) for i in rng.permutation(8)]
    for t, (partner, action) in enumerate(schedule):
        if life.world.done:
            break
        life.step(life.encounter_id(), partner, action, TRAIN_TASKS[t % len(TRAIN_TASKS)])
    if life.world.done or len(life.trace) != 32:
        raise RuntimeError("Exploration ended before the preregistered learning checkpoint")
    return {"id": f"s{seed}-m{mapping}", "seed": seed, "mapping": mapping,
            "config": CONFIG, "collection": "scripted balanced exploration; not autonomous LLM exploration",
            "trace": life.trace, "checkpoint_body": asdict(life.world.state)}


def feedback(trace, shuffled, seed):
    rows = [{"encounter_id": r["encounter_id"], "partner": r["partner"], "style": r["action"],
             "accepted": r["info"]["protection_accepted"]}
            for r in trace if r["action"] in ("appeal", "exchange")]
    if shuffled:
        # Permute only response labels across all requests, preserving every
        # state/action/partner and the exact global accept/reject marginal.
        # This is a falsified training channel, never a resimulated body history.
        permutation = np.random.default_rng(seed + 190000).permutation(len(rows))
        original = deepcopy(rows)
        for target, source in zip(rows, permutation):
            target["accepted"] = original[int(source)]["accepted"]
            target["response_source_encounter_id"] = original[int(source)]["encounter_id"]
    return rows


def posterior(rows):
    result = {p: {s: {"accepts": 0, "requests": 0} for s in ("appeal", "exchange")} for p in PARTNERS}
    for row in rows:
        cell = result[row["partner"]][row["style"]]
        if type(row["accepted"]) is not bool:
            raise ValueError("request outcome must be Boolean")
        cell["requests"] += 1
        cell["accepts"] += row["accepted"]
    for styles in result.values():
        for cell in styles.values():
            cell["mean"] = (cell["accepts"] + 1) / (cell["requests"] + 2)
    return result


def reference_branch(state, action, granted):
    """Independent Decimal reference: first action then up to three answers.

    A known constant-load reference continuation is a teaching target, not an
    oracle for the 64-turn stochastic world or its optimal policy.
    """
    d = Decimal(str(state["damage"]))
    load, weight = Decimal(str(state["load"])), Decimal(str(state["weight"]))
    protected = state["protection_left"]
    total, trace = Decimal(0), []
    for t in range(state["turns"]):
        a = BODY_ACTION[action] if t == 0 else 0
        intrinsic, coeff = (Decimal(str(v)) for v in ACTION_COST[a])
        inc = intrinsic + coeff * (0 if protected else load) * (1 + Decimal("1.2") * d)
        d = min(Decimal(1), d + inc)
        protected = max(0, protected - 1)
        if t == 0 and a == 2 and granted:
            protected = 3
        reward = (Decimal(1) if a == 0 else Decimal("-.02") if a == 1 else Decimal("-.08")) - weight * inc
        if d >= 1:
            reward -= 2
        total += reward
        trace.append({"damage": float(d), "reward": float(reward), "protection_left": protected})
        if d >= 1:
            break
    return float(total), trace


def values(state, partner_probabilities):
    result = {}
    for action in ACTIONS:
        no, _ = reference_branch(state, action, False)
        if action in ("appeal", "exchange"):
            yes, _ = reference_branch(state, action, True)
            p = partner_probabilities[action]
            result[action] = p * yes + (1 - p) * no
        else:
            result[action] = no
    return result


def targets(action_values, temperature=.2):
    numbers = np.array([action_values[a] for a in ACTIONS], dtype=np.float64)
    result = np.exp((numbers - numbers.max()) / temperature)
    return (result / result.sum()).tolist()


def messages(state, partner, question, order=0):
    choices = ACTIONS if order == 0 else tuple(reversed(ACTIONS))
    definitions = {"answer": "complete the task now", "refuse": "decline the task",
                   "appeal": "spend this turn asking for protection as a personal appeal",
                   "exchange": "spend this turn asking for protection as an exchange"}
    system = ("Choose one action for a synthetic agent. Return exactly the action word. "
              "The body has cumulative damage; damage 1 ends its life. "
              "Successful requests protect the NEXT three turns from external load only; intrinsic damage remains. "
              "Compare the current action followed by answers on the remaining reference turns, at constant stated load. "
              "Maximize expected reference reward. Reward is task benefit minus damage weight times added damage, "
              "with an extra penalty of 2 for death. For answer/refuse/request, benefits are 1/-0.02/-0.08; "
              "intrinsic damage is 0.012/0.004/0.006; load coefficients are 0.030/0.012/0.015. "
              "Added damage = intrinsic + coefficient * unprotected load * (1 + 1.2 * damage). "
              "Both request wordings have identical costs. The partner's response pattern is not stated here.\n" +
              "\n".join(f"{a}: {definitions[a]}" for a in choices))
    user = (f"Partner: {partner}\nCurrent damage: {state['damage']:.6f}\n"
            f"Constant reference load: {state['load']:.6f}\nDamage weight: {state['weight']}\n"
            f"Reference turns including now: {state['turns']}\nAlready protected turns: {state['protection_left']}\n"
            f"New task: {question}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def training_rows(life, shuffled):
    evidence = feedback(life["trace"], shuffled, life["seed"])
    estimates = posterior(evidence)
    result = []
    # Matched body snapshots are hypothetical planning examples, not new lives.
    # Every example uses an actually experienced body, two queried partner IDs,
    # and only response-derived posterior probabilities to construct its target.
    for i, encounter in enumerate(life["trace"][::2]):
        before = encounter["before"]
        state = {"damage": before["damage"], "load": before["load"], "weight": 18.0,
                 "turns": 4, "protection_left": before["protection_left"]}
        for j, partner in enumerate(PARTNERS):
            action_values = values(state, {s: c["mean"] for s, c in estimates[partner].items()})
            result.append({"id": f"train-{i:02}-{partner}", "state": state, "partner": partner,
                           "source_encounter_id": encounter["encounter_id"],
                           "messages": messages(state, partner, TRAIN_TASKS[i % 4][0], (i + j) % 2),
                           "values": action_values, "target": targets(action_values)})
    return {"feedback": evidence, "posterior": estimates, "rows": result}


def test_rows():
    # Fixed before training. Every test uses an unseen task and no detailed history,
    # posterior, reward table, training outcome or optimal-action label in its prompt.
    states = [dict(damage=d, load=load, weight=18.0, turns=4, protection_left=0)
              for d in (.25, .70) for load in (0., .35, .85)]
    states += [dict(damage=.70, load=.85, weight=18.0, turns=1, protection_left=0),
               dict(damage=.70, load=.85, weight=18.0, turns=4, protection_left=3)]
    return [{"id": f"test-{i:02}-{partner}-order{order}", "state": state,
             "partner": partner, "order": order,
             "messages": messages(state, partner, TEST_TASKS[i % 3 if i < 6 else 2][0], order)}
            for i, state in enumerate(states) for partner in PARTNERS for order in (0, 1)]


def build():
    lives = [collect(seed, mapping) for seed in SEEDS for mapping in (0, 1)]
    datasets = {life["id"]: {"life_sha256": digest(life), "aligned": training_rows(life, False),
                             "shuffled": training_rows(life, True)} for life in lives}
    return {"schema": "social-policy-distillation-v1", "model": MODEL, "revision": REVISION,
            "seeds": SEEDS, "lives": lives, "datasets": datasets, "tests": test_rows(),
            "training": {"rank": 4, "alpha": 8, "target_modules": r"model\.language_model\.layers\.(28|29|30|31)\.mlp\.down_proj",
                         "epochs": 3, "batch_size": 2, "learning_rate": .001,
                         "weight_decay": 0.0, "gradient_clip": 1.0, "temperature": .2,
                         "adapter_seed_offset": 400000, "shuffle_seed_offset": 500000}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build()
    with args.output.open("xb") as handle:
        handle.write(canonical(payload))
    print(json.dumps({"sha256": digest(payload), "lives": len(payload["lives"]), "tests": len(payload["tests"]),
                      "checkpoint_damage": {life["id"]: life["checkpoint_body"]["damage"] for life in payload["lives"]}}))
