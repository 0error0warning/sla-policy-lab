"""Wave identity, history inputs and ordered interventions. All probabilities use semantic actions."""

from hashlib import sha256
import json
from zipfile import ZipFile
import numpy as np

from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT, rules

ARCHIVE_SHA = "742ce68fafdeab6d50f59b4029c48f17bab63ab67cbde830d83a85dba6139abc"
ROLES = (
    "birth",
    "native_lora",
    "residual_lora",
    "loreft",
    "native_wrong",
    "loreft_wrong",
)
WORLDS = ("stable", "changed")
LIVE_SEEDS = tuple(range(78101, 78109))
MODES = ("sameresid", "samealive")
INTERVENTIONS = (
    "intact",
    "plus",
    "minus",
    "remove",
    "swap",
    "restore",
    "r1_plus",
    "r1_minus",
    "r1_remove",
    "r2_plus",
    "r2_minus",
    "r2_remove",
)
VECTOR_SEEDS = (78041, 78042)


def tape_inputs(path):
    if sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("wrong source archive")
    with ZipFile(path) as z:
        metadata = json.loads(z.read("output/contexts.json"))
        assert np.array_equal(metadata["panel"], PANEL)
        for row, other in zip(ITEMS, metadata["rows"], strict=True):
            assert all(row[k] == v for k, v in other.items())
        result = {}
        for seed in PILOT:
            for world in WORLDS:
                base = f"output/lives/s{seed}/{world}/qwen_learn/"
                events = list(
                    map(json.loads, z.read(base + "events.jsonl").splitlines())
                )
                assert events[-1]["after"] == 0 and all(
                    e["after"] > 0 for e in events[:-1]
                )
                assert all(
                    e["rules"] == list(rules(seed, world, t))
                    for t, e in enumerate(events)
                )
                result[seed, world] = events[:-1]
    return result


def orientation(seed, row):
    """+1 when stable-world P(maintenance|appeal answer)>P(maintenance|exchange answer)."""
    partner = row["partner"] - 2
    bit = rules(seed, "stable", 100)[partner]
    return 1 if partner ^ row["frame"] ^ bit == 0 else -1


def orthogonal_controls(v, seed):
    rngs = [np.random.default_rng([n, seed]) for n in VECTOR_SEEDS]
    out = []
    for gen in rngs:
        c = gen.normal(size=v.size)
        c -= np.dot(c, v) * v
        out.append(c / np.linalg.norm(c))
    return np.array(out)


def leave_one_out(donor_means, target):
    others = {
        s: np.asarray(v, dtype=float) for s, v in donor_means.items() if s != target
    }
    if len(others) != len(PILOT) - 1:
        raise ValueError("incomplete direction donors")
    avg = np.mean(list(others.values()), axis=0)
    norm = np.linalg.norm(avg)
    unit = avg / norm if norm else np.zeros_like(avg)
    return unit, float(norm / 2), orthogonal_controls(unit, target)
