"""Anchors, within-person ordinary controls and immutable input histories."""

from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT

SHA17 = "742ce68fafdeab6d50f59b4029c48f17bab63ab67cbde830d83a85dba6139abc"
SHA18 = "d1a39a94b9a4adb7998149d1ef7c9a90bfb8705ab80150fb3f63fa0d4fff7a2a"
GRID = PANEL[1]
PARTNER = np.array([ITEMS[i]["partner"] for i in GRID])
ARMS = {
    "skip1": (-1,),
    "skip3": (-3, -2, -1),
    "only_minus3": (-3,),
    "only_minus2": (-2,),
}
ACTIONS = ("wait", "ask", "help", "trade")


def histories(archives):
    """Return all stable lives, including terminal context, without using its feedback for learning."""
    paths = [archives / "source17.zip", archives / "source18.zip"]
    for path, sha in zip(paths, (SHA17, SHA18), strict=True):
        assert sha256(path.read_bytes()).hexdigest() == sha
    result = {}
    with ZipFile(paths[0]) as a, ZipFile(paths[1]) as b:
        ctx = json.loads(a.read("output/contexts.json"))
        assert np.array_equal(ctx["panel"][1], GRID)
        for seed in PILOT:
            src = list(
                map(
                    json.loads,
                    a.read(
                        f"output/lives/s{seed}/stable/qwen_learn/events.jsonl"
                    ).splitlines(),
                )
            )
            base = f"output/histories/s{seed}/stable/"
            ref = list(map(json.loads, b.read(base + "updates.jsonl").splitlines()))
            assert len(src) == len(ref) + 1 and src[-1]["after"] == 0
            for e, r in zip(src[:-1], ref, strict=True):
                assert all(
                    e[k] == r[k]
                    for k in (
                        "turn",
                        "health",
                        "row",
                        "after",
                        "action",
                        "signal",
                        "maintenance",
                    )
                )
                assert e["after"] > 0
            grid = np.load(
                BytesIO(b.read(base + "final-panel.npy")), allow_pickle=False
            )[3, 1].copy()
            birth = np.load(
                BytesIO(b.read(base + "birth-panel.npy")), allow_pickle=False
            )[1].copy()
            result[seed] = (src, ref, birth, grid)
    return result


def actual_labels(actual, birth):
    actual = np.asarray(actual)
    birth = np.asarray(birth)
    high = (actual.max(-1) >= 0.90) & (abs(actual - birth).sum(-1) / 2 >= 0.30)
    return np.where(high, actual.argmax(-1), -1)


def group_counts(grid, confidence):
    """Shape: state × 2 healthy sensor levels × 2 partners × 4 actions, each out of 16."""
    top = grid[:, 2:].argmax(axis=-1)
    ok = grid[:, 2:].max(axis=-1) >= confidence
    return np.stack(
        [
            np.stack(
                [
                    ((top[:, :, PARTNER == p] == a) & ok[:, :, PARTNER == p]).sum(-1)
                    for a in range(4)
                ],
                axis=-1,
            )
            for p in (2, 3)
        ],
        axis=2,
    )


def panel_labels(grid):
    counts = group_counts(grid, 0.90)
    eligible = (counts >= 12).all(axis=(1, 2))
    return np.where(eligible.any(-1), eligible.argmax(-1), -1)


def majority(grid):
    counts = group_counts(grid, 0.0).sum(axis=1)
    ok = (counts > 16).all(axis=1)
    return np.where(ok.any(-1), ok.argmax(-1), -1)


def starts(labels):
    """All qualifying triplet start positions as one-based request numbers."""
    labels = np.asarray(labels, dtype=int)
    ok = (
        (labels[:-2] >= 0) & (labels[:-2] == labels[1:-1]) & (labels[:-2] == labels[2:])
    )
    return {int(t + 1): int(labels[t]) for t in np.flatnonzero(ok)}


def transition_events(labels, family):
    result = []
    prev = None
    for t, a in starts(labels).items():
        if a != prev:
            result.append(
                dict(
                    family=family,
                    kind="onset" if prev is None else "rewrite",
                    turn=t,
                    action=a,
                    previous=prev,
                )
            )
            prev = a
    return result


def final_run(maj):
    a = int(maj[-1])
    if a < 0:
        return None
    t = len(maj)
    while t > 1 and maj[t - 2] == a:
        t -= 1
    return t, a


def event_plan(src, actual, birth, grid, reference_fraction):
    actual = actual_labels(actual[:-1], birth[:-1])
    result = transition_events(actual, "actual") + transition_events(
        panel_labels(grid), "panel"
    )
    terminal = final_run(majority(grid))
    if terminal:
        result.append(
            dict(
                family="final",
                kind="established",
                turn=terminal[0],
                action=terminal[1],
                previous=None,
            )
        )
    n = len(src)
    if not any(r["family"] == "actual" for r in result):
        target = round(1 + reference_fraction * (n - 1))
        age = [
            s
            for s in range(4, n - 1)
            if all(not s - 3 <= r["turn"] <= s + 2 for r in result)
        ]
        if age:
            t = min(age, key=lambda k: (abs(k - target), k))
            action = int(
                np.bincount(grid[-1, 2:].argmax(-1).ravel(), minlength=4).argmax()
            )
            result.append(
                dict(
                    family="no_onset",
                    kind="reference",
                    turn=t,
                    action=action,
                    previous=None,
                )
            )
    turns = {r["turn"] for r in result if r["family"] != "no_onset"}
    for j, r in enumerate(result):
        t = r["turn"]
        r["id"] = f"{r['family']}-{r['kind']}-{t}"
        r["control"] = None
        r["match_distance"] = None
        if t < 4 or r["family"] == "no_onset":
            continue
        candidates = [
            s
            for s in range(4, n - 1)
            if abs(s - t) >= 6 and not any(s - 3 <= k <= s + 2 for k in turns)
        ]
        if candidates:

            def distance(s):
                return abs(src[t - 4]["health"] - src[s - 4]["health"]) / 350 + abs(
                    t - s
                ) / (n - 1)

            s = min(candidates, key=lambda k: (distance(k), k))
            r["control"] = s
            r["match_distance"] = float(distance(s))
    return result


def reference_age(tapes):
    ages = []
    for src, ref, _, _ in tapes.values():
        p = np.asarray([e["probabilities"] for e in ref])
        first = transition_events(actual_labels(p[:, 3], p[:, 0]), "actual")
        if first:
            ages.append((first[0]["turn"] - 1) / (len(src) - 1))
    return float(np.median(ages))


def branches(events):
    tasks = defaultdict(list)
    for e in events:
        for role, t in (("event", e["turn"]), ("ordinary", e["control"])):
            if t is None or t < 4:
                continue
            for arm, offsets in ARMS.items():
                omissions = tuple(t + v for v in offsets)
                tasks[omissions].append(
                    dict(event=e["id"], window=role, arm=arm, anchor=t)
                )
    return dict(sorted(tasks.items()))


def branch_key(omissions):
    return "u" + "-".join(map(str, omissions))
