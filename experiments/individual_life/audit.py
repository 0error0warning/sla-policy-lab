"""Reconstruct every individual lineage from the published environment and parameter ledgers."""

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.design import (
    BIRTHS,
    CHECKS,
    PARAMS,
    rows,
    starting,
    tape,
)
from experiments.social_closed_loop.core import BodyValue


def read_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def checksum(bits):
    return sha256(bits.tobytes()).hexdigest()


def sampled(p, u):
    for i in range(3):
        if u <= sum(p[: i + 1]):
            return i
    return 3


def body_path(h, action, maintained, noise):
    # Remaining distance to 700 is multiplicative before clipping.
    path = [h]
    for c, eta in zip(
        ([0, 0.25, 1.75, 1.75][action], *([0 if maintained else 1] * 4)),
        noise,
        strict=True,
    ):
        path.append(
            max(0.0, 700.0 - (700.0 - path[-1]) * (1.0 + c * (1.0 + eta) / 350.0))
        )
    return path


def reconstruct(folder, meta, events, birth):
    changes = read_lines(folder / "updates.jsonl")
    rules = meta["rule"]
    expected_roles = ["main", "shadow"] if rules in ("true1", "true8") else ["main"]
    length = len(events)
    with np.load(folder / "parameters.npz") as z:
        bits = z["birth"]
        assert bits.shape == (PARAMS,) and bits.dtype == np.dtype("<u4")
        np.testing.assert_array_equal(bits, birth)
        assert checksum(bits) == meta["birth_sha256"]
        split = int(np.prod(z["A_shape"]))
        assert split + int(np.prod(z["B_shape"])) == PARAMS
        for role in expected_roles:
            actions = [u for u in changes if u["role"] == role]
            assert len(actions) == (0 if rules == "frozen" else length)
            curr = bits.copy()
            steps = z[role + "_xor"]
            assert len(steps) == len(actions)
            blocks = defaultdict(list)
            for idx, (u, step) in enumerate(zip(actions, steps, strict=True), 1):
                assert u["step"] == u["state_action_turn"] == idx
                t = events[idx - 1]
                release = (
                    idx if rules.endswith("1") else min(8 * ((idx + 7) // 8), length)
                )
                assert u["release_at"] == release and u["applied_at_terminal"] == (
                    release == length
                )
                assert 1 <= u["feedback_turn"] <= release
                blocks[release].append(u["feedback_turn"])
                source = events[u["feedback_turn"] - 1]
                if rules in ("true1", "wrong1") or (
                    role == "main" and rules == "true8"
                ):
                    assert u["feedback_turn"] == idx
                wrong_label = rules == "wrong1" or (
                    rules == "true1" and role == "shadow"
                )
                assert u["action_executed"] == t["action"]
                assert (
                    u["action_credited"]
                    == (t["action"] + (t["wrong_action_offset"] if wrong_label else 0))
                    % 4
                )
                assert u["applied_signal"] == source["centered_signal"]
                if wrong_label:
                    assert u["action_credited"] != t["action"]
                if release - 8 * ((idx - 1) // 8) > 1 and (
                    rules == "shuffle8" or (rules == "true8" and role == "shadow")
                ):
                    assert u["feedback_turn"] != idx
                assert (
                    u["behavior_probability"] == t["prob_current"][u["action_credited"]]
                )
                assert np.isclose(
                    u["importance_ratio"],
                    u["training_probability"] / u["behavior_probability"],
                    rtol=1e-12,
                )
                assert u["applied_ratio"] == min(2.0, max(0.5, u["importance_ratio"]))
                assert checksum(curr) == u["before"]
                nxt = curr ^ step
                assert checksum(nxt) == u["after"]
                assert u["changed_coordinates"] == np.count_nonzero(step)
                diff = nxt.view("<f4").astype(float) - curr.view("<f4")
                norms = [np.linalg.norm(diff[:split]), np.linalg.norm(diff[split:])]
                np.testing.assert_allclose(
                    norms, [u["A_change_l2"], u["B_change_l2"]], rtol=1e-8, atol=1e-11
                )
                assert np.isclose(
                    np.linalg.norm(norms), u["nominal_step_norm"], rtol=1e-4, atol=4e-6
                )
                assert np.linalg.norm(norms) <= 0.400004
                curr = nxt
            if rules in ("true8", "shuffle8"):
                start = 1
                for end, indices in blocks.items():
                    assert sorted(indices) == list(range(start, end + 1))
                    start = end + 1
            np.testing.assert_array_equal(curr, z[role + "_final"])
            if role == "main":
                assert checksum(curr) == meta["final_sha256"]
    main = [r for r in changes if r["role"] == "main"]
    assert len(changes) == sum(r["role"] in expected_roles for r in changes)
    return [meta["birth_sha256"]] + [u["after"] for u in main], len(changes)


def verify_folder(folder, tables, items, body, birth, paired):
    events = read_lines(folder / "life.jsonl")
    meta = json.loads((folder / "life.json").read_bytes())
    sense, seed, origin, rule = folder.parts[-4:]
    assert (meta["seed"], meta["birth"], meta["rule"], meta["visible"]) == (
        int(seed[1:]),
        origin,
        rule,
        sense == "visible",
    )
    assert (
        meta["turns"] == len(events)
        and meta["next_living_decisions"] == len(events) - 1
    )
    world_id = (meta["seed"] - (68011 if meta["seed"] < 68100 else 68101)) % 4
    assert meta["world"] == [world_id // 2, world_id % 2]
    hashes, updates = reconstruct(folder, meta, events, birth)
    assert (
        meta["physical_death"]
        and events[-1]["terminal"]
        and not any(e["terminal"] for e in events[:-1])
    )
    assert [t for t in range(len(events) + 1) if t < len(events) and t in CHECKS] == [
        t for t in CHECKS if "t" + str(t) in meta["living_probes"]
    ]
    low = next((t["turn"] for t in events[:-1] if t["health_after"] <= 10), None)
    assert (low is not None) == ("first_low" in meta["living_probes"])
    if low is not None:
        assert low == meta["living_probes"]["first_low"]["after_turn"]
    with np.load(folder / "living-probes.npz") as held:
        for k, marker in meta["living_probes"].items():
            j = marker["after_turn"]
            assert (
                j < len(events)
                and marker["actual_alive_health"] == events[j]["health_before"] > 0
            )
            assert marker["version"] == events[j]["before_learning_version"]
            for role in ["main", "shadow"] if rule in ("true1", "true8") else ["main"]:
                p = held[k + "-" + role]
                assert p.shape == (4, 96, 4) and np.isfinite(p).all()
                if sense == "hidden":
                    np.testing.assert_array_equal(p, np.broadcast_to(p[0], p.shape))
                if rule == "frozen":
                    np.testing.assert_array_equal(p, held["birth-main"])
    idx, uni, noise, offset = tables
    previous = 0
    version = 0
    h = 350.0
    rewards = -8.0
    max_err = 0.0
    noise_error = 0.0
    native = []
    native_changed = 0
    for k, t in enumerate(events):
        assert (
            t["turn"] == k + 1
            and t["context"] == idx[k]
            and t["health_before"] == h > 0
        )
        a = t["action"]
        r = items[idx[k]]
        j = r["partner"] - 2
        assert (
            t["partner"] == ("P86", "P93")[j]
            and t["frame"] == r["frame"]
            and t["request"] == r["text"]
            and t["layout"] == r["layout"]
        )
        assert (
            t["wrong_action_offset"] == offset[k]
            and t["action_uniform"] == uni[k, 0]
            and t["partner_uniform"] == uni[k, 1]
        )
        # Local and Colab inverse-normal implementations can differ by a few FP64 ulps.
        observed = tuple(t["body_noise"])
        exogenous = (
            t["context"],
            t["action_uniform"],
            t["partner_uniform"],
            observed,
            t["wrong_action_offset"],
        )
        assert paired.setdefault(k, exogenous) == exogenous
        noise_error = max(noise_error, float(np.max(np.abs(noise[k] - observed))))
        assert noise_error < 1e-13
        fav = 2 + (j ^ t["frame"] ^ meta["world"][j])
        p = 0.15 if a == 1 else 0.9 if a == fav else 0.1
        resp = None if a == 0 else t["partner_uniform"] < p
        assert t["maintenance"] is resp
        b = body_path(h, a, resp, noise[k])
        np.testing.assert_allclose(b, t["five_events"], atol=1e-10, rtol=0)
        assert t["five_events"][-1] == t["health_after"] < h
        assert t["terminal"] == (t["health_after"] == 0)
        h = t["health_after"]
        rew = body.rul(h) - body.rul(t["health_before"])
        assert (
            abs(rew - t["reference_change"]) < 1e-10
            and abs(t["personal_baseline_before"] - rewards / (k + 8)) < 1e-10
        )
        assert abs(t["centered_signal"] - (rew - t["personal_baseline_before"])) < 1e-10
        rewards += rew
        for name, act in (
            ("current", a),
            ("previous", t["counterfactual_previous_choice"]),
            ("birth", t["counterfactual_birth_choice"]),
        ):
            p = t["prob_" + name]
            assert (
                min(p) >= 0
                and abs(sum(p) - 1) < 3e-6
                and sampled(p, t["action_uniform"]) == act
            )
        assert (
            t["previous_decision_version"] == previous
            and t["before_learning_version"] == version
        )
        assert t["before_hash"] == hashes[version]
        if previous == version:
            max_err = max(
                max_err,
                float(
                    np.max(np.abs(np.asarray(t["prob_previous"]) - t["prob_current"]))
                ),
            )
        if version == 0:
            max_err = max(
                max_err,
                float(np.max(np.abs(np.asarray(t["prob_birth"]) - t["prob_current"]))),
            )
        if rule == "true1":
            assert t["deferred_version"] == (k // 8) * 8 and t["shadow_version"] == k
        expected = (
            0
            if rule == "frozen"
            else 1
            if rule.endswith("1")
            else k + 1 - version
            if (k + 1) % 8 == 0 or t["terminal"]
            else 0
        )
        assert t["updates_after_settlement"] == expected
        previous, version = version, version + expected
        assert (
            t["after_learning_version"] == version
            and t["after_hash"] == hashes[version]
        )
        if "same_state_native_suffix" in t:
            n = t["same_state_native_suffix"]
            native.append(float(np.max(np.abs(np.asarray(n) - t["prob_current"]))))
            native_changed += int(sampled(n, t["action_uniform"]) != a)
    assert h == 0 and version == meta["main_steps"] and max_err < 3e-5
    return len(events), updates, max_err, native, native_changed, noise_error


def run(root, source, out):
    items = rows()
    body = BodyValue()
    tapes = {}
    paired = defaultdict(dict)
    max_noise_error = 0.0
    births = {
        key: np.concatenate([x.ravel() for x in starting(source, key)]).view("<u4")
        for key in BIRTHS
    }
    counts = Counter()
    max_dup = 0.0
    errors = []
    flips = 0
    for f in sorted(root.glob("*/s*/*/*/life.json")):
        folder = f.parent
        seed = int(folder.parts[-3][1:])
        if seed not in tapes:
            tapes[seed] = tape(seed, items)
        n, u, d, p, z, noise_error = verify_folder(
            folder, tapes[seed], items, body, births[folder.parts[-2]], paired[seed]
        )
        counts.update(
            {
                "lives": 1,
                "death_settlements": 1,
                "living_successors": n - 1,
                "encounters": n,
                "main_and_shadow_microsteps": u,
            }
        )
        max_noise_error = max(max_noise_error, noise_error)
        max_dup = max(max_dup, d)
        errors += p
        flips += z
    assert counts["lives"] > 0
    result = {
        **counts,
        "independent_seeds": sorted(tapes),
        "birth_parameter_sha256": {k: checksum(v) for k, v in births.items()},
        "equal_snapshot_max_probability_error": max_dup,
        "max_local_noise_reconstruction_error": max_noise_error,
        "exact_exogenous_pairing_across_arms": True,
        "same_state_suffix_rows": len(errors),
        "same_state_suffix_max_abs": float(np.max(errors)) if errors else None,
        "same_state_suffix_q95_abs": float(np.quantile(errors, 0.95))
        if errors
        else None,
        "same_state_suffix_changed_sampled_choices": flips,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "source", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.source, a.output)
