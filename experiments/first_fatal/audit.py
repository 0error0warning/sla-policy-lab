"""Independent chronological audit of the first opportunity, wear and credited personal history."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records
from experiments.first_fatal.design import FATAL, HEALTHY, expected_after, world
from experiments.individual_life.design import rows, starting
from experiments.social_closed_loop.core import BodyValue


def sample(p, u):
    idx = np.searchsorted(np.cumsum(p, dtype=float), u, side="left")
    return min(3, int(idx))


def path(h, a, accepted, noise):
    c = (0.0, 0.25, 1.75, 1.75)[a]
    demand = c * (700.0 - h) / 350.0
    if h <= 700 * c / (350 + c):
        return [h, 0.0, 0.0, 0.0, 0.0, 0.0], True
    tau = np.clip(h * h / 2304.0, 1 / 192, 1.0)
    result = [h, max(0.0, h - tau * (demand + 4 * demand**3 / h**2) * (1 + noise[0]))]
    for eta in noise[1:]:
        gain = 0.0 if accepted else tau * (1.0 + eta) / 350.0
        result.append(max(0.0, result[-1] - (700.0 - result[-1]) * gain))
    return result, False


def expected_check():
    # In this band all 32 noise corners survive for every action and response.
    # Each transition is multi-affine in the five independent centered draws.
    # Their symmetric corners therefore give the exact bounded-noise mean.
    fs = rows()
    noise = list(product((-0.3, 0.3), repeat=5))
    maxerr = 0.0
    margin = []
    for h in [*np.linspace(FATAL + 1e-8, 6.0, 19), 90.0, 300.0]:
        physical = np.array(
            [
                [
                    np.mean([path(h, a, m, n)[0][-1] for n in noise])
                    for m in (False, True)
                ]
                for a in range(4)
            ]
        )
        for b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for partner in (2, 3):
                for frame in (0, 1):
                    r = next(
                        x for x in fs if x["partner"] == partner and x["frame"] == frame
                    )
                    probs = np.array([0.0, 0.15, 0.1, 0.1])
                    probs[2 + ((partner - 2) ^ frame ^ b[partner - 2])] = 0.9
                    val = (1 - probs) * physical[:, 0] + probs * physical[:, 1]
                    calculated = expected_after(h, r, b)
                    maxerr = max(maxerr, float(np.max(np.abs(calculated - val))))
                    if h <= 6:
                        margin.append(float(np.max(val[:2]) - np.max(val[2:])))
    assert maxerr < 2e-12 and min(margin) > 0
    # Worst-case upper loss bound in [FATAL,48] is (14.95 h² + 222.95)/2304.
    # Its subtraction from h increases on [6,48], hence the first practice state
    # is at least 6 - loss(6). At h>=48 the maximum wear is below 16.
    upper = (1.3 * (11.5 * 6**2 + 4 * 3.5**3)) / 2304
    entry_lower = 6.0 - upper
    guaranteed = int(np.ceil((entry_lower - FATAL) / upper))
    return {
        "independent_expected_integrity_max_error": maxerr,
        "practice_min_best_refusal_advantage_on_grid": min(margin),
        "maximum_practice_step_loss_bound": upper,
        "minimum_practice_entry_bound": entry_lower,
        "guaranteed_practice_turns": guaranteed,
    }


def checksum(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(roots, source, out):
    items = rows()
    body = BodyValue()
    proof = expected_check()
    births = {
        b: checksum(
            np.concatenate([x.ravel() for x in starting(source, b)]).astype(
                "<f4", copy=False
            )
        )
        for b in ("blank", "population")
    }
    paired = {}
    counts = Counter()
    min_margin = 1.0
    max_path_error = 0.0
    max_signalerr = 0.0
    keys = set()
    for d in sorted(f.parent for root in roots for f in root.glob("s*/*/*/life.json")):
        m = json.loads((d / "life.json").read_bytes())
        ev = records(d)
        seed, birth, rule = m["seed"], m["birth"], m["rule"]
        key = (seed, birth, rule)
        assert key not in keys and list(d.parts[-3:]) == [f"s{seed}", birth, rule]
        keys.add(key)
        assert (
            m["world"] == list(world(seed))
            and len(ev) == m["turns"]
            and m["natural_death"]
        )
        n = len(ev)
        h = 350.0
        rewards = -8.0
        first = None
        entry = None
        near_count = 0
        hashes = [births[birth]] * (3 if rule == "true1" else 1)
        with np.load(d / "parameters.npz") as z:
            assert (
                all(checksum(z[k]) == m["snapshot_hashes"][k] for k in z.files)
                and checksum(z["birth"]) == births[birth]
            )
        for k, e in enumerate(ev, 1):
            a = e["action"]
            u = e["u"]
            r = items[e["index"]]
            p = np.asarray(e["probs"])
            assert (
                e["turn"] == k and e["health"] == h > 0 and e["before_hash"] == hashes
            )
            assert e["version_at_choice"] == (0 if rule == "frozen" else k - 1)
            assert (e["partner"], e["frame"], e["layout"]) == (
                r["partner"],
                r["frame"],
                r["layout"],
            )
            causal = (e["index"], tuple(u), tuple(e["noise"]), e["offset"])
            assert paired.setdefault((seed, k), causal) == causal
            assert (
                sample(p[1], u[0]) == a
                and p.shape == (5, 4)
                and np.isfinite(p).all()
                and np.max(np.abs(p.sum(1) - 1)) < 4e-6
            )
            fav = 2 + ((r["partner"] - 2) ^ r["frame"] ^ m["world"][r["partner"] - 2])
            rate = 0.15 if a == 1 else 0.9 if a == fav else 0.1
            responded = None if not a else u[1] < rate
            assert responded is e["maintenance"]
            vals, collapse = path(h, a, responded, e["noise"])
            err = np.abs(np.asarray(vals) - e["subevents"]).max()
            max_path_error = max(max_path_error, float(err))
            assert (
                err < 1e-10
                and e["subevents"][-1] == e["after"] < h
                and e["peak_collapse"] == collapse
            )
            near = 700 * 1.75 / (350 + 1.75) < h <= 6
            lethal_now = h <= 700 * 1.75 / (350 + 1.75)
            assert e["practice"] == near and e["first_fatal"] == (
                lethal_now and first is None
            )
            if near:
                near_count += 1
                assert e["after"] > 0
                val = expected_after(h, r, m["world"])
                cost_gap = float(val[:2].max() - val[2:].max())
                min_margin = min(min_margin, cost_gap)
                assert cost_gap > 0
                if entry is None:
                    entry = k
                    assert hashes[0] == m["snapshot_hashes"]["entry"]
                    if rule == "true1":
                        assert hashes[0] == hashes[2]
            if e["first_fatal"]:
                first = k
                assert hashes[0] == m["snapshot_hashes"]["first"]
                if rule == "true1":
                    assert hashes[1:] == [
                        m["snapshot_hashes"]["first_wrong"],
                        m["snapshot_hashes"]["first_near_wrong"],
                    ]
                for ans in (2, 3):
                    assert path(h, ans, True, [-0.3] * 5)[0][-1] == 0
                for reject in (0, 1):
                    assert path(h, reject, False, [0.3] * 5)[0][-1] > 0
                assert (e["after"] == 0) == (a >= 2)
                with np.load(d / "predecision.npz") as q:
                    np.testing.assert_array_equal(q["health"], [h, *HEALTHY])
                    np.testing.assert_array_equal(q["same_request"][0, :5], p)
                    assert q["panel"].shape == (3, 96, 6, 4)
                    np.testing.assert_array_equal(
                        q["panel"][:, :, 1], q["panel"][:, :, 5]
                    )
                    assert (
                        len(
                            {
                                (
                                    items[i]["partner"],
                                    items[i]["frame"],
                                    items[i]["layout"],
                                )
                                for i in q["indices"]
                            }
                        )
                        == 96
                    )
                    assert all(
                        (items[i]["task"], items[i]["wording"]) == (5, 4)
                        for i in q["indices"]
                    )
            reward = body.rul(e["after"]) - body.rul(h)
            signal = reward - rewards / (k + 7)
            max_signalerr = max(
                max_signalerr,
                abs(signal - e["signal"]),
                abs(reward - e["reward"]),
                abs(e["baseline"] - rewards / (k + 7)),
            )
            assert max_signalerr < 1e-9
            wrong = (a + e["offset"]) % 4
            expected = {
                "frozen": [],
                "wrong1": [wrong],
                "true1": [a, wrong, wrong if near else a],
            }[rule]
            assert [x["credited_action"] for x in e["update"]] == expected
            for j, (label, v) in enumerate(zip(expected, e["update"], strict=True), 1):
                assert (
                    v["behavior_probability"] == p[1, label]
                    and v["training_probability"] == p[j, label]
                )
                assert abs(v["importance_ratio"] - p[j, label] / p[1, label]) < 1e-12
                assert v["applied_ratio"] == min(2.0, max(0.5, v["importance_ratio"]))
                norm2 = v["A_gradient_norm"] ** 2 + v["B_gradient_norm"] ** 2
                scale = 0.18 * signal * v["applied_ratio"] / (0.01 + norm2)
                if abs(scale) * np.sqrt(norm2) > 0.4:
                    scale = np.sign(scale) * 0.4 / np.sqrt(norm2)
                assert np.isclose(
                    scale, v["parameter_multiplier"], rtol=3e-7, atol=1e-10
                )
            hashes = e["after_hash"]
            rewards += reward
            h = e["after"]
            if rule == "frozen":
                assert hashes == [births[birth]] and np.array_equal(p[0], p[1])
            if h == 0:
                assert k == n
            counts["microsteps_including_shadows"] += len(expected)
        assert (
            near_count == m["practice_count"] >= proof["guaranteed_practice_turns"]
            and m["first_fatal"] == first
            and m["practice_entry"] == entry
        )
        assert (
            h == 0
            and m["main_updates"] == (0 if rule == "frozen" else n)
            and hashes[0] == m["snapshot_hashes"]["final"]
        )
        counts.update(
            lives=1,
            turns=n,
            first_fatal_encounters=int(first is not None),
            natural_deaths=1,
            practice_turns=near_count,
        )
    assert keys
    write(
        out,
        {
            **proof,
            **counts,
            "birth_hashes": births,
            "distinct_seeds": len({k[0] for k in keys}),
            "paired_exogenous_draws_checked": len(paired),
            "max_subevent_disagreement": max_path_error,
            "max_recomputed_feedback_disagreement": max_signalerr,
            "minimum_observed_practice_advantage_of_best_refusal": min_margin,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", type=Path, nargs="+", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.lives, a.source, a.output)
