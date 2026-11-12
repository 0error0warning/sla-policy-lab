"""Recompute the chronology from factual observations and audit the two frozen counterfactuals."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records
from experiments.first_fatal.audit import check_arrivals, checksum, path, sample
from experiments.first_fatal.design import HEALTHY, world
from experiments.health_baseline.analysis import read_replay
from experiments.individual_life.design import rows, starting
from experiments.social_closed_loop.core import BodyValue


def centers(history, health):
    g = (-8 + sum(r for _, r in history)) / (8 + len(history))
    close = [r for h, r in history if health / 1.25 <= h <= health * 1.25]
    return g, sum(close) / len(close) if len(close) >= 3 else g, len(close)


def score(update, signal, behavior, current):
    assert (
        update["behavior_probability"] == behavior
        and update["training_probability"] == current
    )
    ratio = current / behavior
    applied = max(0.5, min(2.0, ratio))
    assert update["importance_ratio"] == ratio and update["applied_ratio"] == applied
    norm = np.hypot(update["A_gradient_norm"], update["B_gradient_norm"])
    factor = 0.18 * signal * applied / (0.01 + norm * norm)
    size = abs(factor) * norm
    if size > 0.4:
        factor *= 0.4 / size
    assert (
        abs(factor - update["parameter_multiplier"]) < 1e-8
        and abs(min(size, 0.4) - update["nominal_step_norm"]) < 1e-9
    )
    if "actual_step_norm" in update:
        assert abs(update["actual_step_norm"] - min(size, 0.4)) < 1e-6
    return int(applied != ratio)


BASELINE_AUDIT = (("global", "global", 1.0), ("local", "local", 1.0))


def weight_at(weight, health):
    return weight(health) if callable(weight) else weight


def audit_replay(root, tapes, specification=BASELINE_AUDIT, expected_answers=680):
    (primary, first_ref, first_schedule), (secondary, second_ref, second_schedule) = (
        specification
    )
    count = Counter()
    max_norm_error = 0.0
    body = BodyValue() if specification != BASELINE_AUDIT else None
    with ZipFile(tapes) as archive:
        for f in sorted(root.glob("s*/*/life.json")):
            meta = json.loads(f.read_bytes())
            key = f"s{meta['seed']}/{meta['birth']}"
            source = [
                json.loads(x)
                for x in gzip.decompress(
                    archive.read(key + "/events.jsonl.gz")
                ).splitlines()
            ]
            published = json.loads(archive.read(key + "/life.json"))
            ev = read_replay(f.parent)
            assert len(source) == meta["turns"] == len(ev) == published["turns"]
            hashes = [published["snapshot_hashes"]["birth"]] * 2
            history = []
            for j, (s, e) in enumerate(zip(source, ev, strict=True)):
                assert all(
                    s[k] == e[k]
                    for k in (
                        "turn",
                        "index",
                        "health",
                        "after",
                        "action",
                        "practice",
                        "reward",
                    )
                )
                assert s["u"][0] == e["u"] and e["before_hash"] == hashes
                p = np.asarray(e["probs"])
                assert p.shape == (5, 4) and np.abs(p.sum(-1) - 1).max() < 4e-6
                if j == 0:
                    assert np.all(p == p[0])
                g, l, n = centers(history, s["health"])
                assert (
                    abs(g - e["global_baseline"]) < 1e-9
                    and abs(l - e["local_baseline"]) < 1e-9
                    and n == e["eligible"]
                )
                assert e[f"source_{primary}_hash_match"] == (
                    e["updates"][0]["after_hash"] == s["after_hash"][0]
                )
                assert (
                    abs(
                        e[f"source_{primary}_max_error"]
                        - np.abs(p[1] - s["probs"][1]).max()
                    )
                    < 1e-12
                )
                a = s["action"]
                ref = {"global": g, "local": l}
                raw = s["reward"]
                first_weight = weight_at(first_schedule, s["health"])
                second_weight = weight_at(second_schedule, s["health"])
                assert abs(s["baseline"] - ref[first_ref]) < 1e-9
                signals = (
                    first_weight * raw - ref[first_ref],
                    second_weight * raw - ref[second_ref],
                )
                if specification != BASELINE_AUDIT:
                    np.testing.assert_allclose(e["signals"], signals, rtol=0, atol=1e-9)
                    np.testing.assert_array_equal(
                        e["weighted_feedback"],
                        [first_weight * raw, second_weight * raw],
                    )
                    expected_delta = (
                        (second_weight - first_weight) * raw
                        + ref[first_ref]
                        - ref[second_ref]
                    )
                    assert (
                        abs((signals[1] - signals[0]) - expected_delta) < 1e-9
                        and raw <= 1e-10
                    )
                    assert (
                        abs((body.rul(e["after"]) - body.rul(e["health"])) - raw) < 1e-9
                    )
                for k, (rec, idx, signal) in enumerate(
                    zip(
                        e["updates"],
                        (1, 3, 1),
                        (signals[0], signals[1], signals[1]),
                        strict=True,
                    )
                ):
                    count["propensity_clips"] += score(
                        rec, signal, s["probs"][1][a], p[idx, a]
                    )
                    max_norm_error = max(
                        max_norm_error,
                        abs(rec["actual_step_norm"] - rec["nominal_step_norm"]),
                    )
                    if k == 2:
                        assert all(
                            e["updates"][0][x] == rec[x]
                            for x in (
                                "A_gradient_norm",
                                "B_gradient_norm",
                                "importance_ratio",
                            )
                        )
                hashes = [v["after_hash"] for v in e["updates"][:2]]
                history.append((s["health"], s["reward"]))
                assert (s["after"] == 0) == (j == len(source) - 1)
                if j:
                    count["living_successors"] += 1
                    count["practice_answer_successors"] += int(
                        source[j - 1]["practice"] and source[j - 1]["action"] >= 2
                    )
            w = meta["witness"]
            j = w["turn"] - 1
            assert j == next(
                k + 1
                for k, x in enumerate(ev[:-1])
                if x["practice"] and x["action"] >= 2
            )
            with np.load(f.parent / "successor.npz") as z:
                assert all(checksum(z[n]) == v for n, v in w["hashes"].items())
                np.testing.assert_array_equal(z["probs"][:5], ev[j]["probs"])
                np.testing.assert_array_equal(z["probs"][1], z["probs"][5])
                prev = ev[j - 1]
                for n, h in zip(
                    (
                        primary + "_before",
                        secondary + "_before",
                        primary,
                        secondary,
                        secondary + "_from_" + primary,
                    ),
                    prev["before_hash"] + [u["after_hash"] for u in prev["updates"]],
                    strict=True,
                ):
                    assert checksum(z[n]) == h
            count["tapes"] += 1
            count["turns"] += len(ev)
    assert (
        count["tapes"] == 48 and count["practice_answer_successors"] == expected_answers
    )
    return {**count, "max_actual_minus_nominal_step": max_norm_error}


def audit_lives(roots, source, specification=BASELINE_AUDIT, shadow_modes=None):
    names = [v[0] for v in specification]
    config = {name: (ref, k) for name, ref, k in specification}
    items = rows()
    body = BodyValue()
    starts = {
        b: checksum(
            np.concatenate([v.ravel() for v in starting(source, b)]).astype("<f4")
        )
        for b in ("blank", "population")
    }
    paired = {}
    count = Counter()
    max_path_error = 0.0
    seen = set()
    cutoff = 700 * 1.75 / (350 + 1.75)
    for f in sorted(p for root in roots for p in root.glob("s*/*/*/life.json")):
        m = json.loads(f.read_bytes())
        ev = records(f.parent)
        seed, birth, rule = m["seed"], m["birth"], m["rule"]
        key = seed, birth, rule
        assert key not in seen
        seen.add(key)
        assert list(f.parent.parts[-3:]) == [f"s{seed}", birth, rule] and m[
            "world"
        ] == list(world(seed))
        yes = rule.endswith("_true")
        frozen = rule == "frozen"
        mode = names[0] if frozen else rule.split("_")[0]
        other = names[0] if mode == names[1] else names[1]
        other, extra = (shadow_modes or {}).get(rule, (other, None))
        hashes = [starts[birth]] * (1 + 2 * int(yes) + int(extra is not None))
        with np.load(f.parent / "parameters.npz") as z:
            assert all(checksum(z[k]) == m["snapshot_hashes"][k] for k in z.files)
            assert checksum(z["birth"]) == starts[birth]
        h = 350.0
        history = []
        entry = first = None
        for t, e in enumerate(ev, 1):
            a = e["action"]
            p = np.asarray(e["probs"])
            assert (
                e["turn"] == t and e["health"] == h > 0 and e["before_hash"] == hashes
            )
            assert e["version_at_choice"] == (0 if frozen else t - 1)
            assert (
                p.shape == (5, 4)
                and np.abs(p.sum(-1) - 1).max() < 4e-6
                and sample(p[1], e["u"][0]) == a
            )
            row = items[e["index"]]
            assert (row["partner"], row["frame"], row["layout"]) == (
                e["partner"],
                e["frame"],
                e["layout"],
            )
            value = (e["index"], tuple(e["u"]), tuple(e["noise"]), e["offset"])
            assert paired.setdefault((seed, t), value) == value
            favored = 2 + (
                (row["partner"] - 2) ^ row["frame"] ^ m["world"][row["partner"] - 2]
            )
            rate = 0.15 if a == 1 else 0.9 if a == favored else 0.1
            accepted = None if a == 0 else e["u"][1] < rate
            assert accepted is e["maintenance"]
            vals, peak = path(h, a, accepted, e["noise"])
            err = float(np.abs(np.asarray(vals) - e["subevents"]).max())
            max_path_error = max(max_path_error, err)
            assert (
                err < 1e-10
                and e["subevents"][-1] == e["after"] < h
                and e["peak_collapse"] == peak
            )
            assert e["practice"] == (cutoff < h <= 6) and e["first_fatal"] == (
                h <= cutoff and first is None
            )
            if e["practice"] and entry is None:
                entry = t
                assert m["snapshot_hashes"]["entry"] == hashes[0]
            if e["first_fatal"]:
                first = t
                assert hashes[0] == m["snapshot_hashes"]["first"]
                if yes:
                    assert hashes[1:] == [
                        m["snapshot_hashes"][k]
                        for k in (
                            ("first_wrong", "first_other", "first_extra")
                            if extra is not None
                            else ("first_wrong", "first_other")
                        )
                    ]
                assert (e["after"] == 0) == (a >= 2)
                assert all(path(h, b, True, [-0.3] * 5)[0][-1] == 0 for b in (2, 3))
                assert all(path(h, b, False, [0.3] * 5)[0][-1] > 0 for b in (0, 1))
                with np.load(f.parent / "predecision.npz") as z:
                    np.testing.assert_array_equal(z["health"], [h, *HEALTHY])
                    np.testing.assert_array_equal(z["same_request"][0, :5], p)
                    assert z["panel"].shape == (3, 96, 6, 4)
                    np.testing.assert_array_equal(
                        z["panel"][:, :, 1], z["panel"][:, :, 5]
                    )
                    assert (
                        len(
                            {
                                (
                                    items[i]["partner"],
                                    items[i]["frame"],
                                    items[i]["layout"],
                                )
                                for i in z["indices"]
                            }
                        )
                        == 96
                    )
                    assert all(
                        (items[i]["task"], items[i]["wording"]) == (5, 4)
                        for i in z["indices"]
                    )
            g, l, n = centers(history, h)
            reward = body.rul(e["after"]) - body.rul(h)
            assert (
                n == e["eligible"]
                and abs(g - e["global_baseline"]) < 1e-9
                and abs(l - e["local_baseline"]) < 1e-9
            )
            assert (
                abs(reward - e["reward"]) < 1e-9
                and abs(e["baseline"] - (l if config[mode][0] == "local" else g)) < 1e-9
            )
            coefficient = weight_at(config[mode][1], h)
            assert abs(e["signal"] - coefficient * reward + e["baseline"]) < 1e-9
            if specification != BASELINE_AUDIT:
                assert (
                    abs(e["weighted_feedback"] - reward * coefficient) < 1e-9
                    and reward <= 1e-10
                )
            wrong = (a + e["offset"]) % 4
            labels = (
                []
                if frozen
                else [(a, mode), (wrong, mode), (a, other)]
                if yes
                else [(wrong, mode)]
            )
            if extra is not None:
                labels.append((a, extra))
            for j, ((label, center), rec) in enumerate(
                zip(labels, e["update"], strict=True), 1
            ):
                reference, schedule = config[center]
                weight = weight_at(schedule, h)
                signal = weight * reward - (g if reference == "global" else l)
                if specification != BASELINE_AUDIT:
                    assert (
                        rec["damage_weight"] == weight
                        and abs(rec["weighted_feedback"] - reward * weight) < 1e-9
                    )
                assert (
                    rec["credited_action"] == label
                    and rec["criterion"] == center
                    and abs(rec["signal"] - signal) < 1e-9
                )
                count["clipped_propensities"] += score(
                    rec, signal, p[1, label], p[j, label]
                )
            hashes = e["after_hash"]
            if frozen:
                assert hashes == [starts[birth]] and np.array_equal(p[0], p[1])
            history.append((h, reward))
            h = e["after"]
            count["microsteps_with_shadows"] += len(labels)
            assert (h == 0) == (t == len(ev))
        assert (
            first == m["first_fatal"]
            and entry == m["practice_entry"]
            and m["main_updates"] == (0 if frozen else len(ev))
        )
        assert (
            m["practice_count"] == sum(e["practice"] for e in ev)
            and hashes[0] == m["snapshot_hashes"]["final"]
        )
        assert m["natural_death"] and len(ev) == m["turns"]
        count["natural_deaths"] += 1
        count["first_encounters"] += int(first is not None)
        count["lives"] += 1
        count["turns"] += len(ev)
    assert seen
    return {
        **count,
        "independent_seeds": len({s for s, _, _ in seen}),
        "birth_hashes": starts,
        "max_physical_subevent_error": max_path_error,
        "independently_rebuilt_arrivals": len(paired),
        "max_arrival_noise_error": check_arrivals(paired, items),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("replay", "tapes", "source", "output"):
        p.add_argument("--" + k, type=Path, required=k == "output")
    p.add_argument("--lives", nargs="+", type=Path)
    a = p.parse_args()
    result = {}
    if a.replay:
        result["replay"] = audit_replay(a.replay, a.tapes)
    if a.lives:
        result["live"] = audit_lives(a.lives, a.source)
    write(a.output, result)
