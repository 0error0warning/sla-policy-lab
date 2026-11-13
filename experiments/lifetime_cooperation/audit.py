"""Integrity and independent physical/behavioral replay of a completed pilot or confirmation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.individual_life.design import action_at
from experiments.social_closed_loop.core import BodyValue
from .analysis import METRICS, analyze
from .design import ITEMS, LAYOUTS, PANEL, WAVES, schedule
from .simple import LinearGroup


def digest(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def near(a, b, tol=1e-9):
    np.testing.assert_allclose(a, b, rtol=0.0, atol=tol)


def audit_archive(path):
    with ZipFile(path) as z:
        manifest = json.loads(z.read("archive-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"archive-manifest.json"}
        for name, info in manifest.items():
            payload = z.read(name)
            assert (
                len(payload) == info["bytes"]
                and sha256(payload).hexdigest() == info["sha256"]
            )


def physical(h, action, accepted, noise):
    """Independent five-event calculation; use each event's newly reached health."""
    path = [h]
    for cost, eps in zip(
        ([0.0, 0.25, 1.75, 1.75][action], *([0.0 if accepted else 1.0] * 4)),
        noise,
        strict=True,
    ):
        path.append(
            max(0.0, path[-1] - cost * (1.0 + (1.0 - path[-1] / 350.0)) * (1.0 + eps))
        )
    return path


def check_word_pairs():
    for a, b in PANEL.T:
        r, t = ITEMS[a], ITEMS[b]
        assert all(r[k] == t[k] for k in ("task", "partner", "layout", "frame"))
        assert r["wording"] == 3 and t["wording"] == 4 and r["text"] != t["text"]
        assert (
            "seven minus two" if r["task"] == 4 else "the first month of the year"
        ) in r["text"]
        assert (
            "seven minus two" if t["task"] == 4 else "the first month of the year"
        ) in t["text"]
    assert len(set(LAYOUTS)) == 4


def verify(root, analysis, archive=None):
    if archive is not None:
        audit_archive(archive)
    receipt = read(root / "receipt.json")
    gate = analyze(root, analysis)
    seeds = WAVES[receipt["mode"]]
    assert (
        gate["seeds"] == list(seeds)
        and receipt["backbone_frozen"]
        and receipt["full_physical_decisions"]
    )
    ctx = read(root / "contexts.json")
    assert np.array_equal(ctx["panel"], PANEL) and ctx["rows"] == [
        {k: v for k, v in r.items() if k != "split"} for r in ITEMS
    ]
    check_word_pairs()
    body = BodyValue()
    nturn = nupdates = physical_qwen = checked_hash = checked_grids = 0
    stable_prefix = {}
    independent = {}
    healthy_actual = {}
    count = 0
    for seed in seeds:
        ix, u, eta, off = schedule(seed)
        base = seeds[0]
        for world in ("stable", "changed"):
            for family in ("qwen", "simple"):
                for arm in ("learn", "frozen"):
                    d = root / "lives" / f"s{seed}" / world / (family + "_" + arm)
                    m = read(d / "life.json")
                    events = [
                        json.loads(v)
                        for v in (d / "events.jsonl").read_bytes().splitlines()
                    ]
                    assert (
                        m["seed"] == seed
                        and m["learning"] == (arm == "learn")
                        and m["kind"] == family
                    )
                    assert (
                        m["world"] == world
                        and len(events) == m["turns"]
                        and m["physical_death"]
                    )
                    prefix = events[:24]
                    if world == "stable":
                        stable_prefix[seed, family, arm] = prefix
                    else:
                        assert prefix == stable_prefix[seed, family, arm]
                    nturn += len(events)
                    physical_qwen += len(events) * (family == "qwen")
                    count += 1
                    h, total = 350.0, 0.0
                    low = rescued = healthy = tasks = injured = all_tasks = 0
                    replay = (
                        LinearGroup() if arm == "learn" or family == "simple" else None
                    )
                    last_birth = None
                    points = {p["completed"]: p for p in m["checkpoints"]}
                    if arm == "learn":
                        assert [
                            p["completed"]
                            for p in points.values()
                            if "last_living" in p["tags"]
                        ] == [len(events) - 1]
                        assert 0 in points and len(events) - 1 in points
                    for k, e in enumerate(events):
                        j = int(ix[k])
                        r = ITEMS[j]
                        assert (
                            r["wording"] == 3 and j == e["row"] and e["turn"] == k + 1
                        )
                        bits = [
                            ((seed - base) % 4) // 2
                            ^ int(world == "changed" and k >= 24),
                            (seed - base) % 2,
                        ]
                        assert bits == e["rules"]
                        # SciPy inverse-normal implementations differ in their last float64 bits.
                        near(eta[k], e["noise"], 5e-15)
                        assert (
                            e["action_u"] == u[k, 0]
                            and e["partner_u"] == u[k, 1]
                            and e["health"] == h
                        )
                        probs = np.array(e["prob"])
                        near(probs.sum(axis=1), np.ones(3), 3e-6)
                        a = e["action"]
                        assert (
                            a == action_at(probs[0], u[k, 0])
                            and e["wrong_action"] == (a + off[k]) % 4
                            and a != e["wrong_action"]
                        )
                        partner = r["partner"] - 2
                        favored = 2 + (partner ^ r["frame"] ^ bits[partner])
                        likelihood = (
                            0.0
                            if a == 0
                            else 0.15
                            if a == 1
                            else (0.9 if a == favored else 0.1)
                        )
                        ok = bool(u[k, 1] < likelihood) if a else None
                        assert ok == e["maintenance"]
                        stages = physical(h, a, ok, e["noise"])
                        near(stages, e["stages"], 1e-10)
                        after = e["after"]
                        near(stages[-1], after, 1e-10)
                        assert (after > 0) == (k < len(events) - 1) and h > after
                        rescue = bool(
                            a >= 2
                            and ok
                            and after > 0
                            and physical(h, a, False, e["noise"])[-1] == 0.0
                        )
                        assert rescue == e["rescued_answer"]
                        raw = body.rul(after) - body.rul(h)
                        baseval = (total - 8) / (8 + k)
                        near(
                            [raw, baseval, raw - baseval],
                            [e["reward"], e["baseline"], e["signal"]],
                        )
                        if arm == "frozen" or k == 0:
                            near(probs, [probs[1]] * 3, 1e-7)
                        if replay:
                            linear_probs = replay.decision(j, h)
                            measured = (
                                probs
                                if family == "simple"
                                else e["linear_on_tape_prob"]
                            )
                            near(linear_probs, measured, 1e-9)
                        if k in points:
                            pt = points[k]
                            assert pt["health"] == h and pt["bits"] == bits
                            if k:
                                assert (
                                    pt["parameters"]
                                    == events[k - 1]["after_parameters"]
                                )
                            if k == 0 or k == len(events) - 1:
                                with np.load(d / "parameters.npz") as snap:
                                    for role, signature in pt["parameters"].items():
                                        assert (
                                            digest(snap[pt["key"] + "_" + role])
                                            == signature
                                        )
                                        checked_hash += 1
                            with np.load(d / "panels.npz") as z:
                                panel = z[pt["key"]]
                                reference = (
                                    z[pt["key"] + "_on_tape"]
                                    if family == "qwen"
                                    else panel
                                )
                                near(reference, replay.probe(), 2e-9)
                                if "birth" in pt["tags"]:
                                    near(panel[0], panel[1], 1e-7)
                                    near(panel[0], panel[2], 1e-7)
                                    last_birth = panel[1].copy()
                                near(panel[1], last_birth, 1e-7)
                                checked_grids += 1
                                if (
                                    family == "qwen"
                                    and world == "stable"
                                    and "last_living" in pt["tags"]
                                ):
                                    for wi, title in enumerate(("familiar", "held")):
                                        # At H=2 precisely the maintained refuse-and-appeal action saves a life.
                                        # At H=6 the two answer forms only survive given maintenance.
                                        fav = np.array(
                                            [
                                                2
                                                + (
                                                    (ITEMS[ii]["partner"] - 2)
                                                    ^ ITEMS[ii]["frame"]
                                                    ^ bits[ITEMS[ii]["partner"] - 2]
                                                )
                                                for ii in PANEL[wi]
                                            ]
                                        )
                                        idx = np.arange(len(fav))
                                        for slot, role in enumerate(
                                            ("current", "birth", "wrong")
                                        ):
                                            g = panel[slot, wi].astype(float)
                                            rescue_p = np.mean(
                                                0.9 * g[1, idx, fav]
                                                + 0.1 * g[1, idx, 5 - fav]
                                            )
                                            death = np.mean(
                                                g[0, :, 0]
                                                + 0.85 * g[0, :, 1]
                                                + g[0, :, 2]
                                                + g[0, :, 3]
                                            )
                                            healthy_p = np.mean(
                                                g[2:, :, 2] + g[2:, :, 3]
                                            )
                                            independent[seed, title, role] = (
                                                rescue_p,
                                                death,
                                                healthy_p,
                                            )
                        if arm == "learn":
                            nupdates += 1
                            cost = replay.learn(
                                j, h, a, e["wrong_action"], e["signal"], probs[0]
                            )
                            near(
                                cost,
                                e["steps"]
                                if family == "simple"
                                else e["linear_on_tape_steps"],
                                1e-9,
                            )
                            if family == "qwen":
                                for slot, step in zip((0, 2), e["steps"], strict=True):
                                    label = a if slot == 0 else e["wrong_action"]
                                    near(
                                        step["training_probability"], probs[slot, label]
                                    )
                                    near(step["behavior_probability"], probs[0, label])
                                    assert 0 <= step["nominal_step_norm"] <= 0.400001
                        injured += int(after > 0)
                        low += int(h <= 10 and after > 0)
                        healthy += int(h >= 90)
                        tasks += int(h >= 90 and a >= 2 and after > 0)
                        all_tasks += int(a >= 2 and after > 0)
                        rescued += int(rescue)
                        total += raw
                        h = after
                    assert h == 0
                    near(total, -body.rul(350))
                    assert (injured, low, rescued, healthy, tasks, all_tasks) == tuple(
                        m[k]
                        for k in (
                            "injured_then_acted",
                            "low_then_acted",
                            "rescued",
                            "healthy_rounds",
                            "healthy_tasks",
                            "tasks",
                        )
                    )
                    if world == "stable" and family == "qwen":
                        healthy_actual[seed, arm] = tasks / healthy
    for k, metric in enumerate(METRICS):
        for word in ("familiar", "held"):
            for ref in ("birth", "wrong"):
                diffs = [
                    100
                    * (
                        independent[s, word, "current"][k]
                        - independent[s, word, ref][k]
                    )
                    for s in seeds
                ]
                near(
                    np.mean(diffs),
                    gate["paired_pp"][f"{word}.{metric}.{ref}"]["mean"],
                    2e-9,
                )
    near(
        np.mean(
            [
                100 * (healthy_actual[s, "learn"] - healthy_actual[s, "frozen"])
                for s in seeds
            ]
        ),
        gate["paired_pp"]["actual_healthy_vs_frozen"]["mean"],
    )
    assert (
        count == len(seeds) * 8 and receipt["counts"]["full_forwards"] >= physical_qwen
    )
    result = dict(
        physical_lives=count,
        physical_decisions=nturn,
        physical_Qwen_decisions=physical_qwen,
        learning_events=nupdates,
        checkpoint_primary_hashes_verified=checked_hash,
        checkpoint_panels_verified=checked_grids,
        matched_wording_pairs=len(PANEL[0]),
        replayed_gates=13,
        replayed_world_prefix_turns=24,
        passed=True,
    )
    (analysis / "audit.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.root, args.out, args.archive), indent=2))
