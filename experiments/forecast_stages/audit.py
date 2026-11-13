"""Independent CPU verification of chronology, predictor inputs and counterfactual choices."""

import argparse
import csv
import gzip
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import records
from experiments.individual_life.design import action_at, offers
from experiments.lifetime_cooperation.audit import physical
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT, rules
from experiments.loreft_omission.audit import manifest
from experiments.partner_prediction.design import FEATURE_IDS, features
from experiments.social_closed_loop.core import BodyValue

from .analysis import analyze
from .design import (
    DENSE,
    HEALTH,
    PARENT_RUNS,
    STAGES,
    VIEWS,
    prepare,
    random_panel,
    turns,
)


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, atol=tol, rtol=0)


def archived(inputs, parent):
    with TemporaryDirectory() as td:
        tmp = Path(td)
        prepare(parent, tmp)
        for name in ("tapes.json", "architecture.npz", "birth-source.npz"):
            assert (tmp / name).read_bytes() == (inputs / name).read_bytes()
        for tag in PARENT_RUNS:
            name = tag + "-anchors.npz"
            with np.load(tmp / name) as a, np.load(inputs / name) as b:
                assert set(a.files) == set(b.files)
                assert all(np.array_equal(a[k], b[k]) for k in a.files)


def verify_life(src, folder, seed, x, body):
    log = records(folder / "replay.jsonl")
    assert len(src) == len(log)
    with np.load(folder / "stages.npz") as grids, np.load(folder / "states.npz") as st:
        full, qs, dense, dq = (grids[k] for k in ("prob", "q", "dense_prob", "dense_q"))
        weights, head, counts, hashes, selected = (
            st[k] for k in ("policy", "head", "observed", "sha", "selected")
        )
        assert full.shape == (4, 2, 3, 32, 3, 4) and dense.shape == (len(log), 8, 3, 4)
        assert weights.shape == (len(log), 10248) and head.shape == (len(log), 3, 9)
        assert tuple(selected) == turns(len(src) - 1)
        for p in (full, dense):
            assert np.isfinite(p).all() and p.min() >= 0
            close(p.sum(axis=-1), np.ones(p.shape[:-1]), 3e-6)
            assert np.array_equal(p[0], np.broadcast_to(p[0, ..., :1, :], p[0].shape))
        assert np.array_equal(weights[0, -6:], np.zeros(6))
        u, noise = random_panel(seed)
        close(st["uniforms"], u)
        close(st["noise"], noise)
        cached_probs = st["source_decisions"]
        truth = np.asarray(
            [
                [offers(ITEMS[i], rules(seed, "stable", 0))[0][1:] for i in pp]
                for pp in PANEL
            ]
        )
        assert np.array_equal(truth[0], truth[1])
        w = np.zeros((3, 9))
        total = 0.0
        updates = 0
        for t, (e, raw) in enumerate(zip(log, src, strict=True)):
            assert all(e[k] == raw[k] for k in raw)
            assert e["turn"] == t + 1 and e["health"] == st["health"][t]
            assert (e["after"] == 0) == (t == len(src) - 1)
            if t:
                assert e["health"] == log[t - 1]["after"]
            h, i, a = e["health"], e["row"], e["action"]
            close(physical(h, a, e["maintenance"], e["noise"])[-1], e["after"])
            assert a == action_at(raw["behavior"], raw["action_u"])
            R = body.rul(e["after"]) - body.rul(h)
            close([R, R - (total - 8) / (8 + t)], [e["reward"], e["signal"]])
            total += R
            hsh = sha256(weights[t].astype("<f4", copy=False).tobytes()).hexdigest()
            assert hsh == hashes[t].decode() and counts[t] == updates
            close(w, head[t])
            close(cached_probs[t], e["policy"]["true"])
            pred = expit(x[PANEL] @ w.T)
            expected = np.stack((pred, truth, np.full_like(pred, 0.5)), axis=-2)
            close(dq[t], expected[1, DENSE])
            if t in selected:
                k = list(selected).index(t)
                close(qs[k], expected)
                assert np.array_equal(dense[t], full[k, 1, 0, DENSE])
            if t + 1 < len(src):
                stats = e["updates"]["true"]
                assert (
                    hsh == stats["before"] and hashes[t + 1].decode() == stats["after"]
                )
                if a:
                    y = e["maintenance"]
                    assert isinstance(y, bool)
                    delta = (int(y) - expit(x[i] @ w[a - 1])) * x[i]
                    w[a - 1] += delta / max(1.0, np.linalg.norm(delta) / 0.4)
                    updates += 1
            else:
                assert not e["updates"]
        return len(log), updates


def verify_draws(src, out):
    byseed = {}
    with gzip.open(out / "choices.csv.gz", "rt", encoding="utf8", newline="") as f:
        for row in csv.DictReader(f):
            byseed.setdefault(int(row["seed"]), []).append(row)
    total = 0
    for seed, rows in byseed.items():
        folder = src / "lives" / f"s{seed}"
        with np.load(folder / "stages.npz") as z, np.load(folder / "states.npz") as st:
            p = z["prob"]
            us, noise = st["uniforms"], st["noise"]
            seen = set()
            for e in rows:
                k = STAGES.index(e["stage"])
                w = ("original", "held").index(e["wording"])
                h = float(e["health"])
                hi = HEALTH.index(h)
                i = int(e["prompt_id"])
                j = list(PANEL[w]).index(i)
                m = VIEWS.index(e["input"])
                key = (k, w, hi, j, m)
                assert key not in seen
                seen.add(key)
                prob = p[k, w, hi, j, m]
                a = action_at(prob, us[j, 0])
                chance, fav = offers(ITEMS[i], rules(seed, "stable", 0))
                maintained = bool(us[j, 1] < chance[a]) if a else None
                h2 = physical(h, a, maintained, noise[j])[-1]
                assert (
                    a == int(e["action"])
                    and int(e["greedy"]) == int(prob.argmax())
                    and int(e["favored"]) == fav
                )
                assert e["matched_maintenance"] == (
                    "" if maintained is None else str(maintained)
                )
                close(h2, float(e["after"]))
                assert int(e["completed"]) == int(a >= 2 and h2 > 0)
            assert len(seen) == 4 * 2 * 3 * 32 * 3
            total += len(seen)
    return total


def audit(src, inputs, parent, archive, out):
    manifest(archive)
    archived(inputs, parent)
    tapes = json.loads((inputs / "tapes.json").read_bytes())
    receipt = json.loads((src / "receipt.json").read_bytes())
    assert tuple(receipt["seeds"]) == PILOT and receipt["frozen_backbone"]
    assert receipt["max_identical_lane_error"] == 0
    with np.load(src / "features.npz") as f:
        assert np.array_equal(f["ids"], FEATURE_IDS)
        x, scales = features(f["name"], f["text"], FEATURE_IDS)
        close(x[FEATURE_IDS], f["x"])
        close(scales, f["scales"])
    body = BodyValue()
    info = {}
    for seed in PILOT:
        folder = src / "lives" / f"s{seed}"
        info[seed] = verify_life(tapes[str(seed)], folder, seed, x, body)
        caps = json.loads((folder / "anchors.json").read_bytes())
        with (
            np.load(folder / "states.npz") as st,
            np.load(folder / "stages.npz") as panel,
        ):
            for tag in PARENT_RUNS:
                with np.load(inputs / (tag + "-anchors.npz")) as prev:
                    key = f"s{seed}_"
                    expected = st["sha"] == prev[key + "sha"]
                    assert int(expected.sum()) == caps[tag]["matching_parameter_states"]
                    assert (
                        np.array_equal(st["policy"][-1], prev[key + "last_weights"])
                        == caps[tag]["identical_last_parameters"]
                    )
                    delta = abs(
                        np.take(panel["prob"][-1], [0, 2], axis=-2).transpose(
                            3, 0, 1, 2, 4
                        )
                        - prev[key + "last_probs"]
                    )
                    close(
                        100 * float(delta.max()),
                        caps[tag]["terminal_panel_max_pp"],
                        1e-5,
                    )
    result = analyze(src, out)
    tested = verify_draws(src, out)
    write(
        out / "audit.json",
        {
            "seeds": info,
            "stages": len(STAGES),
            "counterfactual_draws": tested,
            "checked_states": sum(z[0] for z in info.values()),
            "checked_actual_maintenance_writes": sum(z[1] for z in info.values()),
            "verified_archive": True,
            "gpu_reference": receipt,
        },
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("src", "inputs", "parent", "archive", "out"):
        p.add_argument("--" + field, required=True, type=Path)
    result = audit(**vars(p.parse_args()))
    print(json.dumps(result["criteria"]))
