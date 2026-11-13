"""CPU reconstruction of chronology, executed actions, body, individual predictor weights and test panels."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import records
from experiments.individual_life.design import action_at
from experiments.lifetime_cooperation.audit import physical
from experiments.lifetime_cooperation.design import (
    ITEMS,
    PANEL,
    PILOT,
    physical_step,
    rules,
    schedule,
)
from experiments.loreft_omission.audit import manifest
from experiments.partner_damage.design import swapped
from experiments.social_closed_loop.core import BodyValue

from .analysis import analyze
from .design import (
    ARMS,
    FEATURE_IDS,
    LANDMARKS,
    MODES,
    SHADOWS,
    features,
    prepare_seeds,
)


def same(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def audit_life(seed, phase, arm, rows, x, weights, panel, body, source):
    ids, uniforms, noise, _ = schedule(seed)
    h, total = 350.0, 0.0
    if phase == "replay":
        keys = (*ARMS, *SHADOWS)
        assert len(rows) == len(source)
    else:
        keys = (arm, *SHADOWS) if arm == "true" else (arm,)
    models = {k: np.zeros((3, 9)) for k in keys}
    updates = {k: 0 for k in keys}
    last = {}
    for k, r in enumerate(rows):
        assert r["health"] == h > 0 and r["turn"] == k + 1 and r["row"] == ids[k]
        same([r["action_u"], r["partner_u"]], uniforms[k])
        same(r["noise"], noise[k])
        i, a, y = r["row"], r["action"], r["maintenance"]
        if phase == "replay":
            assert all(r[f] == source[k][f] for f in source[k])
            s = source[k]
            assert a == action_at(s["behavior"], s["action_u"])
            policies = r["policy"]
            stats = r["updates"]
        else:
            assert a == action_at(r["policy"], r["action_u"])
            policies = {arm: r["policy"]}
            stats = {arm: r["update"]} if "update" in r else {}
            if k == 0:
                same(r["policy"], r["birth"])
        if k == 0 and phase == "replay":
            for p in policies.values():
                same(p, policies["body"])
        bits = rules(seed, "stable", k)
        assert tuple(r["rules"]) == bits
        h2, _, m, rescued = physical_step(
            h, a, ITEMS[i], bits, r["partner_u"], noise[k]
        )
        assert h2 == r["after"] and m == y and r["rescued_answer"] == rescued
        same(physical(h, a, y, r["noise"])[-1], h2)
        reward = body.rul(h2) - body.rul(h)
        baseline = (total - 8) / (8 + k)
        same(
            [reward, baseline, reward - baseline],
            [r["reward"], r["baseline"], r["signal"]],
        )
        assert (h2 > 0) == bool(stats)
        for name, u in stats.items():
            if k:
                assert last[name] == u["before"]
            current = policies[name][a]
            behavior = r["behavior"][a] if phase == "replay" else current
            same(
                [current, behavior, np.clip(current / behavior, 0.5, 2)],
                [u["current"], u["behavior"], u["importance"]],
            )
            assert u["actual_l2"] >= 0 and u["orthogonality_error"] < 2e-5
            last[name] = u["after"]
        for name in keys:
            model = models[name]
            rec = r["forecasts"][name]
            positions = [i, swapped(i)]
            pred = expit(x[positions] @ model.T)
            same(pred, rec["before"])
            pid = ITEMS[i]["partner"]
            supervised = (
                name != "body"
                and (name != "drop93" or pid != 3)
                and (name != "drop86" or pid != 2)
            )
            if a != 0 and h2 > 0 and supervised:
                write_idx = swapped(i) if name == "wrong" else i
                expected = expit(np.dot(model[a - 1], x[write_idx]))
                delta = x[write_idx] * (int(y) - expected)
                model[a - 1] += delta / max(1.0, np.linalg.norm(delta) / 0.4)
                updates[name] += 1
            same(expit(x[positions] @ model.T), rec["after"])
            assert updates[name] == rec["count"]
        total += reward
        h = h2
    assert h == 0 and rows[-1]["after"] == 0
    same(total, -body.rul(350))
    for name, w in models.items():
        same(weights["head_" + name], w)
        same(panel[name], expit(x[PANEL] @ w.T))
    for name, value in last.items():
        a = weights["policy_" + name].astype("<f4")
        assert sha256(a.tobytes()).hexdigest() == value
        assert a.shape == ((10242 if name == "body" else 10248),)
    return {
        "turns": len(rows),
        "body_updates_per_arm": len(rows) - 1,
        "supervised": updates,
    }


def audit(root, inputs, source17, source19, archive, out):
    manifest(archive)
    with TemporaryDirectory() as tmp:
        dest = Path(tmp)
        prepare_seeds(source17, source19, dest)
        assert json.loads((dest / "tapes.json").read_bytes()) == json.loads(
            (inputs / "tapes.json").read_bytes()
        )
        with (
            np.load(dest / "archive-ends.npz") as old,
            np.load(inputs / "archive-ends.npz") as actual,
        ):
            for name in old.files:
                same(old[name], actual[name])
    locked = json.loads((inputs / "tapes.json").read_bytes())
    receipt = json.loads((root / "receipt.json").read_bytes())
    assert (
        receipt["frozen_backbone"]
        and tuple(receipt["seeds"]) == PILOT
        and "L4" in receipt["device"]
    )
    body = BodyValue()
    with np.load(root / "features.npz") as f:
        assert np.array_equal(f["ids"], FEATURE_IDS)
        assert all(ITEMS[i]["wording"] == 3 for i in LANDMARKS)
        x, scale = features(f["name"], f["text"], FEATURE_IDS)
        same(x[FEATURE_IDS], f["x"])
        same(f["scales"], scale)
        same(np.linalg.norm(x[FEATURE_IDS], axis=1), np.ones(len(FEATURE_IDS)))
        assert np.all(f["windows"][:, 1] < f["windows"][:, 2]) and np.all(
            f["windows"][:, 2] < f["windows"][:, 3]
        )
    for a, b in zip(*PANEL, strict=True):
        r, t = ITEMS[a], ITEMS[b]
        assert (
            all(r[k] == t[k] for k in ("partner", "frame", "task", "layout"))
            and r["text"] != t["text"]
        )
    info = {}
    with np.load(inputs / "archive-ends.npz") as old:
        for seed in PILOT:
            for phase in ("replay", "live"):
                dst = root / "lives" / f"s{seed}" / phase
                with (
                    np.load(dst / "weights.npz") as weights,
                    np.load(dst / "panel.npz") as panel,
                ):
                    assert panel["policy"].shape == (len(MODES), 2, 4, 32, 4)
                    born = panel["policy"][MODES.index("birth")]
                    if not info:
                        reference = born.copy()
                    assert np.array_equal(born, reference)
                    for name in ("replay",) if phase == "replay" else ARMS:
                        rows = records(dst / (name + ".jsonl"))
                        info[f"{seed}.{phase}.{name}"] = audit_life(
                            seed,
                            phase,
                            name,
                            rows,
                            x,
                            weights,
                            panel,
                            body,
                            locked[str(seed)],
                        )
                        if phase == "replay":
                            stats = json.loads((dst / "legacy.json").read_bytes())
                            diffs = abs(
                                np.stack([u["policy"]["body"] for u in rows])
                                - old[f"s{seed}_actual"]
                            )
                            pdiff = abs(panel["policy"][0, 1] - old[f"s{seed}_held"])
                            same(
                                [diffs.max(), diffs.mean(), pdiff.max(), pdiff.mean()],
                                [
                                    stats[k]
                                    for k in (
                                        "max_actual",
                                        "mean_actual",
                                        "max_held",
                                        "mean_held",
                                    )
                                ],
                            )
    result = analyze(root, out)
    evidence = {
        "lives": info,
        "hashes_verified": True,
        "locked_input_rows": len(FEATURE_IDS),
        "reference_histories": len(PILOT),
        "autonomous": len(PILOT) * len(ARMS),
        "paired_wording_rows": len(PANEL[0]),
        "cpu_reconstruction": "saved Qwen features, achieved forecast parameters, decisions, bodies and readouts; no CPU Qwen re-inference",
    }
    write(out / "audit.json", evidence)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "inputs", "source17", "source19", "archive", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    result = audit(**vars(a))
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_pass")}))
