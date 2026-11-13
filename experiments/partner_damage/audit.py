"""Independent CPU reconstruction from the original physical records and exported probability/weight panels."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.audit import physical
from experiments.lifetime_cooperation.design import ITEMS, PILOT
from experiments.loreft_omission.audit import manifest
from experiments.social_closed_loop.core import BodyValue
from .analysis import analyze
from .design import CELLS, DOSES, PARTNER, SHA17, SHA19, SMALL, swapped


def get(p):
    return json.loads(p.read_bytes())


def same(a, b, tol=0.0):
    if tol:
        np.testing.assert_allclose(a, b, atol=tol, rtol=0.0)
    else:
        assert np.array_equal(a, b)


def sha(x):
    return sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def audit(root, inputs, src17, src19, out, archive=None):
    if archive:
        manifest(archive)
    assert (
        sha256(src17.read_bytes()).hexdigest() == SHA17
        and sha256(src19.read_bytes()).hexdigest() == SHA19
    )
    receipt = get(root / "receipt.json")
    assert receipt["frozen_backbone"] and receipt["birth_route_max_error"] == 0
    result = analyze(root, inputs, out)
    body = BodyValue()
    catalog = get(inputs / "events.json")
    e_by = {(e["seed"], e["turn"]): e for e in catalog}
    cases = 0
    rows = 0
    all_n = 0
    silent = 0
    equal_hash = 0
    unequal_hash = 0
    delayed = 0
    anchors = 0
    phys = []
    seen = set()
    with (
        ZipFile(src17) as f,
        ZipFile(src19) as w,
        np.load(inputs / "anchors.npz") as db,
    ):
        for s in PILOT:
            tape = list(
                map(
                    json.loads,
                    f.read(
                        f"output/lives/s{s}/stable/qwen_learn/events.jsonl"
                    ).splitlines(),
                )
            )
            preexisting = 0.0
            chosen = {}
            all_n += len(tape) - 1
            with np.load(
                BytesIO(w.read(f"output/lives/s{s}/normal.npz")), allow_pickle=False
            ) as z:
                weights, panel = z["weights"], z["panels"]
            for k, e in enumerate(tape[:-1]):
                t = k + 1
                a = e["action"]
                h = e["health"]
                raw = e["reward"]
                base = (preexisting - 8) / (8 + k)
                preexisting += raw
                same(base, e["baseline"], 1e-10)
                cf0 = physical(h, a, e["maintenance"], e["noise"])[-1]
                same(e["after"], cf0, 1e-9)
                same(body.rul(cf0) - body.rul(h), raw, 1e-8)
                if a == 0:
                    silent += 1
                    assert e["maintenance"] is None
                    continue
                after = [physical(h, a, flag, e["noise"])[-1] for flag in (True, False)]
                sig = [body.rul(v) - body.rul(h) - base for v in after]
                phys.append(sig)
                rows += 1
                partner = ITEMS[e["row"]]["partner"]
                chosen.setdefault((partner, e["maintenance"]), t)
                if (s, t) not in e_by:
                    continue
                case = e_by[s, t]
                key = case["key"]
                assert case["factual"] == e
                assert (s, t) not in seen
                seen.add((s, t))
                d = root / "cases" / key
                r = get(d / "receipt.json")
                other = swapped(e["row"])
                assert ITEMS[other]["partner"] == 5 - partner and all(
                    ITEMS[other][key_] == ITEMS[e["row"]][key_]
                    for key_ in ("text", "frame", "task", "wording", "layout")
                )
                zname = lambda suffix: db[key + "_" + suffix]
                same(weights[k], zname("pre"))
                same(weights[k + 1], zname("post"))
                same(panel[k], zname("prepanel"))
                same(panel[k + 1], zname("postpanel"))
                fact = 0 if e["maintenance"] else 3
                for j, (lab, (_, light)) in enumerate(zip(CELLS, DOSES, strict=True)):
                    block = r["cells"][lab]
                    dose = 0 if light else 1
                    same(
                        [block["after"], block["signal"], block["damage"]],
                        [after[dose], sig[dose], h - after[dose]],
                        2e-8,
                    )
                    assert block["maintenance"] == DOSES[j][0]
                with np.load(d / "factorial.npz") as z:
                    cw = z["cell_weights"]
                    p = z["panels"]
                    tid = z["identity_train"]
                    assert p.shape == (5, 4, 32, 4) and tid.shape == (3, 4, 32, 4)
                    same(p.sum(-1), np.ones(p.shape[:-1]), 3e-6)
                    same(z["pre_weights"], weights[k])
                    for j in (0, 2):
                        same(cw[j], cw[j + 1])
                        same(p[1 + j], p[2 + j])
                        equal_hash += 1
                    unequal_hash += int(not np.array_equal(cw[0], cw[2]))
                    same(tid[0], p[1 + fact])
                    same(abs(p[0] - panel[k]).max(), r["pre_panel_error"])
                    same(abs(p[1 + fact] - panel[k + 1]).max(), r["post_panel_error"])
                    assert (
                        sha(weights[k + 1]) == r["source_expected_hash"]
                        and sha(cw[fact]) == r["source_update_hash"]
                    )
                    anchors += int(sha(cw[fact]) == sha(weights[k + 1]))
                    for j in range(4):
                        st = r["steps"][j]
                        assert st["before"] == sha(weights[k]) and st["after"] == sha(
                            cw[j]
                        )
                        same(
                            st["actual_l2"],
                            np.linalg.norm(cw[j].astype(float) - weights[k]),
                            1e-8,
                        )
                    x, y = (
                        z["grad_source"].astype(float),
                        z["grad_swapped"].astype(float),
                    )
                    same(
                        r["gradient_cosine"],
                        np.dot(x, y)
                        / max(1e-16, np.linalg.norm(x) * np.linalg.norm(y)),
                        3e-5,
                    )
                with np.load(d / "swaps.npz") as z:
                    b = z["panels"]
                    assert b.shape == (6, 4, 32, 4)
                    same(b[:2], p[[0, 1 + fact]])
                    same(b[2], b[0])
                    same(b[3], b[1])
                    same(b[5][:, PARTNER == partner], b[1][:, PARTNER == partner])
                    same(b[5][:, PARTNER != partner], b[0][:, PARTNER != partner])
                    if np.isfinite(z["delta_cosine"]).any():
                        assert np.nanmax(abs(z["delta_cosine"])) <= 1 + 1e-10
                with np.load(d / "sites.npz") as z:
                    ca = z["calibration"].astype(float)
                    p0 = z["panels"]
                    assert p0.shape == (16, 5, 4) and tuple(ca.shape) == (2, 6, 4)
                    same(ca[:, 0], np.broadcast_to(ca[0, 0], (2, 4)))
                    eg, tg = z["early"], z["terminal"]
                    same(eg.shape, (5, 10242))
                    same(tg.shape, (5, 10242))
                    same(
                        p0[:, 0],
                        np.array([panel[k, hi, j] for hi, j in SMALL]),
                        max(1e-6, r["pre_panel_error"] + 1e-7),
                    )
                    vv = 50 * np.abs(ca[:, 1:] - ca[:, :1]).sum(-1)
                    ej, lj = r["site"]["early_index"], r["site"]["terminal_index"]
                    feasible = [
                        (ei, li)
                        for ei in range(5)
                        for li in range(5)
                        if min(vv[0, ei], vv[1, li]) >= 0.5
                    ]
                    cand = feasible or list(np.ndindex(5, 5))
                    assert (ej, lj) == min(
                        cand,
                        key=lambda pair: (
                            abs(vv[0, pair[0]] - vv[1, pair[1]]),
                            -min(vv[0, pair[0]], vv[1, pair[1]]),
                            pair,
                        ),
                    )
                    assert r["site"]["matched"] == (
                        bool(feasible) and abs(vv[0, ej] - vv[1, lj]) < 0.5
                    )
                if case["focus"]:
                    with np.load(d / "continuation.npz") as z:
                        for ii, n in enumerate(z["turns"]):
                            q = z["panels"][ii]
                            same(q[0], q[1])
                            same(q[2], q[3])
                            same(
                                q[fact],
                                panel[n - 1],
                                max(1e-6, r["continuation"]["archive_max"][ii] + 1e-7),
                            )
                            delayed += 1
                cases += 1
            ordinary = {
                t for (seed, t), v in e_by.items() if seed == s and v["ordinary"]
            }
            assert ordinary == set(chosen.values())
    assert seen == set(e_by)
    phys = np.asarray(phys)
    assert result["physics_sign_flip"] == np.sum((phys[:, 0] > 0) & (phys[:, 1] < 0))
    tokenrows = get(root / "representation-rows.json")
    with np.load(root / "representations.npz") as z:
        for line in result["readouts"]:
            x = unit(z[line["layer"]].astype(float))
            train = [j for j, v in enumerate(tokenrows) if v["split"] == 0]
            pv = np.mean(
                [x[j] for j in train if tokenrows[j]["partner"] == 3], axis=0
            ) - np.mean([x[j] for j in train if tokenrows[j]["partner"] == 2], axis=0)
            center = np.mean(x[train], axis=0)
            ix = [
                j
                for j, v in enumerate(tokenrows)
                if v["split"] == 1 and v["health"] == line["health"]
            ]
            values = (x[ix] - center) @ pv
            truth = [tokenrows[j]["partner"] for j in ix]
            acc = np.mean(
                [
                    (1.0 if (v > 0) == (label == 3) else 0.0) if v != 0 else 0.5
                    for v, label in zip(values, truth, strict=True)
                ]
            )
            same(acc, line["accuracy"], 1e-10)
    data = dict(
        nonterminal_events=all_n,
        silent_unobservable=silent,
        physical_counterfactual_pairs=rows,
        cases=cases,
        matched_label_parameter_and_probability_pairs=equal_hash,
        changed_weight_light_vs_heavy=unequal_hash,
        factual_updates_PR19_exact=anchors,
        continuation_snapshots=delayed,
        pinned_birth_route_difference=receipt["birth_route_max_error"],
        readout_contexts=len(tokenrows),
        restore_and_routed_equations_verified=True,
        position_pairs_verified=cases,
    )
    write(out / "audit.json", data)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "inputs", "source17", "source19", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            audit(a.root, a.inputs, a.source17, a.source19, a.out, a.archive), indent=2
        )
    )
