"""Audit the stochastic encounters and test transfer at novel transition states and wings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.study import basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import arr, fit
from experiments.lethal_boundary.design import ANCHOR_Q, HEALTH, INTERIOR, QS, levels
from experiments.lethal_boundary.evaluate import (
    BANDS,
    MI,
    index,
    interpolation,
    physics_matrix,
    policies,
    scores,
    summaries,
    summarize,
)
from experiments.lethal_boundary.jitter import LocalHealthPolicy, signals
from experiments.lethal_boundary.states import Bank, probes
from experiments.lethal_boundary.transfer import closed_form_record
from experiments.lethal_boundary.transition import AXIS, BORDER, CELLS, SEEDS
from experiments.lethal_boundary.verify_fits import independent_settle
from experiments.social_closed_loop.core import BodyValue


def regions(curve):
    summary = summaries(curve)
    center = (HEALTH > BORDER[0]) & (HEALTH < BORDER[1])
    for label, mask in (
        ("between_anchors", center),
        ("wings", BANDS["transition"] & ~center),
    ):
        avg = curve[INTERIOR][:, mask].mean(axis=(0, 1))
        summary.update({label + "_" + key: float(avg[idx]) for key, idx in MI.items()})
    return summary


def run(root, cache, arch, reference, out):
    bank = Bank(cache, arch)
    body = BodyValue()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(reference / "physical.npz") as ref:
        dv = ref["conditional_death"]
        rv = ref["conditional_rul_change"]
    physical = [physics_matrix(dv, rv, m) for m in (0, 1)]
    records = []
    surfaces = []
    events = np.zeros((2, 2), int)
    nchoices = nsettle = nsteps = exchanged = nselected = 0
    refits = []
    idx = index(AXIS, HEALTH)
    for s in SEEDS:
        folder = root / f"s{s}"
        with np.load(folder / "tape.npz") as t:
            a, mu, u, e = draw_tape(s, bank.training[None][2])
            for k, v in zip(
                ("action", "mu", "uniform", "noise"), (a, mu, u, e), strict=True
            ):
                np.testing.assert_array_equal(t[k], v)
            fx, border, profile = t["fixed"], t["border"], t["profile"]
        nchoices += a.size
        mask = fx != border
        selected = np.broadcast_to(mask[:, :, None], a.shape)
        nselected += int(selected.sum())
        np.testing.assert_array_equal(mask.sum(0), np.ones(768))
        assert selected.sum() == 6144 and not np.intersect1d(HEALTH, border).size
        subset = np.array(
            [
                r["partner"] * 16 + r["frame"] * 8 + r["task"] * 2 + profile[i]
                for i, r in enumerate(bank.train)
            ]
        )
        for label in np.unique(subset):
            for j in range(4):
                np.testing.assert_array_equal(
                    [((border[j] == x) & (subset == label)).sum() for x in BORDER],
                    [3, 3],
                )
        stats = []
        surf = []
        for m in (0, 1):
            loc = folder / f"m{m}"
            vals = {}
            curves = {}
            fav = np.array([2 + (r["partner"] ^ r["frame"] ^ m) for r in bank.train])
            expected = np.where(
                a == 0,
                -1,
                u
                < np.where(
                    a == 1,
                    0.15,
                    np.where(
                        a == fav[None, :, None], ANCHOR_Q[profile][None, :, None], 0.1
                    ),
                ),
            )
            with np.load(loc / "outcomes.npz") as z:
                np.testing.assert_array_equal(z["response"], expected)
                for key, h in (("fixed", fx), ("border", border)):
                    y = independent_settle(h, a, expected, e)
                    np.testing.assert_allclose(y, z[key + "_after"], atol=2e-12, rtol=0)
                    np.testing.assert_allclose(
                        body.interpolate(y, body.reference)
                        - body.interpolate(h[:, :, None], body.reference),
                        z[key + "_reward"],
                        atol=1e-10,
                        rtol=0,
                    )
                    nsettle += a.size
                for j, h in enumerate(BORDER):
                    take = (border[:, :, None] == h) & (expected == 1) & (a >= 2)
                    events[j] += np.array(
                        [int(take.sum()), int((z["border_after"][take] > 0).sum())]
                    )
                perm = z["permutation"].astype(int)
                sequence = np.arange(a.size).reshape(a.shape)
                np.testing.assert_array_equal(np.sort(perm.ravel()), sequence.ravel())
                np.testing.assert_array_equal(perm[~selected], sequence[~selected])
                labs = np.array(
                    [
                        r["partner"] * 4 + r["frame"] * 2 + profile[i]
                        for i, r in enumerate(bank.train)
                    ]
                )
                stratum = (labs[None, :, None] * 4 + a) * 3 + (expected + 1)
                np.testing.assert_array_equal(stratum, stratum.ravel()[perm])
                allh = np.broadcast_to(border[:, :, None], a.shape)
                exchanged += int((allh[selected] != allh.ravel()[perm][selected]).sum())
                if (s, m) == (SEEDS[0], 0):
                    grid = bank.record_train(profile)
                    before = body.interpolate(border, body.reference)
                    scale = np.minimum(1.0, before)
                    cont = (
                        body.interpolate(z["border_after"], body.reference)
                        / scale[:, :, None]
                    )
                    shuffled = np.where(
                        selected,
                        cont.ravel()[perm] * scale[:, :, None] - before[:, :, None],
                        z["border_reward"],
                    )
                    for name, (sensor, signal) in CELLS.items():
                        h = fx if sensor == "fixed" else border
                        source = fx if signal == "fixed" else border
                        r = shuffled if signal == "shuffled" else z[signal + "_reward"]
                        model = LocalHealthPolicy(bank, "linear")
                        fit(
                            model,
                            {**grid, "g": arr(basis(h), "cpu")},
                            signals(a, r, mu, source, body),
                        )
                        with np.load(loc / (name + ".npz")) as weights:
                            err = max(
                                float(np.abs(v.detach().numpy() - weights[k]).max())
                                for k, v in model.named_parameters()
                            )
                        with np.load(loc / (name + "-probe.npz")) as probe:
                            err = max(
                                err,
                                float(
                                    np.abs(
                                        probes(model, bank, "record", AXIS)["panel"]
                                        - probe["panel"]
                                    ).max()
                                ),
                            )
                        assert err < 2e-5
                        refits.append(
                            {"seed": s, "mapping": m, "arm": name, "max_error": err}
                        )
            logs = json.loads((loc / "steps.json").read_bytes())
            for name in CELLS:
                assert len(logs[name]) == 75 and np.isfinite(logs[name]).all()
                nsteps += 75
                with np.load(loc / (name + "-probe.npz")) as z:
                    raw = z["mean"]
                actual = policies(raw, "iid")[:, idx]
                measured = scores(actual, physical[m], dv[:, 2, 1])
                curves[name] = measured
                vals[name] = regions(measured)
                if name == "border":
                    anchors = np.unique(np.r_[levels("M", s), BORDER])
                    for mode in ("nearest", "loglinear"):
                        approx = interpolation(
                            closed_form_record(raw, mode), AXIS, anchors, mode
                        )
                        vals[name + "-" + mode] = regions(
                            scores(approx, physical[m], dv[:, 2, 1])
                        )
            stats.append(vals)
            surf.append(curves)
        records.append(
            {
                k: {j: float(np.mean([v[k][j] for v in stats])) for j in stats[0][k]}
                for k in stats[0]
            }
        )
        surfaces.append({k: (surf[0][k] + surf[1][k]) / 2 for k in surf[0]})
    absolute = {
        k: {j: summarize([v[k][j] for v in records]) for j in records[0][k]}
        for k in records[0]
    }
    contrasts = {
        other: {
            j: summarize([v["border"][j] - v[other][j] for v in records])
            for j in records[0][other]
        }
        for other in records[0]
        if other != "border"
    }
    deaths = summarize(
        [
            v["fixed"]["transition_total_death"] - v["border"]["transition_total_death"]
            for v in records
        ]
    )
    vs = contrasts["fixed"]
    nn = contrasts["border-nearest"]
    passed = (
        deaths["mean"] >= 0.01
        and deaths["ci95"][0] > 0
        and deaths["positive"] >= 10
        and vs["transition_safe_task"]["mean"] >= 0
        and vs["healthy_safe_task"]["mean"] >= -0.05
        and vs["doomed_intrinsic_fatal_answer"]["mean"] <= 0.05
    )
    beyond = (
        passed
        and nn["transition_total_death"]["mean"] <= 0
        and nn["transition_safe_task"]["mean"] >= 0
    )
    write(out / "per-seed.json", {"seeds": SEEDS, "metrics": records})
    write(
        out / "summary.json",
        {
            "seeds": SEEDS,
            "new_anchor_healths": BORDER.tolist(),
            "observed_maintained_answers_then_survivals": events.tolist(),
            "absolute": absolute,
            "border_minus_control": contrasts,
            "transition_death_reduction": deaths,
            "targeted_repair_pass": bool(passed),
            "beyond_nearest_pass": bool(beyond),
            "test_transition_points": int(BANDS["transition"].sum()),
            "test_points_between_new_anchors": int(
                ((HEALTH > BORDER[0]) & (HEALTH < BORDER[1])).sum()
            ),
            "raw_action_replays": nchoices,
            "independent_settlements": nsettle,
            "verified_optimizer_steps": nsteps,
            "sampled_replacements_before_mirroring": nselected,
            "permuted_across_the_two_healths": exchanged,
            "permuted_total": 2 * nselected,
            "refitted_checkpoints": refits,
        },
    )
    np.savez_compressed(
        out / "seed-surfaces.npz",
        health=HEALTH,
        q=QS,
        seeds=SEEDS,
        **{k: np.stack([v[k] for v in surfaces]) for k in CELLS},
    )
    print(
        "Actual death contrast",
        deaths,
        "safe task",
        vs["transition_safe_task"],
        "passed",
        bool(passed),
        bool(beyond),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "cache", "arch", "reference", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(a.root, a.cache, a.arch, a.reference, a.output)
