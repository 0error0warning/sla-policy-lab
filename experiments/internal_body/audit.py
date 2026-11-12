"""Recompute chosen actions, actual body settlements, and terminal individual weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.study import PARTNERS, WORDS
from experiments.direct_policy.protocol import (
    ACTIONS,
    COSTS,
    probabilities,
    render,
    settle,
    write,
)
from experiments.internal_body.credit import RULES, Learner
from experiments.internal_body.design import HIGH_H, SPLIT_H
from experiments.internal_body.lives import WORLDS, life_tape, run_one
from experiments.internal_body.personal import ARMS, LifeActor, LifeInputs
from experiments.social_closed_loop.core import BodyValue


def acquisition(d, native, seeds, rows, body):
    counts = {
        "sampled_choices": 0,
        "paired_physical_transitions": 0,
        "critical_answers": 0,
    }
    for seed in seeds:
        a, mu, u, eps = draw_tape(seed, native[:768])
        with np.load(d / f"behavior-{seed}.npz") as tape:
            for k, item in (
                ("action", a),
                ("mu", mu),
                ("partner_uniform", u),
                ("body_noise", eps),
            ):
                np.testing.assert_array_equal(tape[k], item)
        counts["sampled_choices"] += a.size
        for m in (0, 1):
            p = np.array(
                [
                    [0.0, *probabilities(PARTNERS[r["partner"]], r["frame"], m)]
                    for r in rows[:768]
                ]
            )
            accepted = u < p[np.arange(768)[None, :, None], a]
            response = np.where(a == 0, -1, accepted)
            with np.load(d / f"s{seed}-m{m}/outcomes.npz") as observed:
                np.testing.assert_array_equal(response, observed["response"])
                for levels, h_key, r_key in (
                    (SPLIT_H, "after", "reward"),
                    (HIGH_H, "high_after", "high_reward"),
                ):
                    h0 = np.broadcast_to(np.asarray(levels)[:, None, None], a.shape)
                    h = h0.copy()
                    for j in range(5):
                        cost = np.asarray(COSTS)[a] if j == 0 else (response != 1)
                        h = np.maximum(
                            0.0, h - cost * (2.0 - h / 350.0) * (1.0 + eps[..., j])
                        )
                    r = body.interpolate(h, body.reference) - body.interpolate(
                        h0, body.reference
                    )
                    np.testing.assert_allclose(h, observed[h_key], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(r, observed[r_key], atol=1e-12, rtol=0)
                    counts["paired_physical_transitions"] += a.size
                assert np.all(observed["high_after"] > 0)
                counts["critical_answers"] += int(
                    ((a >= 2) & (np.array(SPLIT_H)[:, None, None] <= 2.5)).sum()
                )
    return counts


def physical_trace(tr, tape, data, world, rev, body, export=None, key=None):
    ix, u, noise = tape
    t = len(tr)
    assert tr[0, 1] == 350.0 and (tr[-1, 4] == 0.0 or t == 256)
    np.testing.assert_array_equal(tr[:, 0], ix[:t])
    np.testing.assert_array_equal(tr[1:, 1], tr[:-1, 4])
    np.testing.assert_allclose(tr[:, 7:11].sum(1), 1.0, atol=3e-7, rtol=0)
    np.testing.assert_array_equal(
        (u[:t, :1] > tr[:, 7:11].cumsum(1)[:, :3]).sum(1), tr[:, 2]
    )
    for j, r in enumerate(tr):
        i, action = int(r[0]), int(r[2])
        row = data.rows[i]
        p = row["partner"] % 2
        mapping = 1 - world[p] if rev and j >= 32 and p == 0 else world[p]
        pr = probabilities(PARTNERS[p], row["frame"], mapping)
        yes = bool(u[j, 1] < pr[action - 1]) if action else None
        assert r[3] == (-1 if yes is None else int(yes))
        after, events = settle(r[1], action, yes, noise[j])
        reward = body.rul(after) - body.rul(r[1])
        assert abs(after - r[4]) < 1e-10 and abs(reward - r[5]) < 1e-10
        if export is not None:
            event = {
                "life": key,
                "turn": j + 1,
                "partner": ("P86", "P93")[p],
                "request": row["text"],
                "action": ACTIONS[action],
                "native_word": WORDS[action],
                "policy": r[7:11].tolist(),
                "response": None if yes is None else bool(yes),
                "template_rendering": render(action, row["task"], j % 3),
                "health_before": r[1],
                "health_after": after,
                "body_substeps": events,
                "exogenous_action_partner_uniforms": u[j].tolist(),
                "substep_noise": noise[j].tolist(),
                "reference_life_change": reward,
                "scaled_outcome": r[6],
                "gradient_norm": r[-2],
                "normalizer": r[-1],
            }
            export.write(json.dumps(event, ensure_ascii=False) + "\n")


def lives(folder, seeds, series, data, body):
    all_arms = ARMS if series == "life" else RULES
    nlife = nturn = restores = first32 = 0
    for seed in seeds:
        with (
            np.load(folder / f"{series}-{seed}-raw.npz") as traces,
            np.load(folder / f"{series}-{seed}-parameters.npz") as checkpoints,
        ):
            for wi, world in enumerate(WORLDS):
                tape = life_tape(seed, world, data.rows)
                for key, values in zip(("context", "uniform", "noise"), tape):
                    np.testing.assert_array_equal(traces[f"w{wi}-{key}"], values)
                for arm in all_arms:
                    ta = traces[f"w{wi}-v0-{arm}"]
                    tb = traces[f"w{wi}-v1-{arm}"]
                    np.testing.assert_array_equal(ta[:32], tb[:32])
                    first32 += 1
                    for rev, tr in enumerate((ta, tb)):
                        physical_trace(tr, tape, data, world, rev, body)
                        nlife += 1
                        nturn += len(tr)
                        if seed in (seeds[0], seeds[-1]):
                            actor = (
                                LifeActor(data, folder, seed, arm)
                                if series == "life"
                                else Learner(data, folder, seed, arm)
                            )
                            again, _ = run_one(actor, seed, world, rev, tape, body)
                            np.testing.assert_array_equal(again, tr)
                            if arm != "fixed":
                                np.testing.assert_array_equal(
                                    actor.parameters(),
                                    checkpoints[f"w{wi}-v{rev}-{arm}"],
                                )
                                restores += 1
    return {
        "lives": nlife,
        "turns": nturn,
        "all_paired_first32_equal": first32,
        "bitwise_restored_final_checkpoints": restores,
    }


def main():
    p = argparse.ArgumentParser()
    for k in ("root", "cache", "gpu", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--placement-capture", type=Path)
    a = p.parse_args()
    root = a.root
    primary = root / ("population" if (root / "population").is_dir() else "results")
    data = LifeInputs(a.cache)
    body = BodyValue()
    series = [
        ("population", primary, [*range(64001, 64005), *range(64101, 64121)]),
        ("native", root / "native", [*range(64401, 64405), *range(64501, 64521)]),
        ("credit", root / "credit", [*range(64601, 64605), *range(64701, 64721)]),
        ("budget", root / "budget", list(range(64811, 64823))),
    ]
    result = {"acquisition": {}, "lifetimes": {}}
    for key, d, seeds in series:
        result["acquisition"][key] = acquisition(d, data.n, seeds, data.rows, body)
    if a.placement_capture:
        with np.load(a.placement_capture / "common.npz") as z:
            n = z["native"]
        result["acquisition"]["placement"] = acquisition(
            root / "placement", n, range(65001, 65013), data.rows, body
        )
    for name, d, seeds in (
        ("life", primary, series[0][2]),
        ("credit", root / "credit", series[2][2]),
    ):
        result["lifetimes"][name] = lives(d, seeds, name, data, body)
    if (root / "routing" / "confirm.json").exists():
        from experiments.internal_body.routing import identity_or_layout

        nlife = nturn = identical = 0
        for seed in range(64701, 64721):
            d = root / "routing"
            info = json.loads((d / f"routing-{seed}.json").read_bytes())
            env_seed = info["environment_seed"]
            partition = np.asarray(info["partition"])
            assert np.array_equal(np.bincount(partition), [12, 12])
            with (
                np.load(d / f"routing-{seed}-raw.npz") as z,
                np.load(d / f"routing-{seed}-parameters.npz") as ck,
            ):
                for j, world in enumerate(WORLDS):
                    tape = life_tape(env_seed, world, data.rows)
                    for kind in ("id", "layout"):
                        tr = z[f"w{j}-{kind}"]
                        physical_trace(tr, tape, data, world, 0, body)
                        nlife += 1
                        nturn += len(tr)
                        if seed in (64701, 64720):
                            agent = identity_or_layout(
                                data, root / "credit", seed, kind, partition
                            )
                            replayed, _ = run_one(agent, env_seed, world, 0, tape, body)
                            np.testing.assert_array_equal(replayed, tr)
                            np.testing.assert_array_equal(
                                agent.W, ck[f"w{j}-live-{kind}"]
                            )
                            identical += 1
        result["lifetimes"]["routing"] = {
            "lives": nlife,
            "turns": nturn,
            "bitwise_restored_final_checkpoints": identical,
        }
    a.output.mkdir(exist_ok=True, parents=True)
    with (
        (a.output / "full-qwen-encounters.jsonl").open("w", encoding="utf-8") as f,
        np.load(a.gpu / "lifetimes.npz") as tr,
    ):
        total = 0
        for r in json.loads((a.gpu / "lives.json").read_bytes()):
            t = tr[r["key"]]
            total += len(t)
            physical_trace(
                t,
                life_tape(r["seed"], tuple(r["world"]), data.rows),
                data,
                tuple(r["world"]),
                r["reversal"],
                body,
                export=f,
                key=r["key"],
            )
    result["actual_full_qwen_turns_replayed"] = total
    write(a.output / "audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
