"""Replay physical events and independently aggregate population and lifetime interventions."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.consequence_policy.coverage import RANGES, SPLIT, SPLIT_WAVES, WAVES
from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.experiment import (
    contrasts,
    evaluation_reference,
    metrics,
    read_inputs,
)
from experiments.consequence_policy.lifetime import (
    ARMS,
    PersonalActor,
    environment_tape,
    post,
)
from experiments.consequence_policy.policy import Policy, deployed, features
from experiments.consequence_policy.study import (
    CONFIRM,
    PARTNERS,
    PILOT,
    TEST_H,
    TRAIN_H,
)
from experiments.direct_policy.protocol import COSTS, probabilities, rng, settle, write
from experiments.social_closed_loop.core import BodyValue


def read(path):
    return json.loads(path.read_bytes())


def acquisition(output, native, rows, body):
    counts = {"choices": 0, "physical_settlements": 0, "individually_replayed": 0}
    series = [("s", PILOT + CONFIRM, {"": TRAIN_H})]
    for prefix, waves, ranges in (
        ("coverage-", WAVES, RANGES),
        ("allocation-", SPLIT_WAVES, SPLIT),
    ):
        for stage, seeds in waves.items():
            if (output / f"{prefix}{stage}.json").exists():
                series.append((prefix, seeds, ranges))
    for prefix, seeds, ranges in series:
        extension = prefix != "s"
        for seed in seeds:
            chosen, mu, partner_u, noise = draw_tape(seed, native)
            with np.load(output / f"{prefix}{seed}-behavior.npz") as tape:
                np.testing.assert_array_equal(tape["action"], chosen)
                np.testing.assert_array_equal(tape["mu"], mu)
                assert (
                    str(tape["exogenous_sha256"])
                    == sha256(partner_u.tobytes() + noise.tobytes()).hexdigest()
                )
            counts["choices"] += chosen.size
            for m in (0, 1):
                p = np.array(
                    [
                        [0.0, *probabilities(PARTNERS[r["partner"]], r["frame"], m)]
                        for r in rows
                    ]
                )
                acceptance = partner_u < p[np.arange(len(rows))[None, :, None], chosen]
                actual = np.where(chosen == 0, -1, acceptance)
                path = (
                    output
                    / f"{prefix}{seed}-m{m}"
                    / ("raw.npz" if extension else "outcomes.npz")
                )
                with np.load(path) as raw:
                    np.testing.assert_array_equal(raw["response"], actual)
                    for arm, levels in ranges.items():
                        before = np.broadcast_to(
                            np.array(levels)[:, None, None], chosen.shape
                        )
                        h = before.copy()
                        for j in range(5):
                            damage = (
                                np.asarray(COSTS)[chosen] if j == 0 else (actual != 1)
                            )
                            h = np.maximum(
                                0.0,
                                h - damage * (2.0 - h / 350.0) * (1.0 + noise[..., j]),
                            )
                        stored = raw[arm + "_after" if extension else "health_after"]
                        np.testing.assert_allclose(stored, h, rtol=0, atol=1e-12)
                        r = body.interpolate(h, body.reference) - body.interpolate(
                            before, body.reference
                        )
                        np.testing.assert_allclose(
                            raw[arm + "_reward" if extension else "reward"],
                            r,
                            rtol=0,
                            atol=1e-12,
                        )
                        for index in np.linspace(0, chosen.size - 1, 24, dtype=int):
                            ix = np.unravel_index(index, chosen.shape)
                            accepted = None if chosen[ix] == 0 else bool(actual[ix])
                            final, events = settle(
                                before[ix], int(chosen[ix]), accepted, noise[ix]
                            )
                            assert abs(final - stored[ix]) < 1e-12 and len(events) == 6
                            counts["individually_replayed"] += 1
                        counts["physical_settlements"] += chosen.size
    return counts


def lives(output, rows, native, x):
    nlife = nturn = 0
    interventions = []
    for seed in PILOT + CONFIRM:
        runs = read(output / f"life-{seed}.json")
        with np.load(output / f"life-{seed}-traces.npz") as archive:
            for run in runs:
                world, rep = tuple(run["world"]), run["replica"]
                ids, u, noise = environment_tape(seed, world, rep, rows)
                for arm in ARMS:
                    trace = archive[f"w{world[0]}{world[1]}-r{rep}-{arm}"]
                    t = len(trace)
                    assert trace[0, 1] == 350.0 and (t == 256 or trace[-1, 4] == 0.0)
                    np.testing.assert_array_equal(trace[1:, 1], trace[:-1, 4])
                    np.testing.assert_array_equal(ids[:t], trace[:, 0])
                    np.testing.assert_allclose(
                        trace[:, 6:10].sum(axis=1), 1.0, atol=3e-7, rtol=0
                    )
                    np.testing.assert_array_equal(
                        (u[:t, :1] > trace[:, 6:10].cumsum(axis=1)[:, :3]).sum(axis=1),
                        trace[:, 2],
                    )
                    for j, r in enumerate(trace):
                        i, a = int(r[0]), int(r[2])
                        p, f = rows[i]["partner"], rows[i]["frame"]
                        prob = probabilities(PARTNERS[p], f, world[p])
                        accepted = bool(u[j, 1] < prob[a - 1]) if a else None
                        h, _ = settle(r[1], a, accepted, noise[j])
                        assert r[3] == (-1 if accepted is None else int(accepted))
                        assert abs(r[4] - h) < 1e-10
                    assert t == run["arms"][arm]["turns"]
                    assert (trace[:, 2] >= 2).sum() == run["arms"][arm]["tasks"]
                    nlife += 1
                    nturn += t
                if seed in CONFIRM:
                    a = PersonalActor(output, seed, world, "personal")
                    own = run["arms"]["personal"]
                    coeff = np.asarray(own["final_u"])
                    swapped = post(a, native, x, rows, world, u=coeff[::-1])
                    restored = post(a, native, x, rows, world, u=coeff)
                    assert (
                        abs(own["post"]["healthy_style"] - restored["healthy_style"])
                        < 1e-7
                    )
                    interventions.append(
                        {
                            "seed": seed,
                            "mixed": world[0] != world[1],
                            "actual": restored["healthy_style"],
                            "exchanged": swapped["healthy_style"],
                        }
                    )
    effects = {
        kind: float(
            np.mean(
                [
                    r["actual"] - r["exchanged"]
                    for r in interventions
                    if r["mixed"] == condition
                ]
            )
        )
        for kind, condition in (("mixed", True), ("matched", False))
    }
    return {
        "complete_lives": nlife,
        "turns": nturn,
        "partner_exchange": effects,
        "interventions": interventions,
    }


def fresh(output, gpu, rows, native, test, body):
    singleton = np.load(gpu / "independent-test-prefixes.npz")
    assert np.array_equal(singleton["context"], np.flatnonzero(test))
    with np.load(output / "projection.npz") as proj:
        x = features(singleton["hidden"], proj["center"], proj["matrix"])
    test_rows = [r for r, keep in zip(rows, test) if keep]
    partners = np.array([r["partner"] for r in test_rows])
    new_logits = singleton["logits"]
    references = [evaluation_reference(body, test_rows, m) for m in (0, 1)]
    stats = []
    layouts = {i: [] for i in range(24)}
    diffs = []
    fresh_by_seed = []
    for seed in CONFIRM:
        mirrors = []
        for m in (0, 1):
            folder = output / f"s{seed}-m{m}"
            scores = {}
            with np.load(folder / "probe-probabilities.npz") as archived:
                for arm in ("real", "shuffle", "frozen", "blind"):
                    actor = Policy.load(folder / (arm + ".npz"))
                    pi = deployed(actor.grid_logits(new_logits, x, partners, TEST_H))
                    scores[arm] = metrics(pi, references[m])
                    gap = np.max(np.abs(archived[arm] - pi), axis=-1)
                    diffs.append(gap.ravel())
                for layout, observations in layouts.items():
                    mask = np.array([r["layout"] == layout for r in test_rows])
                    low = (
                        archived["frozen"][0, mask, 2:].sum(-1).mean()
                        - archived["real"][0, mask, 2:].sum(-1).mean()
                    )
                    high = (
                        archived["real"][-2:, mask, 2:].sum(-1).mean()
                        - archived["frozen"][-2:, mask, 2:].sum(-1).mean()
                    )
                    observations.append([float(low), float(high)])
            mirrors.append(contrasts(scores))
            fresh_by_seed.append({"seed": seed, "mapping": m, "arms": scores})
        stats.append({k: float(np.mean([r[k] for r in mirrors])) for k in mirrors[0]})
    write(output / "fresh-checkpoints.json", fresh_by_seed)
    old_stats = read(output / "confirm.json")["stats"]
    boot = rng(63997, "paired-bootstrap").integers(
        len(CONFIRM), size=(12000, len(CONFIRM))
    )
    primary = {}
    for k, old in old_stats.items():
        a = np.array([r[k] for r in stats])
        low, high = np.quantile(a[boot].mean(axis=1), [0.025, 0.975])
        passed = a.mean() >= old["threshold"] and (
            k == "healthy_vs_frozen" or (low > 0 and (a > 0).sum() >= 16)
        )
        primary[k] = {
            "mean": float(a.mean()),
            "ci95": [low, high],
            "positive": int((a > 0).sum()),
            "passed": bool(passed),
        }
    comparison = []
    with np.load(gpu / "end-to-end-lives.npz") as actual:
        for r in read(gpu / "live-metrics.json"):
            with np.load(output / f"life-{r['seed']}-traces.npz") as reference:
                w = r["world"]
                old = reference[f"w{w[0]}{w[1]}-r0-{r['arm']}"]
            new = actual[r["run"]]
            n = min(len(old), len(new))
            deviations = np.flatnonzero(new[:n, 2] != old[:n, 2])
            first = int(deviations[0]) if len(deviations) else n
            comparison.append(
                {
                    "run": r["run"],
                    "cached_turns": len(old),
                    "full_turns": len(new),
                    "cached_tasks": int(sum(old[:, 2] >= 2)),
                    "full_tasks": int(sum(new[:, 2] >= 2)),
                    "first_different_action": (first + 1 if first < n else None),
                    "max_prob_through_first_choice": float(
                        np.max(
                            np.abs(
                                old[: min(n, first + 1), 6:10]
                                - new[: min(n, first + 1), 6:10]
                            )
                        )
                    ),
                }
            )
    layout_means = {str(i): np.mean(a, axis=0).tolist() for i, a in layouts.items()}
    delta = np.concatenate(diffs)
    return {
        "primary": primary,
        "primary_passed": all(r["passed"] for r in primary.values()),
        "complete_lives": comparison,
        "layout_means": layout_means,
        "layout_columns": ["fatal_answer_reduction", "healthy_answer_increase"],
        "checkpoint_probability_gap": {
            "max": float(delta.max()),
            "mean": float(delta.mean()),
            "p95": float(np.quantile(delta, 0.95)),
        },
        "native_probability_shift_max": float(
            np.abs(deployed(new_logits / 2.0) - deployed(native[test] / 2.0)).max()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=Path)
    args = parser.parse_args()
    rows, native, x, _, train = read_inputs(args.cache, args.output)
    body = BodyValue()
    if args.gpu:
        result = fresh(args.output, args.gpu, rows, native, ~train, body)
        write(args.output / "fresh-audit.json", result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "primary_passed",
                        "checkpoint_probability_gap",
                        "complete_lives",
                    )
                }
            )
        )
    else:
        acquisition_counts = acquisition(
            args.output, native[train], [r for r in rows if r["split"] == "train"], body
        )
        result = lives(args.output, rows, native, x)
        write(
            args.output / "replay-audit.json",
            {"acquisition": acquisition_counts, "lives": result},
        )
        print(
            json.dumps(
                {
                    "acquisition": acquisition_counts,
                    "lives": result["complete_lives"],
                    "turns": result["turns"],
                    "exchange": result["partner_exchange"],
                }
            )
        )


if __name__ == "__main__":
    main()
