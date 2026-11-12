"""Paired actual-action acquisition, matched suffix optimization, and seed-level tests."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.environment import (
    draw_tape,
    settled_tape,
    shuffled_feedback,
)
from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.policy import (
    Policy,
    deployed,
    features,
    feedback_scores,
    fit_projection,
)
from experiments.consequence_policy.policy import fit as fit5
from experiments.consequence_policy.study import TEMP, TEST_H
from experiments.direct_policy.protocol import rng, write
from experiments.internal_body.design import CONFIRM, HIGH_H, PILOT, SPLIT_H
from experiments.internal_body.suffix import (
    InternalPolicy,
    StateBank,
    distribution,
    fit,
)
from experiments.social_closed_loop.core import BodyValue

ARMS = (
    "linear",
    "nonlinear",
    "postnorm",
    "blind",
    "no_route",
    "shuffle",
    "zero",
    "maintain",
    "high",
    "prompt_lora",
    "native_prompt",
    "native_hidden",
)
WAVES = {"pilot": PILOT, "confirm": CONFIRM}


def raw_score(action, r, mu, scale):
    return np.stack(
        [
            np.mean((action == a) * r / scale[:, None, None], axis=-1) / mu[:, a]
            for a in range(4)
        ],
        axis=-1,
    )


def evaluate(model, bank, ix):
    grid = bank.grid(ix, TEST_H, model.uses_text)
    with torch.no_grad():
        return distribution(model.scores(grid)).cpu().numpy()


def acquire(seed, bank, train, body, out):
    action, mu, partner_u, noise = draw_tape(seed, bank.native[train])
    sources = [bank.rows[i] for i in train]
    np.savez_compressed(
        out / f"behavior-{seed}.npz",
        action=action,
        mu=mu,
        partner_uniform=partner_u,
        body_noise=noise,
    )
    items = []
    for m in (0, 1):
        observed = {}
        for name, levels in (("split", SPLIT_H), ("high", HIGH_H)):
            h, r, response = settled_tape(
                action, partner_u, noise, sources, m, body, levels
            )
            if name == "high":
                assert np.array_equal(observed["split"][2], response)
            observed[name] = (h, r, response)
        h, r, response = observed["split"]
        shuffle, perm = shuffled_feedback(r, seed)
        scale = np.minimum(1.0, [body.rul(a) for a in SPLIT_H])
        high_scale = np.minimum(1.0, [body.rul(a) for a in HIGH_H])
        scores = {
            "real": feedback_scores(action, r, mu, scale),
            "shuffle": feedback_scores(action, shuffle, mu, scale),
            "zero": raw_score(action, r, mu, scale),
            "maintain": feedback_scores(
                action, (response == 1).astype(float), mu, np.ones(len(scale))
            ),
            "high": feedback_scores(action, observed["high"][1], mu, high_scale),
        }
        folder = out / f"s{seed}-m{m}"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / "outcomes.npz",
            after=h,
            reward=r,
            response=response,
            high_after=observed["high"][0],
            high_reward=observed["high"][1],
            permutation=perm,
        )
        audit = [
            {
                "health": level,
                "action": a,
                "n": int((action[j] == a).sum()),
                "protected": int(((action[j] == a) & (response[j] == 1)).sum()),
                "deaths": int(((action[j] == a) & (h[j] == 0)).sum()),
                "high_deaths": int(
                    ((action[j] == a) & (observed["high"][0][j] == 0)).sum()
                ),
                "mean_reward": float(r[j][action[j] == a].mean()),
            }
            for j, level in enumerate(SPLIT_H)
            for a in range(4)
        ]
        write(folder / "data-audit.json", audit)
        items.append(scores)
    return items


def run_seed(seed, bank, body, train, test, ref, x5, output):
    observed = acquire(seed, bank, train, body, output)
    for m, scores in enumerate(observed):
        folder = output / f"s{seed}-m{m}"
        records, probes, history = {}, {}, {}
        n = bank.native[test]
        probes["frozen"] = np.broadcast_to(
            deployed(n / TEMP), (len(TEST_H), len(test), 4)
        ).copy()
        probes["frozen_prompt"] = (
            distribution(bank.grid(test, TEST_H, True)["n"]).cpu().numpy()
        )
        ref5 = Policy(x5.shape[1])
        history["reference5"] = fit5(
            ref5,
            bank.native[train],
            x5[train],
            bank.partners[train],
            SPLIT_H,
            scores["real"],
        )
        ref5.save(folder / "reference5.npz")
        probes["reference5"] = deployed(
            ref5.grid_logits(n, x5[test], bank.partners[test], TEST_H)
        )
        for arm in ARMS:
            model = InternalPolicy(bank, arm)
            key = arm if arm in scores else "real"
            grid = bank.grid(
                train, HIGH_H if arm == "high" else SPLIT_H, model.uses_text
            )
            history[arm] = fit(model, grid, scores[key])
            model.save(folder / (arm + ".npz"))
            probes[arm] = evaluate(model, bank, test)
            if arm == "linear":
                with torch.no_grad():
                    probes["masked_same_parameters"] = (
                        distribution(model.scores(bank.grid(test, TEST_H), masked=True))
                        .cpu()
                        .numpy()
                    )
        for arm, pi in probes.items():
            records[arm] = metrics(pi, ref[m])
        np.savez_compressed(folder / "probes.npz", **probes)
        write(folder / "metrics.json", records)
        write(folder / "steps.json", history)
    generic = InternalPolicy(bank, "generic")
    history = fit(
        generic,
        bank.grid(train, SPLIT_H),
        (observed[0]["real"] + observed[1]["real"]) / 2,
    )
    generic.save(output / f"generic-{seed}.npz")
    write(output / f"generic-{seed}-steps.json", history)


def effects(arms):
    a = arms["linear"]
    return {
        "fatal_reduction": arms["frozen"]["danger_answer"] - a["danger_answer"],
        "healthy_change": a["healthy_answer"] - arms["frozen"]["healthy_answer"],
        "style_over_shuffle": a["healthy_style"] - arms["shuffle"]["healthy_style"],
        "switch_over_blind": a["gap"] - arms["blind"]["gap"],
    }


def summarize(out, seeds):
    paired = []
    for seed in seeds:
        mirror = [
            effects(json.loads((out / f"s{seed}-m{m}/metrics.json").read_bytes()))
            for m in (0, 1)
        ]
        paired.append({k: float(np.mean([v[k] for v in mirror])) for k in mirror[0]})
    thresholds = {
        "fatal_reduction": 0.1,
        "healthy_change": -0.05,
        "style_over_shuffle": 0.1,
        "switch_over_blind": 0.1,
    }
    boot = rng(64801, "bootstrap").integers(len(seeds), size=(12000, len(seeds)))
    stats = {}
    for k, t in thresholds.items():
        a = np.array([v[k] for v in paired])
        ci = np.quantile(a[boot].mean(axis=1), [0.025, 0.975])
        passed = a.mean() >= t
        if len(seeds) == len(CONFIRM) and k != "healthy_change":
            passed = passed and (a > 0).sum() >= 16 and ci[0] > 0
        stats[k] = {
            "mean": float(a.mean()),
            "ci95": ci.tolist(),
            "positive": int((a > 0).sum()),
            "gate": t,
            "passed": bool(passed),
        }
    return {
        "seeds": list(seeds),
        "clusters": paired,
        "stats": stats,
        "passed": all(v["passed"] for v in stats.values()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--wave", choices=WAVES, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wave == "confirm":
        assert json.loads((args.output / "pilot.json").read_bytes())["passed"]
    all_seeds = WAVES[args.wave]
    seeds = args.seeds or all_seeds
    assert set(seeds) <= set(all_seeds) and len(set(seeds)) == len(seeds)
    bank = StateBank(args.cache, args.device)
    train = np.arange(768)
    test = np.arange(768, 960)
    mask = np.isin(np.arange(len(bank.rows)), train)
    center, matrix = fit_projection(bank.z, mask)
    x5 = features(bank.z, center, matrix)
    body = BodyValue()
    ref = [evaluation_reference(body, [bank.rows[i] for i in test], m) for m in (0, 1)]
    for seed in seeds:
        clock = time.monotonic()
        run_seed(seed, bank, body, train, test, ref, x5, args.output)
        print(
            json.dumps({"seed": seed, "seconds": round(time.monotonic() - clock, 2)}),
            flush=True,
        )
    if all((args.output / f"generic-{s}-steps.json").exists() for s in all_seeds):
        result = summarize(args.output, all_seeds)
        write(args.output / (args.wave + ".json"), result)
        print(json.dumps(result["stats"]), flush=True)


if __name__ == "__main__":
    main()
