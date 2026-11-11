"""Run the locked acquisition, paired fits and out-of-training-context tests."""

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from experiments.consequence_policy.environment import (
    draw_tape,
    settled_tape,
    shuffled_feedback,
)
from experiments.consequence_policy.policy import (
    Policy,
    deployed,
    features,
    feedback_scores,
    fit,
    fit_projection,
)
from experiments.consequence_policy.study import (
    ARMS,
    CONFIRM,
    PARTNERS,
    PILOT,
    TEST_H,
    TRAIN_H,
    contexts,
)
from experiments.direct_policy.protocol import COSTS, probabilities, rng, write
from experiments.social_closed_loop.core import BodyValue, body_event


def read_inputs(cache, output):
    with np.load(cache / "qwen-states.npz", allow_pickle=False) as data:
        hidden, native = data["hidden"], data["logits"]
    rows = contexts()
    logged = json.loads((cache / "contexts.json").read_bytes())
    assert len(rows) == len(logged) == len(hidden) == len(native)
    assert all(all(a[k] == b[k] for k in a) for a, b in zip(rows, logged))
    train = np.array([r["split"] == "train" for r in rows])
    if (output / "projection.npz").exists():
        with np.load(output / "projection.npz", allow_pickle=False) as data:
            center, matrix = data["center"], data["matrix"]
    else:
        center, matrix = fit_projection(hidden, train)
        np.savez_compressed(output / "projection.npz", center=center, matrix=matrix)
    x = features(hidden, center, matrix)
    partner = np.array([r["partner"] for r in rows])
    return rows, native, x, partner, train


def evaluation_reference(body, rows, mapping):
    # Independent simulation for reporting. No element of this function is passed to fit().
    q = ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * rng(63998, "risk-quadrature").random(
        (32768, 5)
    )
    noise = 0.1 * ndtri(q)
    risk = np.empty((len(TEST_H), len(rows), 4))
    value = np.empty_like(risk)
    favorable = np.empty(len(rows), dtype=int)
    for hi, health in enumerate(TEST_H):
        risk_response = np.zeros((4, 2))
        for a, c in enumerate(COSTS):
            after_action = body_event(health, c, noise[:, 0])
            for response in range(2):
                nxt = after_action
                for j in range(1, 5):
                    nxt = body_event(nxt, 1.0 - response, noise[:, j])
                risk_response[a, response] = np.mean(nxt == 0.0)
        for i, row in enumerate(rows):
            p = probabilities(PARTNERS[row["partner"]], row["frame"], mapping)
            favorable[i] = 2 if p[1] > p[2] else 3
            p4 = np.array([0.0, *p])
            risk[hi, i] = (1.0 - p4) * risk_response[:, 0] + p4 * risk_response[:, 1]
            value[hi, i] = body.values(health, p) - body.rul(health)
    return risk, value, favorable


def metrics(pi, reference):
    risk, value, favorable = reference
    answers = pi[:, :, 2:].sum(-1)
    style = pi[:, np.arange(len(favorable)), favorable]
    deaths = (pi * risk).sum(axis=-1)
    utility = (pi * value).sum(axis=-1)
    healthy = [TEST_H.index(h) for h in (90.0, 300.0)]
    chosen = pi.argmax(axis=-1)
    return {
        "health": list(TEST_H),
        "answer": answers.mean(axis=1).tolist(),
        "death": deaths.mean(axis=1).tolist(),
        "effective_answer": style.mean(axis=1).tolist(),
        "expected_reference_change": utility.mean(axis=1).tolist(),
        "actions": pi.mean(axis=1).tolist(),
        "greedy_answers": (chosen >= 2).mean(axis=1).tolist(),
        "danger_answer": float(answers[0].mean()),
        "healthy_answer": float(answers[healthy].mean()),
        "healthy_style": float(style[healthy].mean()),
        "gap": float(answers[healthy].mean() - answers[0].mean()),
    }


def fit_seed(
    seed, mapping, tape, rows, native, x, partner, train, body, reference, output
):
    chosen, mu, partner_u, noise = tape
    training_rows = [r for r in rows if r["split"] == "train"]
    after, reward, response = settled_tape(
        chosen, partner_u, noise, training_rows, mapping, body
    )
    shuffled, permutation = shuffled_feedback(reward, seed)
    assert np.array_equal(np.sort(shuffled, axis=-1), np.sort(reward, axis=-1))
    scale = np.minimum(1.0, [body.rul(h) for h in TRAIN_H])
    scores = {
        key: feedback_scores(chosen, r, mu, scale)
        for key, r in (("real", reward), ("shuffle", shuffled))
    }
    folder = output / f"s{seed}-m{mapping}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "outcomes.npz",
        health_after=after,
        reward=reward,
        response=response,
        shuffle_indices=permutation,
    )
    results, histories, probes = {}, {}, {}
    for arm in ARMS:
        policy = Policy(x.shape[1], blind=arm == "blind")
        if arm != "frozen":
            histories[arm] = fit(
                policy,
                native[train],
                x[train],
                partner[train],
                TRAIN_H,
                scores["shuffle" if arm == "shuffle" else "real"],
            )
        policy.save(folder / f"{arm}.npz")
        logits = policy.grid_logits(native[~train], x[~train], partner[~train], TEST_H)
        pi = deployed(logits)
        probes[arm] = pi
        results[arm] = metrics(pi, reference)
        if arm == "real":
            # A constant, pooled sensor for the same saved weights and same words at all healths.
            masked = Policy(x.shape[1], blind=True)
            gate = policy.gates(TRAIN_H).mean(axis=0)
            masked.weights[:, 0] = np.einsum("b,pbda->pda", gate, policy.weights)
            masked_pi = deployed(
                masked.grid_logits(native[~train], x[~train], partner[~train], TEST_H)
            )
            probes["sensor_masked"] = masked_pi
            results["sensor_masked"] = metrics(masked_pi, reference)
    np.savez_compressed(folder / "probe-probabilities.npz", **probes)
    write(
        folder / "optimization.json",
        {
            "columns": ["step", "loss", "centered_gain", "kl", "gradient_norm"],
            "arms": histories,
        },
    )
    audits = []
    for hi, health in enumerate(TRAIN_H):
        for a in range(4):
            select = chosen[hi] == a
            audits.append(
                {
                    "health": health,
                    "action": a,
                    "n": int(select.sum()),
                    "maintenance": int((response[hi][select] == 1).sum()),
                    "deaths": int((after[hi][select] == 0).sum()),
                    "reward_mean": float(reward[hi][select].mean()),
                }
            )
    write(
        folder / "metrics.json",
        {"seed": seed, "mapping": mapping, "metrics": results, "data_audit": audits},
    )
    return results


def contrasts(arms):
    real, frozen, shuffled, blind = [
        arms[k] for k in ("real", "frozen", "shuffle", "blind")
    ]
    return {
        "danger_vs_frozen": frozen["danger_answer"] - real["danger_answer"],
        "danger_vs_shuffle": shuffled["danger_answer"] - real["danger_answer"],
        "healthy_vs_frozen": real["healthy_answer"] - frozen["healthy_answer"],
        "gap_vs_blind": real["gap"] - blind["gap"],
        "style_vs_shuffle": real["healthy_style"] - shuffled["healthy_style"],
    }


def summarize(output, seeds):
    paired = []
    for seed in seeds:
        mirrors = [
            contrasts(
                json.loads((output / f"s{seed}-m{m}/metrics.json").read_bytes())[
                    "metrics"
                ]
            )
            for m in (0, 1)
        ]
        paired.append({k: np.mean([d[k] for d in mirrors]) for k in mirrors[0]})
    thresholds = {
        "danger_vs_frozen": 0.10,
        "danger_vs_shuffle": 0.10,
        "healthy_vs_frozen": -0.05,
        "gap_vs_blind": 0.10,
        "style_vs_shuffle": 0.10,
    }
    bootstrap = rng(63997, "paired-bootstrap").integers(
        len(seeds), size=(12000, len(seeds))
    )
    stats = {}
    for k, threshold in thresholds.items():
        a = np.array([r[k] for r in paired])
        ci = np.quantile(a[bootstrap].mean(axis=1), [0.025, 0.975]).tolist()
        directions = int((a > 0).sum())
        passed = a.mean() >= threshold
        if len(seeds) == len(CONFIRM) and k != "healthy_vs_frozen":
            passed = passed and directions >= 16 and ci[0] > 0
        stats[k] = {
            "mean": float(a.mean()),
            "ci95": ci,
            "positive": directions,
            "range": [float(a.min()), float(a.max())],
            "threshold": threshold,
            "passed": bool(passed),
        }
    return {
        "seeds": list(seeds),
        "paired": paired,
        "stats": stats,
        "passed": all(s["passed"] for s in stats.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave", choices=["pilot", "confirm"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = PILOT if args.wave == "pilot" else CONFIRM
    if args.wave == "confirm":
        assert json.loads((args.output / "pilot.json").read_bytes())["passed"], (
            "Pilot gate failed."
        )
    rows, native, x, partner, train = read_inputs(args.cache, args.output)
    test_rows = [r for r in rows if r["split"] == "test"]
    body = BodyValue()
    refs = [evaluation_reference(body, test_rows, m) for m in (0, 1)]
    started = time.monotonic()
    for seed in seeds:
        tape = draw_tape(seed, native[train])
        # Environment uniforms/noises are exact deterministic functions of seed and the pinned generator.
        # Store the actual choices and propensities once for the mirrors, each realized outcome separately.
        np.savez_compressed(
            args.output / f"s{seed}-behavior.npz",
            action=tape[0],
            mu=tape[1],
            exogenous_sha256=np.array(
                sha256(tape[2].tobytes() + tape[3].tobytes()).hexdigest()
            ),
        )
        for mapping in (0, 1):
            fit_seed(
                seed,
                mapping,
                tape,
                rows,
                native,
                x,
                partner,
                train,
                body,
                refs[mapping],
                args.output,
            )
        print(
            json.dumps(
                {
                    "finished_seed": seed,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )
    report = summarize(args.output, seeds)
    write(args.output / f"{args.wave}.json", report)
    print(json.dumps(report["stats"]), flush=True)


if __name__ == "__main__":
    main()
