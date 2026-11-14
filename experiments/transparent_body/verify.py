"""Independent affine/noise checks and read-only archived predictor routing."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.special import expit, ndtr, ndtri

from experiments.frozen_capability.design import serial
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT
from experiments.partner_prediction.design import FEATURE_IDS, Forecast

from .body import WORDS, BodyCalculator, PersonalForecast
from .interface import messages

SOURCE = "artifacts/model-studies/2026-11-14/forecast-stages/gpu/first/results.zip"
SOURCE_SHA = "ed728c7b04f8bf9775cfa9dff95870ee96e71315ab594ef438c42d96149b7bdd"


def affine_final(h, cost, maintained, noise):
    """Independent closed composition; no calls to calculator or settle."""
    costs = np.array([cost] + [0.0 if maintained else 1.0] * 4)
    c = costs[None, :] * (1 + noise)
    multipliers = 1 + c / 350.0
    suffix = np.cumprod(multipliers[:, ::-1], axis=1)[:, ::-1]
    later = np.column_stack((suffix[:, 1:], np.ones(len(noise))))
    return np.maximum(0.0, h * suffix[:, 0] - (2 * c * later).sum(axis=1))


def physics_checks():
    gen = np.random.default_rng(926001)
    n = 200000
    noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * gen.random((n, 5)))
    records = []
    for h in (2.0, 3.5, 6.0, 8.25, 8.5, 12.0, 90.0, 350.0):
        a = BodyCalculator(nodes=11).conditional(h)
        b = BodyCalculator(nodes=15).conditional(h)
        for j, cost in enumerate((0.0, 0.25, 1.75)):
            for maintained in (False, True):
                if j == 0 and maintained:
                    continue
                word, branch = (
                    WORDS[j],
                    "maintenance" if maintained else "no_maintenance",
                )
                out = affine_final(h, cost, maintained, noise)
                p = float((out == 0).mean())
                q11, q15 = a[word][branch], b[word][branch]
                delta = abs(q11["p_death"] - q15["p_death"])
                error = abs(q15["p_death"] - p)
                formal_error = abs(q11["p_death"] - p)
                # Predeclared numerical acceptance, not a behavior success gate.
                if delta > 0.01 or error > 0.01 or formal_error > 0.01:
                    raise AssertionError((h, word, branch, delta, error, formal_error))
                records.append(
                    {
                        "health": h,
                        "action": word,
                        "branch": branch,
                        "risk_11": q11["p_death"],
                        "risk_15": q15["p_death"],
                        "mc_risk": p,
                        "quadrature_difference": delta,
                        "mc_difference": error,
                        "formal_11_mc_difference": formal_error,
                        "mean_health_difference": abs(
                            q15["expected_health"] - float(out.mean())
                        ),
                        "mc_standard_error": float(np.sqrt(p * (1 - p) / n)),
                    }
                )
    return {
        "sample_count": n,
        "seed": 926001,
        "independent_distribution": "truncated normal, not clipped normal",
        "uses_environment_random_tape": False,
        "rows": records,
        "max_quadrature_risk_difference": max(
            r["quadrature_difference"] for r in records
        ),
        "max_mc_risk_difference": max(r["mc_difference"] for r in records),
        "max_formal_11_mc_risk_difference": max(
            r["formal_11_mc_difference"] for r in records
        ),
    }


def archived_inputs(repo):
    """Verify saved head predictions and construct prompts, without calling Qwen.

    Eight old lives are NOT eight new repetitions. All early/middle/late inputs
    are checked; the compact demonstration pack uses birth and last living head.
    No output of an optimal controller is used as a model decision.
    """
    path = repo / SOURCE
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("saved forecast archive changed")
    calc = BodyCalculator(nodes=11)
    maximum_error, prompts, keys, receipts = 0.0, [], {}, []
    with ZipFile(path) as z:
        with np.load(BytesIO(z.read("output/features.npz"))) as f:
            ids, xx = f["ids"], f["x"]
            if list(ids) != FEATURE_IDS:
                raise ValueError("feature identity mismatch")
            x = np.full((len(ITEMS), 9), np.nan)
            x[ids] = xx
        for seed in PILOT:
            with np.load(BytesIO(z.read(f"output/lives/s{seed}/states.npz"))) as s:
                weights, selected = s["head"], s["selected"]
                counts = s["observed"].copy()
                hashes = [sha256(v.tobytes()).hexdigest() for v in weights]
                w = weights.copy()
            with np.load(BytesIO(z.read(f"output/lives/s{seed}/stages.npz"))) as st:
                archived = st["q"][..., 0, :]
            actual = np.array(
                [[expit(x[ix] @ w[t].T) for ix in PANEL] for t in selected]
            )
            error = float(abs(actual - archived).max())
            maximum_error = max(maximum_error, error)
            if error > 1e-12:
                raise AssertionError("head/feature prediction replay differs")
            # Replay actual observed responses, excluding terminal feedback.
            factual = [
                json.loads(line)
                for line in z.read(f"output/lives/s{seed}/replay.jsonl").splitlines()
            ]
            head = Forecast("true")
            head_error = 0.0
            for t, e in enumerate(factual):
                difference = float(abs(head.w - w[t]).max())
                head_error = max(head_error, difference)
                if difference > 1e-12:
                    raise AssertionError("individual parameter chronology mismatch")
                if t < len(factual) - 1:
                    head.observe(x, e["row"], e["action"], e["maintenance"])
            receipts.append(
                {
                    "seed": seed,
                    "decisions": len(w),
                    "nonterminal_observations": int(counts[-1]),
                    "all_checkpoints_rebuilt": True,
                    "max_prediction_error": error,
                    "max_parameter_replay_error": head_error,
                }
            )
            # One fixed task and frame; two partners and wordings. Risk panel only.
            for stage, t in (("birth", 0), ("late", int(selected[-1]))):
                for wording in (0, 1):
                    for j in (4, 20):
                        i = int(PANEL[wording, j])
                        r = ITEMS[i]
                        assert r["task"] == 5 and r["frame"] == 0 and r["layout"] == 0
                        p = tuple(float(v) for v in expit(w[t] @ x[i]))
                        forecast = PersonalForecast(
                            f"s{seed}",
                            t,
                            t + 1,
                            hashes[t],
                            p,
                            "frozen" if stage == "birth" else "personal",
                        )
                        for h in (2.0, 6.0, 90.0):
                            detail = calc.calculate(h, forecast)
                            for order in (
                                tuple((k + (seed - 77101) % 4) % 4 for k in range(4)),
                            ):
                                ident = f"s{seed}-{stage}-w{wording}-p{r['partner']}-h{h:g}-o{order[0]}"
                                prompt = messages(detail, r["text"], "January", order)
                                prompts.append({"id": ident, "messages": prompt})
                                # Audit-side only; never append these to the model input.
                                keys[ident] = {
                                    "seed": seed,
                                    "stage": stage,
                                    "decision_turn": t + 1,
                                    "partner": r["partner"],
                                    "row": i,
                                    "wording": wording,
                                    "order": order,
                                    "forecast_sha256": hashes[t],
                                    "probabilities": p,
                                    "calculation": detail,
                                }
    return (
        {
            "source": str(SOURCE),
            "source_sha256": sha256(path.read_bytes()).hexdigest(),
            "lives": receipts,
            "maximum_prediction_error": maximum_error,
            "prompt_count": len(prompts),
            "qwen_calls": 0,
            "new_lives": 0,
        },
        prompts,
        keys,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    physical = physics_checks()
    replay, prompts, keys = archived_inputs(a.repo)
    for name, obj in (
        ("physics-checks.json", physical),
        ("archive-replay.json", replay),
        ("example-prompts.json", prompts),
        ("audit-only-calculations.json", keys),
    ):
        (a.out / name).write_bytes(serial(obj))
    print(json.dumps({k: v for k, v in physical.items() if k not in ("rows",)}))
    print(json.dumps(replay))


if __name__ == "__main__":
    main()
