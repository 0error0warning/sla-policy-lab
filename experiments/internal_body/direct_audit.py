"""Check the complete direct small-batch training tapes and optional exact training-fit witnesses."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.environment import draw_tape
from experiments.consequence_policy.policy import feedback_scores
from experiments.consequence_policy.study import KL, PARTNERS
from experiments.direct_policy.protocol import COSTS, probabilities, write
from experiments.internal_body.design import SPLIT_H, all_rows
from experiments.internal_body.evidence import stat
from experiments.social_closed_loop.core import BodyValue


def run(stages, seen=None):
    inputs = all_rows()[:768]
    body = BodyValue()
    seeds = set()
    choices = settlements = steps = 0
    verified_seen_vectors = 0
    max_restore = 0.0
    max_noise_difference = 0.0
    for root in stages:
        with np.load(root / "calibration.npz") as z:
            native = z["native"][:768]
        for f in sorted(root.glob("tape-*.npz")):
            seed = int(f.stem.split("-")[1])
            assert seed not in seeds
            seeds.add(seed)
            action, mu, u, noise = draw_tape(seed, native)
            with np.load(f) as z:
                for key, x in (
                    ("choice", action),
                    ("mu", mu),
                    ("response_uniform", u),
                ):
                    np.testing.assert_array_equal(z[key], x)
                archived_noise = z["noise"]
                max_noise_difference = max(
                    max_noise_difference, float(np.max(np.abs(noise - archived_noise)))
                )
                np.testing.assert_allclose(noise, archived_noise, rtol=0, atol=1e-14)
                noise = archived_noise  # Replay the physical draws actually settled on the GPU host.
            choices += action.size
            seq = json.loads((root / f"schedule-{seed}.json").read_bytes())
            assert len(seq) == 32 and all(
                len(i) == 8 and sorted(h) == list(range(8)) for i, h in seq
            )
            for m in (0, 1):
                folder = root / f"s{seed}-m{m}"
                p = np.array(
                    [
                        [0.0, *probabilities(PARTNERS[row["partner"]], row["frame"], m)]
                        for row in inputs
                    ]
                )
                response = np.where(
                    action == 0, -1, u < p[np.arange(768)[None, :, None], action]
                )
                h0 = np.broadcast_to(np.array(SPLIT_H)[:, None, None], action.shape)
                h = h0.copy()
                for j in range(5):
                    c = np.asarray(COSTS)[action] if j == 0 else response != 1
                    h = np.maximum(0.0, h - c * (2 - h / 350.0) * (1 + noise[..., j]))
                with np.load(folder / "observed.npz") as z:
                    np.testing.assert_array_equal(z["response"], response)
                    np.testing.assert_allclose(z["after"], h, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(
                        z["reward"],
                        body.interpolate(h, body.reference)
                        - body.interpolate(h0, body.reference),
                        rtol=0,
                        atol=1e-12,
                    )
                if seen:
                    schedule = np.asarray(seq)
                    ix, hs = schedule[:, 0], schedule[:, 1]
                    with np.load(folder / "observed.npz") as saved:
                        scale = np.minimum(1.0, [body.rul(x) for x in SPLIT_H])
                        score = feedback_scores(action, saved["reward"], mu, scale)[
                            hs, ix
                        ]
                    key = f"s{seed}-m{m}"
                    measurements = json.loads((seen / (key + ".json")).read_bytes())
                    with np.load(seen / (key + ".npz")) as prob:
                        base = prob["frozen"]
                        for site in prob:
                            pi = prob[site]
                            assert pi.shape == (32, 8, 4)
                            np.testing.assert_allclose(pi.sum(-1), 1, rtol=0, atol=3e-7)
                            gain = np.mean(((pi - base) * score).sum(-1))
                            kl = np.mean((pi * np.log(pi / base)).sum(-1))
                            ans = pi[:, :, 2:].sum(-1)
                            compare = {
                                "chosen_tape_gain": gain,
                                "kl_to_frozen": kl,
                                "net_objective_gain": gain - KL * kl,
                                "low_answer": ans[np.isin(hs, [0, 1])].mean(),
                                "healthy_answer": ans[np.isin(hs, [3, 4, 6, 7])].mean(),
                                "max_single_probability_change": np.abs(
                                    pi - base
                                ).max(),
                            }
                            for metric, v in compare.items():
                                np.testing.assert_allclose(
                                    v, measurements[site][metric], rtol=0, atol=1e-6
                                )
                            verified_seen_vectors += pi.shape[0] * pi.shape[1]
                settlements += action.size
                for name, v in json.loads(
                    (folder / "metrics.json").read_bytes()
                ).items():
                    if name == "frozen":
                        continue
                    assert len(v["steps"]) == 32 and len(v["parameter_names"]) == 2
                    assert v["state_A_change_l2"] > 0 and v["state_B_l2"] > 0
                    max_restore = max(max_restore, v["disabled_frozen_max_logit_gap"])
                    steps += 32
    out = {
        "seeds": sorted(seeds),
        "observed_actions": choices,
        "settlements": settlements,
        "actual_optimizer_steps": steps,
        "max_disabled_frozen_logit_gap": max_restore,
        "max_cross_host_regenerated_body_noise_difference": max_noise_difference,
    }
    if seen:
        values = {}
        for method in ("site3", "site15", "site31", "local31"):
            means = []
            for seed in sorted(seeds):
                pair = [
                    json.loads((seen / f"s{seed}-m{m}.json").read_bytes())[method]
                    for m in (0, 1)
                ]
                means.append({k: np.mean([v[k] for v in pair]) for k in pair[0]})
            values[method] = {k: stat([r[k] for r in means]) for k in means[0]}
        out["full_training_fit"] = values
        out["verified_training_state_probability_vectors"] = verified_seen_vectors
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth", nargs="+", type=Path, required=True)
    p.add_argument("--seen", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    v = run(a.depth, a.seen)
    write(a.output, v)
    print(
        json.dumps(
            {
                k: v[k]
                for k in (
                    "seeds",
                    "observed_actions",
                    "actual_optimizer_steps",
                    "max_disabled_frozen_logit_gap",
                )
            }
        )
    )
