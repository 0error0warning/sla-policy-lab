"""Independent scalar-form updates and body equations, plus raw-array score checks.

Does not call PersonalActor, Policy, runner.replay, runner.life, or summary().
Uses the shared archived feature projection and Reference-RUL numerical table;
it is not an independent re-derivation of that table or a full-Qwen execution.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from experiments.consequence_policy.experiment import read_inputs
from experiments.consequence_policy.study import TEST_H
from experiments.consequence_warmstart.design import BLOCKS, selected
from experiments.direct_policy.protocol import rng, write
from experiments.social_closed_loop.core import BodyValue


def gate(health):
    centers = np.linspace(np.log(2.0), np.log(351.0), 12)
    z = (np.log1p(health) - centers) / (0.7 * (centers[1] - centers[0]))
    v = np.exp(-0.5 * z**2)
    return v / v.sum()


def rates(partner, frame, mapping):
    appeal = 0.9 if ((frame == 0) == ((partner == 0) ^ bool(mapping))) else 0.1
    return np.array([0.0, 0.15, appeal, 1.0 - appeal])


class Independent:
    def __init__(self, source, birth, arm):
        with (
            np.load(source / f"s{birth}-m0/real.npz") as a,
            np.load(source / f"s{birth}-m1/real.npz") as b,
        ):
            self.w = (a["weights"] + b["weights"]) / 2.0
            self.d = (a["weights"] - b["weights"]) / 2.0
        self.u = np.zeros(2)
        self.arm = arm
        if arm == "scratch":
            self.w.fill(0)
        if arm != "personal":
            self.d.fill(0)

    def score(self, native, x, p, h):
        g = gate(h).astype(np.float32)
        correction = np.einsum("b,d,bda->a", g, x, self.w[p], optimize=False)
        direction = np.einsum("b,d,bda->a", g, x, self.d[p], optimize=False)
        z = native / 2.0 + correction + self.u[p] * direction
        s = np.exp(z - max(z))
        s /= sum(s)
        return 0.95 * s + 0.0125, s, direction

    def update(self, native, x, p, h, a, outcome):
        pi, s, d = self.score(native, x, p, h)
        if self.arm == "personal":
            derivative = 0.95 * s / pi * (d - np.dot(s, d))
            self.u[p] = np.clip(
                self.u[p]
                + 0.5
                * (outcome + 0.9)
                * derivative[a]
                / (1 + np.dot(pi, derivative**2)),
                -1,
                1,
            )
        elif self.arm != "prior":
            derivative = np.eye(4)[a] - s
            derivative *= 0.95 * s[a] / pi[a]
            g = gate(h)
            norm = max(1.0, 0.25 * np.dot(g, g) * np.dot(x, x))
            self.w[p] += (
                0.7
                * (outcome + 0.9)
                * (g[:, None, None] * x[None, :, None] * derivative[None, None, :])
                / norm
            )

    def weights(self):
        return self.w + self.u[:, None, None, None] * self.d


def physical(health, action, maintained, noise):
    h = health
    for j, eta in enumerate(noise):
        cost = (0.0, 0.25, 1.75, 1.75)[action] if j == 0 else float(not maintained)
        h = max(0.0, h - cost * (2.0 - h / 350.0) * (1.0 + eta))
    return h


def risk_branches():
    uniforms = rng(63998, "risk-quadrature").random((32768, 5))
    noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * uniforms)
    result = np.empty((len(TEST_H), 4, 2))
    for hi, h in enumerate(TEST_H):
        for a, c in enumerate((0.0, 0.25, 1.75, 1.75)):
            for maintenance in (0, 1):
                values = np.full(32768, h)
                for j in range(5):
                    cost = c if j == 0 else 1 - maintenance
                    values = np.maximum(
                        0, values - cost * (2 - values / 350) * (1 + noise[:, j])
                    )
                result[hi, a, maintenance] = np.mean(values == 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    rows, native, x, _, _ = read_inputs(
        args.source / "capture/output", args.source / "results"
    )
    body = BodyValue()
    index = json.loads((args.run / "index.json").read_bytes())
    branch = risk_branches()
    count = replay_steps = 0
    maxima = {"choice_probability": 0.0, "checkpoint_weights": 0.0, "panel_mean": 0.0}
    opportunities, shuffled = [], []
    for case in index:
        key = f"s{case['environment_seed']}-{case['block']}-w{case['world'][0]}{case['world'][1]}"
        for arm, metrics in case["arms"].items():
            ref = Independent(args.source / "results", case["birth_seed"], arm)
            with np.load(args.run / f"{key}-{arm}.npz") as raw:
                trace = raw["trace"]
                seen = {0: ref.weights().copy()}
                for t, event in enumerate(trace):
                    i, h, a, response, after = (
                        int(event[0]),
                        event[1],
                        int(event[2]),
                        int(event[3]),
                        event[4],
                    )
                    row = rows[i]
                    p = row["partner"]
                    pi, _, _ = ref.score(native[i], x[i], p, h)
                    maxima["choice_probability"] = max(
                        maxima["choice_probability"],
                        float(np.max(abs(pi - event[6:10]))),
                    )
                    if (
                        int(
                            np.searchsorted(
                                np.cumsum(pi), raw["uniforms"][t, 0], side="left"
                            )
                        )
                        != a
                    ):
                        raise ValueError("Independent action differs")
                    maintained = (
                        bool(
                            raw["uniforms"][t, 1]
                            < rates(p, row["frame"], case["world"][p])[a]
                        )
                        if a
                        else None
                    )
                    if response != (-1 if maintained is None else int(maintained)):
                        raise ValueError("Partner response differs")
                    np.testing.assert_allclose(
                        physical(h, a, maintained, raw["noises"][t]),
                        after,
                        atol=1e-12,
                        rtol=0,
                    )
                    outcome = (body.rul(after) - body.rul(h)) / min(1.0, body.rul(h))
                    np.testing.assert_allclose(outcome, event[10], atol=1e-12, rtol=0)
                    ref.update(native[i], x[i], p, h, a, outcome)
                    after_pi, _, _ = ref.score(native[i], x[i], p, h)
                    np.testing.assert_allclose(
                        after_pi, event[12:16], atol=2e-6, rtol=0
                    )
                    seen[t + 1] = ref.weights().copy()
                for label, snap in metrics["checkpoints"].items():
                    count_updates = snap["updates"]
                    saved = raw[f"{label}-weights"]
                    err = float(np.max(abs(seen[count_updates] - saved)))
                    maxima["checkpoint_weights"] = max(
                        maxima["checkpoint_weights"], err
                    )
                    if err > 2e-6:
                        raise ValueError("Independent checkpoint differs")
                    for part, tw in zip(("live", "heldout"), BLOCKS[case["block"]]):
                        ids = selected(rows, tw)
                        probabilities = raw[f"{label}-{part}"]
                        good, risks = [], []
                        for j, i in enumerate(ids):
                            r = rows[i]
                            p = rates(
                                r["partner"], r["frame"], case["world"][r["partner"]]
                            )
                            favored = 2 if p[2] > p[3] else 3
                            good.append(probabilities[:, j, favored])
                            risks.append(
                                (
                                    probabilities[:, j]
                                    * (
                                        (1 - p)[None] * branch[:, :, 0]
                                        + p[None] * branch[:, :, 1]
                                    )
                                ).sum(1)
                            )
                        calculated = {
                            "style": np.mean(good, axis=0),
                            "death": np.mean(risks, axis=0),
                            "answer": probabilities[:, :, 2:].sum(-1).mean(1),
                        }
                        for name, value in calculated.items():
                            err = float(
                                np.max(
                                    abs(value - np.array(snap["panels"][part][name]))
                                )
                            )
                            maxima["panel_mean"] = max(maxima["panel_mean"], err)
                            if err > 2e-7:
                                raise ValueError("Independent panel scores differ")
                if arm == "warmstart_adapter":
                    for label, data in metrics["replays"].items():
                        n = metrics["checkpoints"][label]["updates"]
                        order = np.array(data["shuffle"]["permutation"], dtype=int)
                        np.testing.assert_array_equal(np.sort(order), np.arange(n))
                        partners = np.array(
                            [rows[int(i)]["partner"] for i in trace[:n, 0]]
                        )
                        bands = np.digitize(trace[:n, 1], [6.0, 15.0])
                        np.testing.assert_array_equal(partners, partners[order])
                        np.testing.assert_array_equal(bands, bands[order])
                        for name, mode, perm in (
                            ("true", arm, np.arange(n)),
                            ("shuffle", arm, order),
                            ("scratch", "scratch", np.arange(n)),
                        ):
                            recipient = Independent(
                                args.source / "results", case["birth_seed"], mode
                            )
                            for entry, reward in zip(trace[:n], trace[perm, 10]):
                                i = int(entry[0])
                                recipient.update(
                                    native[i],
                                    x[i],
                                    rows[i]["partner"],
                                    entry[1],
                                    int(entry[2]),
                                    reward,
                                )
                                replay_steps += 1
                            np.testing.assert_allclose(
                                recipient.weights(),
                                raw[f"{label}-replay-{name}-weights"],
                                atol=2e-6,
                                rtol=0,
                            )
                        if label == "last_preterminal":
                            shuffled.append(
                                {
                                    "life": key,
                                    "count": n,
                                    "moved": int((order != np.arange(n)).sum()),
                                    "feedback_changed": int(
                                        (trace[order, 10] != trace[:n, 10]).sum()
                                    ),
                                }
                            )
                count += len(trace)
                opportunities.append(
                    {
                        "life": key,
                        "arm": arm,
                        "turns": len(trace),
                        "actual_nonterminal_parameter_changes": int(
                            (trace[:-1, 11] > 0).sum()
                        ),
                        "actual_total_parameter_changes": int((trace[:, 11] > 0).sum()),
                        "injured_and_can_choose_again": int(
                            (trace[:-1, 4] < trace[:-1, 1]).sum()
                        ),
                        "pre_health_at_most_6": int((trace[:, 1] <= 6).sum()),
                        "low_health_injury_then_next_choice": int(
                            ((trace[:-1, 1] <= 6) & (trace[:-1, 4] > 0)).sum()
                        ),
                    }
                )
    if maxima["choice_probability"] > 2e-6:
        raise ValueError("Independent probabilities differ")
    result = {
        "passed": True,
        "lives": 128,
        "encounters": count,
        "recipient_updates": replay_steps,
        "max_errors": maxima,
        "opportunities": opportunities,
        "shuffle_effectiveness": shuffled,
        "scope": "CPU raw arrays; independent update/body formulas, shared fixed RUL table and cached features",
    }
    write(args.run / "independent-audit.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("opportunities", "shuffle_effectiveness")
            }
        )
    )


if __name__ == "__main__":
    main()
