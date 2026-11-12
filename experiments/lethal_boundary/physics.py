"""Independent conditional-risk integration and observed single-action body tapes."""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from experiments.direct_policy.protocol import COSTS
from experiments.social_closed_loop.core import body_event


def cutoff(cost, eta):
    d = np.asarray(cost) * (1 + np.asarray(eta))
    return 700 * d / (350 + d)


def protected_death(h, cost):
    h = np.asarray(h, dtype=float)
    if cost == 0:
        return np.zeros_like(h)
    t = (h / (cost * (2 - h / 350)) - 1) / 0.1
    return np.clip((ndtr(3.0) - ndtr(t)) / (ndtr(3.0) - ndtr(-3.0)), 0.0, 1.0)


def integration(h, body, seed=67090, power=17):
    h = np.asarray(h, dtype=float)
    u = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(power)
    noise = 0.1 * ndtri(ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * u)
    prob = np.zeros((len(h), 4, 2))
    value = np.zeros_like(prob)
    for a, c in enumerate(COSTS):
        thresholds = 700 * (
            1
            - 1
            / (
                (1 + c * (1 + noise[:, 0]) / 350)
                * np.prod(1 + (1 + noise[:, 1:]) / 350, axis=1)
            )
        )
        # A draw with a threshold above or equal to the starting health dies.
        prob[:, a, 0] = 1 - np.searchsorted(np.sort(thresholds), h, side="left") / len(
            thresholds
        )
        prob[:, a, 1] = protected_death(h, c)
        value[:, a, 0] = body.interpolate(h, body.skip[a]) - body.interpolate(
            h, body.reference
        )
        value[:, a, 1] = body.interpolate(h, body.maintain[a]) - body.interpolate(
            h, body.reference
        )
    return prob, value


def maintenance_table(rows, mapping, q):
    from experiments.lethal_boundary.design import favorite

    values = np.broadcast_to([0.0, 0.15, 0.1, 0.1], (len(rows), 4)).copy()
    values[np.arange(len(rows)), favorite(rows, mapping)] = q
    return values


def action_responses(rows, mapping, q, action, uniform):
    p = maintenance_table(rows, mapping, q)[np.arange(len(rows))[None, :, None], action]
    return np.where(action == 0, -1, uniform < p).astype(np.int8)


def experience(h, chosen, response, noise, body):
    h = np.asarray(h)
    before = h[:, None, None] if h.ndim == 1 else h[:, :, None]
    after = body_event(before, np.asarray(COSTS)[chosen], noise[..., 0])
    for j in range(1, 5):
        after = body_event(after, (response != 1).astype(float), noise[..., j])
    reward = body.interpolate(after, body.reference) - body.interpolate(
        before, body.reference
    )
    return after, reward


def focal_audit(h, chosen, response, after, reward, body):
    out = []
    for j, level in enumerate(h):
        scale = min(1.0, body.rul(level))
        adv = (
            (reward[j] - reward[j].mean(-1, keepdims=True))
            * chosen.shape[-1]
            / (chosen.shape[-1] - 1)
            / scale
        )
        for a in range(4):
            select = chosen[j] == a
            out.append(
                {
                    "slot": j,
                    "health": level,
                    "action": a,
                    "count": int(select.sum()),
                    "maintenance": int((response[j][select] == 1).sum()),
                    "skips": int((response[j][select] == 0).sum()),
                    "deaths": int((after[j][select] == 0).sum()),
                    "maintained_deaths": int(
                        ((response[j] == 1) & (after[j] == 0) & select).sum()
                    ),
                    "mean_reward": float(reward[j][select].mean()),
                    "mean_advantage": float(adv[select].mean()),
                }
            )
    return out
