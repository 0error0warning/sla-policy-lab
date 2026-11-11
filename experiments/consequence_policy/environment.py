"""Observed consequence tapes and complete mortal lives in the inherited body."""

import numpy as np
from scipy.special import ndtr, ndtri

from experiments.consequence_policy.policy import behavior
from experiments.consequence_policy.study import PARTNERS, REPEATS, TRAIN_H
from experiments.direct_policy.protocol import COSTS, probabilities, rng
from experiments.social_closed_loop.core import body_event


def draw_tape(seed, native):
    shape = (len(TRAIN_H), len(native), REPEATS)
    gen = rng(seed, "consequence-checkpoint-tape-v1")
    choice_u = gen.random(shape)
    partner_u = gen.random(shape)
    quantile = ndtr(-3.) + (ndtr(3.) - ndtr(-3.)) * gen.random((*shape, 5))
    noise = .1 * ndtri(quantile)
    mu = behavior(native)
    chosen = (choice_u[..., None] > mu.cumsum(axis=-1)[None, :, None, :3]).sum(-1).astype(np.uint8)
    return chosen, mu, partner_u, noise


def settled_tape(chosen, partner_u, noise, rows, mapping, body, healths=TRAIN_H):
    # Only the environment sees the response rule. Unchosen outcomes are never produced.
    probs = np.array([[0., *probabilities(PARTNERS[r["partner"]], r["frame"], mapping)] for r in rows])
    p = probs[np.arange(len(rows))[None, :, None], chosen]
    protected = (chosen != 0) & (partner_u < p)
    before = np.broadcast_to(np.array(healths)[:, None, None], chosen.shape)
    h = body_event(before, np.array(COSTS)[chosen], noise[..., 0])
    for j in range(1, 5):
        h = body_event(h, (~protected).astype(float), noise[..., j])
    value_before = body.interpolate(before, body.reference)
    reward = body.interpolate(h, body.reference) - value_before
    # -1 denotes no request, 0 an actual refusal, 1 maintenance.
    response = np.where(chosen == 0, -1, protected.astype(np.int8)).astype(np.int8)
    return h, reward, response


def shuffled_feedback(reward, seed):
    gen = rng(seed, "permute-within-eight-v1")
    permutation = np.argsort(gen.random(reward.shape), axis=-1)
    return np.take_along_axis(reward, permutation, axis=-1), permutation.astype(np.uint8)
