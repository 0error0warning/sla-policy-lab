"""Train a health-conditioned residual in four native Qwen output rows.

Cached prefixes are valid because the trunk is frozen. Only chosen actions and their
realized scalar outcomes enter the optimizer. Arrays use physical action order.
"""

from pathlib import Path

import numpy as np

from experiments.consequence_policy.study import (
    BEHAVIOR_UNIFORM,
    EPOCHS,
    EXPLORATION,
    KL,
    LR,
    NBASIS,
    PCA_RANK,
    TEMP,
    basis,
)


def softmax(z):
    unnormalized = np.exp(z - z.max(axis=-1, keepdims=True))
    return unnormalized / unnormalized.sum(axis=-1, keepdims=True)


def deployed(logits):
    return (1.0 - EXPLORATION) * softmax(logits) + EXPLORATION / 4


def behavior(native):
    return BEHAVIOR_UNIFORM / 4 + (1.0 - BEHAVIOR_UNIFORM) * deployed(native / TEMP)


def fit_projection(hidden, training):
    # No actions, frame labels, rewards, healths or test prompts in this calculation.
    z = hidden[training].astype(np.float64)
    center = z.mean(axis=0)
    z -= center
    eig, u = np.linalg.eigh(z @ z.T / len(z))
    keep = np.argsort(eig)[-PCA_RANK:][::-1]
    variance = eig[keep]
    assert np.min(variance) > 0.0
    matrix = z.T @ u[:, keep] / np.sqrt(len(z) * variance)
    matrix /= np.sqrt(np.maximum(variance, variance[0] * 0.0001))
    return center.astype(np.float32), matrix.astype(np.float32)


def features(z, center, matrix):
    x = (np.asarray(z, dtype=np.float32) - center) @ matrix
    return np.concatenate([x, np.ones((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1)


class Policy:
    def __init__(self, width, blind=False):
        self.weights = np.zeros((2, 1 if blind else NBASIS, width, 4), dtype=np.float32)

    def gates(self, health):
        if self.weights.shape[1] == 1:
            return np.ones((len(health), 1), dtype=np.float32)
        return basis(health).astype(np.float32)

    def grid_logits(self, native, x, partner, health):
        g = self.gates(health)
        logits = np.broadcast_to(native[None] / TEMP, (len(g), len(x), 4)).copy()
        for p in range(2):
            mask = partner == p
            response = np.einsum("nd,bda->bna", x[mask], self.weights[p], optimize=True)
            logits[:, mask] += np.tensordot(g, response, axes=(1, 0))
        return logits

    def residual(self, x, partner, health):
        gate = self.gates([health])[0]
        return np.einsum("b,d,bda->a", gate, x, self.weights[partner], optimize=False)

    def single(self, native, x, partner, health):
        return deployed(native / TEMP + self.residual(x, partner, health))

    def save(self, path):
        np.savez_compressed(path, weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            value = data["weights"]
        obj = cls(value.shape[2], blind=value.shape[1] == 1)
        assert obj.weights.shape == value.shape and np.isfinite(value).all()
        obj.weights[:] = value
        return obj


def feedback_scores(chosen, outcomes, propensity, scale):
    # The leave-one-out baseline is independent of this replica's chosen action.
    repeats = chosen.shape[-1]
    assert repeats > 1
    advantage = (
        (outcomes - outcomes.mean(axis=-1, keepdims=True)) * repeats / (repeats - 1)
    )
    advantage /= scale[:, None, None]
    scores = np.stack(
        [
            np.mean(advantage * (chosen == a), axis=-1) / propensity[:, a]
            for a in range(4)
        ],
        axis=-1,
    )
    return scores.astype(np.float32)


def fit(policy, native, x, partner, health, scores):
    base = deployed(native / TEMP)
    g = policy.gates(health)
    m = np.zeros_like(policy.weights)
    v = np.zeros_like(m)
    history = []
    for step in range(1, EPOCHS + 1):
        logits = policy.grid_logits(native, x, partner, health)
        s = softmax(logits)
        pi = (1.0 - EXPLORATION) * s + EXPLORATION / 4
        ratio = np.log(pi / base[None])
        value_grad = -scores + KL * (1 + ratio)
        grad_logits = (
            (1 - EXPLORATION)
            * s
            * (value_grad - (s * value_grad).sum(axis=-1, keepdims=True))
        )
        param_grad = np.empty_like(m)
        for p in range(2):
            mask = partner == p
            by_gate = np.tensordot(g.T, grad_logits[:, mask], axes=(1, 0))
            param_grad[p] = np.einsum(
                "nd,bna->bda", x[mask], by_gate, optimize=True
            ) / (len(x) * len(g))
        m = 0.9 * m + 0.1 * param_grad
        v = 0.999 * v + 0.001 * param_grad**2
        policy.weights -= (
            LR * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        )
        gain = np.mean(np.sum(pi * scores, axis=-1))
        divergence = np.mean(np.sum(pi * ratio, axis=-1))
        history.append(
            [
                step,
                float(-gain + KL * divergence),
                float(gain),
                float(divergence),
                float(np.linalg.norm(param_grad)),
            ]
        )
    assert np.isfinite(policy.weights).all()
    return history
