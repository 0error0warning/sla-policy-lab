"""Raw request words, partner identity and the same smooth visible body sensor."""

import re
import numpy as np

from experiments.consequence_policy.study import basis
from experiments.individual_life.design import RATE, CAP
from .design import ITEMS, LEVELS, PANEL


def tokens(text):
    return set(re.findall(r"[a-z]+", text.lower()))


vocab = sorted({w for row in ITEMS if row["wording"] == 3 for w in tokens(row["text"])})
lookup = {w: i + 1 for i, w in enumerate(vocab)}
lexical = []
for row in ITEMS:
    words = tokens(row["text"])
    v = np.zeros((2, len(vocab) + 1))
    slot = row["partner"] - 2
    v[slot, 0] = 1.0
    for w in words & lookup.keys():
        v[slot, lookup[w]] = 1.0 / np.sqrt(len(words))
    lexical.append(v.ravel() / np.linalg.norm(v))
lexical = np.asarray(lexical)


def features(i, health):
    z = basis([health])[0]
    return np.outer(lexical[i], z / np.linalg.norm(z)).ravel()


GRID = np.array([[[features(i, h) for i in word] for h in LEVELS] for word in PANEL])


class Simple:
    def __init__(self):
        self.W = np.zeros((4, GRID.shape[-1]))

    def probabilities(self, x):
        scores = x @ self.W.T
        q = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return 0.95 * q / q.sum(axis=-1, keepdims=True) + 0.05 / 4

    def update(self, x, label, signal, behavior):
        p = self.probabilities(x)
        q = (p - 0.05 / 4) / 0.95
        direction = -q
        direction[label] += 1.0
        grad = np.outer(0.95 * q[label] / p[label] * direction, x)
        step = RATE * signal * np.clip(p[label] / behavior, 0.5, 2.0) * grad
        norm = np.linalg.norm(step)
        step *= min(1.0, CAP / max(1e-12, norm))
        self.W += step
        return float(np.linalg.norm(step))


class LinearGroup:
    def __init__(self):
        self.current, self.birth, self.wrong = Simple(), Simple(), Simple()

    def decision(self, i, h):
        x = features(i, h)
        return np.stack(
            [v.probabilities(x) for v in (self.current, self.birth, self.wrong)]
        )

    def learn(self, i, h, a, bad, signal, behavior):
        x = features(i, h)
        return [
            self.current.update(x, a, signal, behavior[a]),
            self.wrong.update(x, bad, signal, behavior[bad]),
        ]

    def probe(self):
        return np.stack(
            [v.probabilities(GRID) for v in (self.current, self.birth, self.wrong)]
        )

    def weights(self):
        return {"current": self.current.W, "wrong": self.wrong.W, "birth": self.birth.W}
