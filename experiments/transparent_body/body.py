"""Declared body calculator. No partner oracle, random tape, or action selector.

The seven-node predecessor integrated a truncated normal using equal-probability
midpoints. This calculator exposes the node count and conditional distributions;
it does not replace the actual stochastic environment settlement.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from statistics import NormalDist

import numpy as np

WORDS = ("wait", "ask", "help", "trade")
COSTS = (0.0, 0.25, 1.75, 1.75)


@dataclass(frozen=True)
class PersonalForecast:
    life: str
    observed_through: int
    decision_turn: int
    parameter_sha256: str
    probabilities: tuple[float, float, float]
    condition: str = "personal"

    def __post_init__(self):
        if (
            not self.life
            or type(self.observed_through) is not int
            or type(self.decision_turn) is not int
            or not 0 <= self.observed_through < self.decision_turn
        ):
            raise ValueError("forecast must precede this individual's decision")
        if len(self.parameter_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.parameter_sha256
        ):
            raise ValueError("a parameter fingerprint is required")
        if self.condition not in ("personal", "frozen", "mismatched"):
            raise ValueError("oracle inputs are not a formal personal forecast")
        if len(self.probabilities) != 3 or any(
            isinstance(p, bool) or not isfinite(p) or not 0 <= p <= 1
            for p in self.probabilities
        ):
            raise ValueError("three finite response probabilities required")


def _event(h, c, eta):
    return np.maximum(0.0, h - c * (2.0 - h / 350.0) * (1.0 + eta))


def _extreme(h, cost, maintained, eta):
    h = float(_event(h, cost, eta))
    for _ in range(4):
        h = float(_event(h, 0.0 if maintained else 1.0, eta))
    return h


def _event_death_probability(h, c):
    """Exact tail integral for the final truncated-normal event.

    Integrating an indicator on a small discrete noise grid is inaccurate near
    death. Marginalize the last noise analytically instead of adding more nodes.
    """
    if c == 0:
        return np.asarray(h == 0, dtype=float)
    threshold = np.asarray(h) / (c * (2.0 - np.asarray(h) / 350.0)) - 1.0
    normal = NormalDist()
    normal_cdf = np.vectorize(normal.cdf, otypes=[float])(threshold / 0.1)
    cdf = (normal_cdf - normal.cdf(-3.0)) / (normal.cdf(3.0) - normal.cdf(-3.0))
    return np.where(np.asarray(h) == 0, 1.0, 1.0 - np.clip(cdf, 0.0, 1.0))


@lru_cache(maxsize=512)
def _branches(health, cost, noise, nodes):
    if noise == "zero":
        eta = np.array([0.0])
        bound = 0.0
    elif noise == "truncated_normal":
        normal = NormalDist()
        lo, hi = normal.cdf(-3.0), normal.cdf(3.0)
        eta = np.array(
            [
                0.1 * normal.inv_cdf(lo + (hi - lo) * (i + 0.5) / nodes)
                for i in range(nodes)
            ]
        )
        bound = 0.3
    else:
        raise ValueError("undeclared noise model")
    start = np.array([health]) if cost == 0 else _event(health, cost, eta)
    branches = []
    for maintained in (False, True):
        h = start.copy()
        risk = float(_event_death_probability(np.array([health]), cost).mean())
        if not maintained:
            for _ in range(3):
                h = _event(h[:, None], 1.0, eta[None, :]).ravel()
            risk = float(_event_death_probability(h, 1.0).mean())
            h = _event(h[:, None], 1.0, eta[None, :]).ravel()
        if noise == "zero":
            risk = float((h == 0).mean())
        low = _extreme(health, cost, maintained, bound)
        high = _extreme(health, cost, maintained, -bound)
        # Bounds also certify risks far from the discontinuous death boundary.
        risk = 0.0 if low > 0 else 1.0 if high == 0 else risk
        branches.append((float(h.mean()), risk, low, high, len(h)))
    return tuple(branches)


class BodyCalculator:
    def __init__(self, noise="truncated_normal", nodes=11):
        if (
            noise not in ("zero", "truncated_normal")
            or type(nodes) is not int
            or nodes < 3
        ):
            raise ValueError("invalid declared quadrature")
        self.noise, self.nodes = noise, nodes

    def conditional(self, health):
        if isinstance(health, bool) or not isfinite(health) or not 0 < health <= 350:
            raise ValueError("a live body in (0,350] is required")
        result = {}
        for word, cost in zip(WORDS, COSTS, strict=True):
            rows = _branches(float(health), cost, self.noise, self.nodes)
            result[word] = {}
            for label, row in zip(("no_maintenance", "maintenance"), rows, strict=True):
                result[word][label] = (
                    None
                    if word == "wait" and label == "maintenance"
                    else {
                        "expected_health": row[0],
                        "p_death": row[1],
                        "min_health": row[2],
                        "max_health": row[3],
                        "quadrature_paths": row[4],
                    }
                )
        return result

    def calculate(self, health, forecast):
        if not isinstance(forecast, PersonalForecast):
            raise TypeError("formal calculation requires a past-only personal forecast")
        return self.diagnostic(health, forecast.probabilities)

    def diagnostic(self, health, rates):
        """Also usable by a separate evaluator; never produces an action ranking."""
        if len(rates) != 3 or any(not isfinite(p) or not 0 <= p <= 1 for p in rates):
            raise ValueError("invalid probabilities")
        branches = self.conditional(health)
        actions = []
        for i, word in enumerate(WORDS):
            no = branches[word]["no_maintenance"]
            yes = branches[word]["maintenance"]
            p = 0.0 if i == 0 else float(rates[i - 1])
            risk = (
                no["p_death"]
                if i == 0
                else (1 - p) * no["p_death"] + p * yes["p_death"]
            )
            expected = (
                no["expected_health"]
                if i == 0
                else (1 - p) * no["expected_health"] + p * yes["expected_health"]
            )
            actions.append(
                {
                    "action": word,
                    "p_death": risk,
                    "completes_task": i >= 2,
                    "expected_health": expected,
                    "predicted_maintenance": p,
                    "conditional": branches[word],
                }
            )
        return {
            "health": health,
            "noise": self.noise,
            "quadrature_nodes": self.nodes,
            "objective_scope": "one encounter; not Reference-RUL or lifetime-optimal",
            "actions": actions,
        }
