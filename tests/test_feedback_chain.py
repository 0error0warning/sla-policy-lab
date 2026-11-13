import unittest
from statistics import NormalDist

from experiments.conditional_penalty.reference_check import LowHealthReference
from experiments.first_fatal.design import FATAL
from experiments.first_fatal.design import settle as trial_settle
from experiments.social_closed_loop.core import settle as paper_settle


class FeedbackChainTests(unittest.TestCase):
    def test_continuous_rul_encloses_analytic_single_survival_tick(self):
        oracle = LowHealthReference()
        # At h=2 the second tick always kills. The first survives iff eta < h/f(h)-1.
        h = 2.0
        threshold = (h / (2 - h / 350) - 1) / 0.1
        normal = NormalDist()
        survive = (normal.cdf(threshold) - normal.cdf(-3)) / (
            normal.cdf(3) - normal.cdf(-3)
        )
        value = (1 + survive) / 4
        lo, hi = oracle.bounds(h)
        self.assertLessEqual(lo, value)
        self.assertGreaterEqual(hi, value)
        self.assertLess(hi - lo, 0.0001)
        for health, expected in ((0, 0), (0.1, 0.25)):
            self.assertEqual(oracle.bounds(health), (expected, expected))

    def test_maintained_answer_separates_wear_from_peak_in_modified_body(self):
        noise = [0.0] * 5
        observed, _, collapsed = trial_settle(5, 2, True, noise)
        paper, _ = paper_settle(5, 2, True, noise)
        self.assertFalse(collapsed)
        self.assertAlmostEqual(5 - paper, 3.475)
        self.assertAlmostEqual(5 - observed, 0.11055824110243062)
        self.assertTrue(trial_settle(FATAL, 2, True, noise)[2])
        # The original reward table returns zero here. The continuous reference
        # still predicts a loss under the fixed original silent continuation.
        lo, hi = LowHealthReference().loss(5, observed)
        self.assertGreater(lo, 0.00039)
        self.assertLess(hi, 0.00041)
