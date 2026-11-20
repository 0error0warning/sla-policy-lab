import unittest

from experiments.damage_penalty.design import RULES
from experiments.health_baseline.design import BASELINES, Past


class DamagePenaltyTests(unittest.TestCase):
    def test_only_current_damage_is_weighted(self):
        history = Past()
        for h, r in ((51, -0.8), (6, -0.13), (5.5, -0.04), (5.2, -0.13)):
            history.observe(h, r)
        g, b, n = history.baselines(5)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(b, -0.10)
        one, two = RULES
        self.assertEqual(one.baseline(g, b), two.baseline(g, b))
        self.assertAlmostEqual(one.signal(-0.08, g, b), 0.02)
        self.assertAlmostEqual(two.signal(-0.08, g, b), -0.06)
        self.assertEqual(one.signal(-0.08, g, b), BASELINES[1].signal(-0.08, g, b))

    def test_sparse_reference_and_no_damage(self):
        g, b, n = Past().baselines(350)
        self.assertEqual((g, b, n), (-1.0, -1.0, 0))
        one, two = RULES
        self.assertEqual(one.signal(0.0, g, b), two.signal(0.0, g, b))
        self.assertEqual(one.signal(-0.25, g, b) - two.signal(-0.25, g, b), 0.25)
