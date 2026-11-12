import unittest

import numpy as np

from experiments.health_baseline.analysis import differences
from experiments.health_baseline.design import Past


class PersonalBaselineTests(unittest.TestCase):
    def test_past_only_inclusive_window_and_sparse_fallback(self):
        past = Past()
        self.assertEqual(past.baselines(5), (-1, -1, 0))
        for h, r in ((100, -20), (6.251, -30), (6.25, -2), (5, -1)):
            past.observe(h, r)
        g, l, n = past.baselines(5)
        self.assertEqual((g, l, n), (-61 / 12, -61 / 12, 2))
        past.observe(4, 0)
        g, l, n = past.baselines(5)
        self.assertEqual((g, l, n), (-61 / 13, -1, 3))
        self.assertEqual(past.baselines(100), (g, g, 1))
        self.assertEqual(Past().observations, [])

    def test_shared_draw_disagreement_includes_boundary_crossings_within_class(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.125, 0.375, 0.375, 0.125])
        x = differences(p, q, 0.2)
        self.assertEqual(x, (1, 0, 0.25, 0.25, 0.0, 0))
        self.assertEqual(differences(p, p, 0.125), (0, 0, 0.0, 0.0, 0.0, 1))

    def test_ordered_cdf_disagreement_can_exceed_total_variation(self):
        p = np.array([0.375, 0.25, 0.25, 0.125])
        q = np.array([0.125, 0.25, 0.25, 0.375])
        flip, changed_class, tv, shared, _, _ = differences(p, q, 0.5)
        self.assertEqual((flip, changed_class, tv, shared), (1, 1, 0.25, 0.75))
