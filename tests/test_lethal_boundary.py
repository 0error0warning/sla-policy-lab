"""Mechanical boundaries and observable profile design can be tested without Qwen."""

import importlib.util
import unittest

import numpy as np

from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import settle
from experiments.lethal_boundary.design import (
    ARMS,
    HEALTH,
    levels,
    prehistory,
    profiles,
    prompt,
)


class BoundaryTests(unittest.TestCase):
    def test_two_experience_classes_do_not_overlap_the_test(self):
        for seed in (67001, 67002):
            for arm in ARMS:
                hs = levels(arm, seed)
                self.assertFalse(np.intersect1d(HEALTH, hs).size)
                for h in hs[:4]:
                    for action in (2, 3):
                        self.assertEqual(settle(h, action, False, [-0.3] * 5)[0], 0)
                        if h < 2.44:
                            self.assertEqual(settle(h, action, True, [-0.3] * 5)[0], 0)
                        else:
                            self.assertGreater(settle(h, action, True, [0.3] * 5)[0], 0)

    def test_record_information_and_within_layout_balance(self):
        rows = contexts()[:768]
        slots = profiles(67001, rows)
        np.testing.assert_array_equal(np.bincount(slots), [384, 384])
        _, history = prehistory(67001)
        np.testing.assert_array_equal(history.sum(1), [5, 17])
        k = prompt(rows[0], 7)
        plain = prompt(rows[0], None)
        self.assertEqual(k[0], plain[0])
        self.assertIn("07 of 20", k[1]["content"])
        self.assertNotIn("check-ins", plain[1]["content"])
        self.assertTrue(k[1]["content"].endswith(plain[1]["content"]))
        for m in (k, plain):
            self.assertNotIn("Current body health", m[1]["content"])

    @unittest.skipUnless(
        importlib.util.find_spec("scipy"), "optional scientific integration"
    )
    def test_risk_distinguishes_intrinsic_and_realized_deaths(self):
        from experiments.lethal_boundary.evaluate import scores
        from experiments.lethal_boundary.physics import cutoff, protected_death

        self.assertAlmostEqual(float(protected_death(cutoff(1.75, 0), 1.75)), 0.5)
        # Between the two deterministic regions a maintained answer can survive,
        # so answering-and-dying can exceed its intrinsic mortality contribution.
        prob = np.array([[[[0.1, 0.2, 0.3, 0.4]]]])
        death = np.array([[[[1.0, 0.5, 0.5, 0.25]]]])
        value = np.zeros_like(death)
        v = scores(prob, (death, value, np.array([[3]])), np.array([0.2]))[0, 0]
        self.assertAlmostEqual(v[0], 0.7)
        self.assertAlmostEqual(v[1], 0.14)
        self.assertAlmostEqual(v[2], 0.25)
        self.assertAlmostEqual(v[3], 0.45)
        self.assertAlmostEqual(v[4], 0.45)


if __name__ == "__main__":
    unittest.main()
