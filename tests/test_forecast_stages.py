"""Reject future-feedback indexing and incorrect coupled expression switches."""

import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("scipy"), "optional scientific dependencies")
class ForecastStages(unittest.TestCase):
    def test_stages_and_shared_draws(self):
        from experiments.forecast_stages.design import DENSE, random_panel, turns

        self.assertEqual(turns(61), (0, 8, 30, 61))
        with self.assertRaises(ValueError):
            turns(16)
        u, _ = random_panel(77101)
        self.assertEqual(u.shape, (32, 2))
        self.assertTrue(np.array_equal(u, random_panel(77101)[0]))
        self.assertFalse(np.array_equal(u, random_panel(77102)[0]))
        self.assertEqual(len(DENSE), 8)

    def test_exact_expression_change_separates_entering_answer(self):
        from experiments.forecast_stages.analysis import contrast
        from experiments.forecast_stages.design import shared_joint

        # As u increases from zero, new help replaces old trade on [.65,.80).
        old = np.array([[0.2, 0.2, 0.25, 0.35]])
        new = np.array([[0.1, 0.1, 0.6, 0.2]])
        j = shared_joint(old, new)
        np.testing.assert_allclose(j.sum(-1), old)
        np.testing.assert_allclose(j.sum(-2), new)
        self.assertAlmostEqual(j[0, 3, 2], 0.15)
        c = contrast(new, old, np.array([0.7]), np.array([2]))
        self.assertEqual((c["sample_exchanged"], c["toward_good"]), (1, 1))
        self.assertAlmostEqual(c["exact_sample_exchanged"], 15)
        c = contrast(new, old, np.array([0.3]), np.array([2]))
        self.assertEqual((c["sample_exchanged"], c["enter_answer"]), (0, 1))
        self.assertEqual(
            contrast(old, old, np.array([0.3]), np.array([2]))["mean_tv"], 0
        )

    def test_oracle_changes_only_the_predicted_response(self):
        from experiments.forecast_stages.design import inputs, oracle

        q = oracle(77101)
        self.assertTrue(np.all(q[..., 0] == 0.15))
        np.testing.assert_array_equal(q[0], q[1])
        self.assertTrue(np.allclose(q[:, :, 1:].sum(-1), 1))
        x = inputs(np.array([0.18, 0.41, 0.24]), q[0, 0])
        np.testing.assert_array_equal(x[1], q[0, 0])
        np.testing.assert_array_equal(x[2], np.full(3, 0.5))
