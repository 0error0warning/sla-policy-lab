"""Counterfactual alignment and paired-partner control definitions."""

from importlib.util import find_spec
import unittest
import numpy as np


@unittest.skipUnless(find_spec("scipy"), "study requires SciPy")
class OmissionDesignTests(unittest.TestCase):
    def test_dates_require_same_action_and_allow_rewrite(self):
        from experiments.loreft_omission.design import transition_events

        trace = [-1, 1, 2, 1, 1, 1, -1, 1, 1, 1, 3, 3, 3]
        found = transition_events(trace, "actual")
        self.assertEqual(
            [(e["turn"], e["kind"], e["action"]) for e in found],
            [(4, "onset", 1), (11, "rewrite", 3)],
        )

    def test_healthy_gate_requires_both_partners_and_each_health(self):
        from experiments.loreft_omission.design import (
            PARTNER,
            panel_labels,
            majority,
            final_run,
        )

        x = np.full((3, 4, 32, 4), 0.25)
        x[:, 2:] = [0.01, 0.97, 0.01, 0.01]
        x[0, 3, PARTNER == 3] = [0.01, 0.01, 0.97, 0.01]
        self.assertEqual(panel_labels(x).tolist(), [-1, 1, 1])
        self.assertEqual(final_run(majority(x)), (2, 1))

    def test_omit_window_ends_before_predecision_and_controls_are_disjoint(self):
        from experiments.loreft_omission.design import event_plan, branches

        n = 36
        birth = np.full((n, 4), 0.25)
        decision = birth.copy()
        decision[16:19] = [0.01, 0.01, 0.97, 0.01]
        grid = np.full((n, 4, 32, 4), 0.25)
        for j in range(32):
            grid[:, :, j, j % 4] += 0.03
            grid[:, :, j, (j + 1) % 4] -= 0.03
        hist = [{"health": 350.0 - i} for i in range(n)]
        event = event_plan(hist, decision, birth, grid, 0.5)[0]
        self.assertEqual(event["turn"], 17)
        self.assertGreaterEqual(abs(event["control"] - event["turn"]), 6)
        forks = branches([event])
        self.assertIn((14, 15, 16), forks)
        self.assertIn((16,), forks)
        self.assertTrue(all(17 not in x for x in forks))
        ref = event_plan(hist, birth, birth, grid, 0.5)
        self.assertEqual(len(ref), 1)
        self.assertEqual(ref[0]["family"], "no_onset")
        self.assertIsNone(ref[0]["control"])

    def test_metrics_use_paired_partners_and_all_action_death(self):
        from experiments.loreft_omission.analysis import measures
        from experiments.loreft_omission.design import PARTNER

        x = np.full((4, 32, 4), 0.25)
        x[2:, PARTNER == 2] = [0.0, 1.0, 0.0, 0.0]
        m = measures(x, 77101, 1)
        self.assertAlmostEqual(m["p86"], 100.0)
        self.assertAlmostEqual(m["p93"], 25.0)
        self.assertAlmostEqual(m["total_death"], 96.25)
