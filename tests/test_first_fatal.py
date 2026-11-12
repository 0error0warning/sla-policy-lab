import unittest

import numpy as np

from experiments.first_fatal.analysis import discord, interval
from experiments.first_fatal.audit import expected_check, path
from experiments.first_fatal.design import FATAL, HEALTHY, first_fatal, practice, settle


class FirstFatalTests(unittest.TestCase):
    def test_capacity_boundary_includes_equality_and_has_no_rescue_for_answers(self):
        for h in [2.9, FATAL, np.nextafter(FATAL, np.inf)]:
            for a in (2, 3):
                for response in (True, False):
                    value, _, collapsed = settle(h, a, response, [-0.3] * 5)
                    self.assertEqual(collapsed, h <= FATAL)
                    self.assertEqual(value == 0, h <= FATAL)
        self.assertFalse(first_fatal(0))
        self.assertFalse(practice(FATAL))
        self.assertTrue(practice(np.nextafter(FATAL, np.inf)))
        for a in (0, 1):
            self.assertGreater(
                settle(FATAL, a, None if not a else False, [0.3] * 5)[0], 0
            )

    def test_maintenance_is_same_turn_and_all_wear_is_persistent(self):
        for h in (4.8, 6.0, *HEALTHY):
            for a in (1, 2, 3):
                protected, p, _ = settle(h, a, True, [0.15] * 5)
                unprotected, q, _ = settle(h, a, False, [0.15] * 5)
                self.assertEqual(p[1], q[1])
                self.assertLess(protected, h)
                self.assertGreater(protected, unprotected)
                self.assertEqual(len(p), 6)
                np.testing.assert_allclose(
                    path(h, a, False, [0.15] * 5)[0], q, atol=1e-12, rtol=0
                )
                self.assertLess(settle(protected, 2, True, [0.15] * 5)[0], protected)

    def test_every_always_high_wear_path_gets_repeated_living_practice(self):
        h = 350.0
        hs = []
        while not first_fatal(h):
            hs.append(h)
            h = settle(h, 2, False, [0.3] * 5)[0]
        self.assertEqual(sum(practice(x) for x in hs), 10)
        proof = expected_check()
        self.assertGreaterEqual(proof["guaranteed_practice_turns"], 7)
        self.assertGreater(proof["practice_min_best_refusal_advantage_on_grid"], 0)

    def test_whole_life_contrasts_and_paired_independent_draws(self):
        x = interval([-0.02, -0.02], [-0.02, -0.02])
        self.assertAlmostEqual(x["adjusted_upper"], -0.02)
        self.assertEqual(x["n"], 4)
        d = discord([(1, 0), (1, 1), (0, 1), (1, 0)])
        self.assertEqual(d["true_avoids_other_answers"], 2)
        self.assertEqual(d["true_answers_other_avoids"], 1)
        self.assertEqual(d["same"], 1)
        self.assertEqual(d["paired_exact_two_sided_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
