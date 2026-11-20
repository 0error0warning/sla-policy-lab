"""Guard the score-gap sign, native categories, four-way sampler and counterfactual decomposition."""

import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("scipy"), "study uses SciPy")
class ExpressionLogits(unittest.TestCase):
    def test_equal_translation_does_not_create_expression_odds(self):
        from experiments.expression_logits.analysis import (
            boundary,
            contribution,
            score_terms,
        )

        base = np.array([-2.0, 1.0, 3.0, 3.0])
        shifted = base + 0.25
        self.assertEqual(score_terms(shifted - base), (0.25, 0.0, "equal_shift"))
        d = contribution(base, shifted)
        self.assertAlmostEqual(boundary(base), boundary(shifted))
        self.assertAlmostEqual(d["boundary_from_differential"], 0)
        self.assertAlmostEqual(
            d["boundary_from_common"] + d["boundary_from_two_refusals"], 0
        )

    def test_only_one_word_and_only_refusals(self):
        from experiments.expression_logits.analysis import contribution, score_terms

        base = np.array([1.0, 2.0, 3.0, 4.0])
        new = base + [0, 0, 0, 0.125]
        self.assertEqual(score_terms(new - base), (0.0625, 0.0625, "one_only"))
        d = contribution(base, new)
        self.assertLess(d["boundary_from_answers"], 0)
        self.assertAlmostEqual(d["boundary_from_refusals"], 0)
        ref = base + np.array([1.0, 0, 0, 0])
        d = contribution(base, ref)
        self.assertGreater(d["boundary_from_refusals"], 0)
        self.assertAlmostEqual(d["boundary_from_common"], 0)
        self.assertAlmostEqual(d["boundary_from_differential"], 0)

    def test_categories_and_common_random_draw(self):
        from experiments.expression_logits.analysis import boundary, policy, score_terms
        from experiments.individual_life.design import action_at

        for a, b, c in (
            (0, 0, "neither"),
            (0.1, 0.1, "equal_shift"),
            (-0.1, -0.2, "same_sign_unequal"),
            (-0.1, 0, "one_only"),
            (0.1, -0.2, "opposite"),
        ):
            x = np.array([0.0, 0.0, a, b])
            common, diff, cat = score_terms(x)
            self.assertEqual(cat, c)
            self.assertAlmostEqual(common**2 + diff**2, (a * a + b * b) / 2)
        z = np.array([-5.0, -4.0, 0.5, 3.0])
        new = z + [0, 0, 0, 0.25]
        u = (boundary(z) + boundary(new)) / 2
        self.assertEqual((z.argmax(), new.argmax()), (3, 3))
        self.assertEqual((action_at(policy(z), u), action_at(policy(new), u)), (2, 3))

    def test_case_and_identical_draw_for_twin(self):
        from experiments.expression_logits.design import (
            CASE_COL,
            CASE_ID,
            TWIN_ID,
            check_case,
        )
        from experiments.lifetime_cooperation.design import PANEL

        check_case()
        self.assertEqual(tuple(PANEL[:, CASE_COL]), (TWIN_ID, CASE_ID))
