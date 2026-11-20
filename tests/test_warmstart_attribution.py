"""Health projection identity and matched-omission control invariants."""

import unittest

import numpy as np

from experiments.consequence_policy.study import basis
from experiments.consequence_warmstart.attribution import band, drop_mask, matched_mask


class AttributionTests(unittest.TestCase):
    def test_logit_projection_factors_health_language_and_action(self):
        g, q = basis([120.0, 6.0])
        x, y = np.array([1.0, 0.5, -0.2]), np.array([-0.1, 1.0, 1.0])
        d = np.array([0.1, -0.3, 0.4, -0.2])
        weight = np.einsum("b,d,a->bda", g, x, d)
        direct = np.einsum("b,d,bda->a", q, y, weight)
        np.testing.assert_allclose(direct, np.dot(q, g) * np.dot(x, y) * d, atol=1e-16)
        self.assertLess(np.dot(q, g) / np.dot(g, g), 1e-8)

    def test_omissions_are_observed_event_types(self):
        trace = np.zeros((5, 16))
        trace[:, 2] = [2, 3, 1, 1, 0]
        trace[:, 3] = [1, 0, 1, 0, -1]
        np.testing.assert_array_equal(
            drop_mask(trace, "answer_maintained"), [True, False, False, False, False]
        )
        np.testing.assert_array_equal(
            drop_mask(trace, "answer_refused"), [False, True, False, False, False]
        )
        np.testing.assert_array_equal(
            drop_mask(trace, "nonanswer"), [False, False, True, True, True]
        )

    def test_count_matched_mask_preserves_partner_and_health_counts(self):
        rows = [{"partner": 0}, {"partner": 1}]
        trace = np.zeros((32, 16))
        trace[:, 0] = np.arange(32) % 2
        trace[:, 1] = np.repeat([3.0, 10.0, 50.0, 200.0], 8)
        mask = np.arange(32) % 3 == 0
        case = {"environment_seed": 78201, "world": [0, 1], "block": "wording"}
        new = matched_mask(mask, trace, rows, case, "answer_maintained", 0)
        for p in (0, 1):
            for b in ("le6", "6_to15", "15_to90", "gt90"):
                idx = (trace[:, 0] == p) & np.array([band(h) == b for h in trace[:, 1]])
                self.assertEqual(int(mask[idx].sum()), int(new[idx].sum()))
        np.testing.assert_array_equal(
            new, matched_mask(mask, trace, rows, case, "answer_maintained", 0)
        )


if __name__ == "__main__":
    unittest.main()
