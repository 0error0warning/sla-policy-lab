"""Regression checks for private births, withholding feedback and living boundaries."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.individual_life.design import (
    action_at,
    offers,
    permutation,
    rows,
    starting,
    world,
)


class PrivateLifeTests(unittest.TestCase):
    def test_births_are_separate_and_new_partner_slots_are_equal(self):
        a = np.arange(2 * 12 * 33 * 4, dtype=np.float32).reshape(2, 12, 33, 4)
        b = np.ones((4, 2560), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            fn = Path(tmp) / "source.npz"
            np.savez(fn, A=a, B=b)
            a1, b1 = starting(fn, "population")
            a2, b2 = starting(fn, "population")
            np.testing.assert_array_equal(a1[0], a.mean(0))
            np.testing.assert_array_equal(a1[0], a1[1])
            a1.fill(0)
            b1.fill(0)
            np.testing.assert_array_equal(a2[1], a.mean(0))
            np.testing.assert_array_equal(b2, b)
            blank_a, blank_b = starting(fn, "blank")
            self.assertTrue(blank_a.any())
            self.assertFalse(blank_b.any())
            with self.assertRaises(ValueError):
                starting(fn, "typo")

    def test_scrambled_credit_preserves_completed_block_exactly(self):
        for n in range(1, 9):
            old = list(range(9, 9 + n))
            mapped = [old[i] for i in permutation(68011, 9, n)]
            self.assertCountEqual(old, mapped)
            if n > 1:
                self.assertTrue(all(a != b for a, b in zip(old, mapped)))
            else:
                self.assertEqual(old, mapped)

    def test_counterfactual_sampling_uses_semantic_probabilities(self):
        self.assertEqual(action_at([0.1, 0.2, 0.4, 0.3], 0.45), 2)
        self.assertEqual(action_at([0.5, 0.2, 0.2, 0.1], 0.45), 0)
        self.assertEqual(action_at([0.25] * 4, 0.25), 0)
        self.assertEqual(action_at([0.25] * 4, 0.9999), 3)

    @unittest.skipUnless(
        importlib.util.find_spec("scipy"), "optional scientific integration"
    )
    def test_common_tape_and_independent_body_transition(self):
        from experiments.direct_policy.protocol import settle
        from experiments.individual_life.audit import body_path
        from experiments.individual_life.design import tape

        items = rows()
        indices, _u, noise, offset = tape(68011, items)
        self.assertEqual(len(indices), 2008)
        for j in range(0, 2008, 8):
            self.assertCountEqual(
                [
                    (items[i]["partner"] - 2, items[i]["frame"])
                    for i in indices[j : j + 8]
                ],
                [(a, b) for a in (0, 1) for b in (0, 1)] * 2,
            )
        for j in range(0, 1992, 24):
            self.assertCountEqual(
                [items[i]["layout"] for i in indices[j : j + 24]], range(24)
            )
        self.assertEqual(set(offset), {1, 2, 3})
        for bits in (world(68011), world(68014)):
            p, fav = offers(items[indices[0]], bits)
            self.assertEqual(p[fav], 0.9)
            self.assertEqual(p[1], 0.15)
        for h in (350.0, 6.0, 2.0, 0.1):
            for action in range(4):
                for outcome in (True, False) if action else (None,):
                    _, path = settle(h, action, outcome, noise[0])
                    np.testing.assert_allclose(
                        body_path(h, action, outcome, noise[0]),
                        path,
                        rtol=0,
                        atol=1e-10,
                    )
                    self.assertTrue(path[-1] == 0 or h - path[-1] >= 0.175 - 1e-10)


if __name__ == "__main__":
    unittest.main()
