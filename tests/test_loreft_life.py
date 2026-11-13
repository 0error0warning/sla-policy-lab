"""Regression tests for a hidden-state intervention, its controls, and promotion boundaries."""

from importlib.util import find_spec
import unittest
import numpy as np


@unittest.skipUnless(find_spec("scipy"), "study analysis uses SciPy")
class StudyDesign(unittest.TestCase):
    def test_donor_target_cannot_enter_direction(self):
        from experiments.loreft_life.design import PILOT, leave_one_out

        donor = {s: np.array([1.0, s % 2, 0.0]) for s in PILOT}
        target = PILOT[0]
        v, alpha, r = leave_one_out(donor, target)
        donor[target][:] = -1e6
        vv, aa, rr = leave_one_out(donor, target)
        np.testing.assert_array_equal(v, vv)
        np.testing.assert_array_equal(r, rr)
        self.assertEqual(alpha, aa)
        np.testing.assert_allclose(r @ v, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1.0)

    def test_conditional_choice_and_partner_switch(self):
        from experiments.loreft_life.analysis import arm_stats, history_contrast
        from experiments.loreft_life.design import PANEL, ITEMS, orientation, PILOT

        seed = PILOT[0]
        a = np.full((2, 4, 32, 4), 0.25)
        b = a.copy()
        for w, j in enumerate(PANEL):
            for i, idx in enumerate(j):
                row = ITEMS[idx]
                sgn = orientation(seed, row)
                if row["partner"] == 2:
                    a[w, :, i, 2] += 0.06 * sgn
                    a[w, :, i, 3] -= 0.06 * sgn
                    b[w, :, i, 2] -= 0.06 * sgn
                    b[w, :, i, 3] += 0.06 * sgn
        s, l = history_contrast(a, b, seed)
        self.assertAlmostEqual(s, 12.0)
        self.assertAlmostEqual(l, 0.0)
        x = arm_stats(a, seed, "stable")
        self.assertAlmostEqual(x["held_death"], 96.25)
        self.assertAlmostEqual(x["held_healthy"], 50.0)
        self.assertGreater(x["held_cooperation"], 25.0)


@unittest.skipUnless(
    find_spec("torch") and find_spec("scipy"),
    "Torch and SciPy are required for equation-level checks",
)
class Equations(unittest.TestCase):
    def test_orthonormal_trainable_identity_and_persist(self):
        import torch
        from experiments.loreft_life.adapters import LoReFT, ResidualLoRA, advance

        initial = torch.eye(5)[:2].clone()
        h = torch.tensor([0.2, 0.9, -0.2, 0.7, 0.1])
        m = LoReFT(initial)
        l = ResidualLoRA(initial)
        torch.testing.assert_close(m.delta(h), torch.zeros_like(h), atol=0, rtol=0)
        torch.testing.assert_close(l.delta(h), torch.zeros_like(h), atol=0, rtol=0)
        grads = torch.autograd.grad(
            torch.dot(h + m.delta(h), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), m.live()
        )
        update = advance(m, grads, -0.41, 0.42, 0.37)
        self.assertGreater(update["actual_l2"], 0.0)
        torch.testing.assert_close(m.R @ m.R.T, torch.eye(2), atol=1e-6, rtol=0)
        direct = m.R.T @ (m.W @ h + m.bias - m.R @ h)
        torch.testing.assert_close(m.delta(h), direct, atol=2e-7, rtol=1e-6)
        m2 = LoReFT(initial)
        m2.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(m2.delta(h), m.delta(h)))

    def test_steering_swaps_only_the_target_projection(self):
        import torch
        from experiments.loreft_life.engine import Engine

        class Offset:
            def __init__(self, v):
                self.v = torch.tensor(v, dtype=torch.float32)

            def delta(self, x):
                return self.v

        e = Engine.__new__(Engine)
        base = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        dirs = torch.eye(3)
        v = e._variants(
            base, (Offset([2.0, 4.0, 1.0]), Offset([-2.0, 1.0, 2.0])), 0, dirs, 0.5
        ).float()
        self.assertTrue(torch.equal(v[0], v[5]))
        torch.testing.assert_close(v[1] - v[2], dirs[0])
        torch.testing.assert_close(v[4] - v[0], -4 * dirs[0])
        self.assertAlmostEqual(float((v[3] - v[0]).norm()), float((v[8] - v[0]).norm()))
