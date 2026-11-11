"""Independent small checks of the chosen-action learning and the sensor interface."""

import unittest

import numpy as np

from experiments.consequence_policy.policy import Policy, deployed, feedback_scores, fit
from experiments.consequence_policy.study import TEST_H, TRAIN_H, basis, contexts


class ConsequencePolicyTests(unittest.TestCase):
    def test_leave_one_replica_out_with_unequal_propensities(self):
        chosen = np.array([[[0, 1, 1, 3]]])
        r = np.array([[[-1., .5, -.5, 0.]]])
        mu = np.array([[.2, .4, .3, .1]])
        result = feedback_scores(chosen, r, mu, np.array([.5]))[0, 0]
        expected = np.zeros(4)
        for j, action in enumerate(chosen.ravel()):
            baseline = np.delete(r.ravel(), j).mean()
            expected[action] += (r.ravel()[j] - baseline) / (.5 * mu[0, action] * 4)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        shifted = feedback_scores(chosen, r + 25., mu, np.array([.5]))
        np.testing.assert_allclose(shifted[0, 0], result, atol=1e-6)

    def test_zero_consequences_cannot_train_a_policy(self):
        p = Policy(3)
        x = np.array([[.4, .1, 1.], [.3, .7, 1.]], np.float32)
        # Symmetric original logits also have exactly zero regularizer gradient.
        native = np.zeros((2, 4), np.float32)
        fit(p, native, x, np.array([0, 1]), TRAIN_H, np.zeros((8, 2, 4), np.float32))
        np.testing.assert_array_equal(p.weights, 0.)

    def test_one_actor_equals_grid_and_blinding_is_health_invariant(self):
        gen = np.random.default_rng(412)
        x = gen.normal(size=(4, 3)).astype(np.float32)
        native = gen.normal(size=(4, 4)).astype(np.float32)
        partner = np.array([0, 0, 1, 1])
        for blind in (True, False):
            actor = Policy(3, blind=blind)
            actor.weights[:] = gen.normal(size=actor.weights.shape)
            pi = deployed(actor.grid_logits(native, x, partner, TEST_H))
            for j, h in enumerate(TEST_H):
                for i in range(4):
                    np.testing.assert_allclose(actor.single(native[i], x[i], partner[i], h), pi[j, i], atol=2e-7)
            if blind:
                np.testing.assert_array_equal(pi[0], pi[-1])
        np.testing.assert_allclose(basis(TEST_H).sum(axis=1), 1.)

    def test_test_tasks_and_wordings_disjoint_and_orders_balanced(self):
        rows = contexts()
        train, test = [[r for r in rows if r["split"] == k] for k in ("train", "test")]
        for field in ("task", "wording", "text"):
            self.assertFalse({r[field] for r in train} & {r[field] for r in test})
        for group in (train, test):
            for p in (0, 1):
                for frame in (0, 1):
                    counts = np.bincount([r["layout"] for r in group if r["partner"] == p and r["frame"] == frame])
                    self.assertEqual(len(counts), 24)
                    self.assertEqual(counts.min(), counts.max())


if __name__ == "__main__":
    unittest.main()
