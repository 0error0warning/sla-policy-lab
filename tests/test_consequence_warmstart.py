"""Initialization isolation, finite-difference update, recipient replay and causal cutoffs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.consequence_policy.lifetime import PersonalActor
from experiments.consequence_policy.policy import Policy
from experiments.consequence_policy.study import contexts
from experiments.consequence_warmstart.design import permutation, selected, tape
from experiments.consequence_warmstart.runner import replay


class WarmstartTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        gen = np.random.default_rng(914)
        for m in (0, 1):
            dest = self.root / f"s1-m{m}"
            dest.mkdir()
            p = Policy(3)
            p.weights[:] = gen.normal(0, 0.1, p.weights.shape)
            p.save(dest / "real.npz")

    def actor(self, arm):
        return PersonalActor(self.root, 1, (0, 1), arm)

    def test_warm_is_prior_without_direction_and_scratch_is_zero(self):
        warm, prior, scratch = [
            self.actor(k) for k in ("warmstart_adapter", "prior", "scratch")
        ]
        np.testing.assert_array_equal(warm.main.weights, prior.main.weights)
        np.testing.assert_array_equal(warm.direction.weights, 0)
        np.testing.assert_array_equal(scratch.main.weights, 0)
        self.assertTrue(np.any(warm.main.weights))

    def test_shared_update_matches_finite_difference_log_probability(self):
        warm, scratch = self.actor("warmstart_adapter"), self.actor("scratch")
        scratch.main.weights[:] = warm.main.weights
        x = np.array([[0.8, -0.2, 1.0]], np.float32)
        native = np.array([[0.1, 0.9, -0.2, 0.8]], np.float32)
        row, h, a, r = {"partner": 0}, 90.0, 2, -0.4
        pi, logits, d = warm.score(0, row, h, native, x)
        before = warm.main.weights.copy()
        eps, position = 0.002, (0, 8, 0, a)
        logps = []
        for shift in (eps, -eps):
            warm.main.weights[:] = before
            warm.main.weights[position] += shift
            logps.append(np.log(warm.score(0, row, h, native, x)[0][a]))
        warm.main.weights[:] = before
        expected = 0.7 * (r + 0.9) * (logps[0] - logps[1]) / (2 * eps)
        for actor in (warm, scratch):
            actor.update(0, a, h, x[0], pi, logits, d, r)
        np.testing.assert_array_equal(warm.main.weights, scratch.main.weights)
        np.testing.assert_array_equal(warm.main.weights[1], before[1])
        self.assertAlmostEqual(
            float(warm.main.weights[position] - before[position]), expected, delta=2e-5
        )

    def test_replay_uses_recipient_scores_not_donor(self):
        rows = [{"partner": 0}]
        x, native = (
            np.array([[0.6, 0.2, 1.0]], np.float32),
            np.zeros((1, 4), np.float32),
        )
        trace = np.zeros((2, 16))
        trace[:, 1] = 90
        trace[:, 2] = 2
        trace[:, 10] = [-0.4, -1.1]
        recipient, manual = self.actor("scratch"), self.actor("scratch")
        replay(recipient, trace, np.array([1, 0]), rows, native, x)
        for reward in (-1.1, -0.4):
            pi, z, d = manual.score(0, rows[0], 90, native, x)
            manual.update(0, 2, 90, x[0], pi, z, d, reward)
        np.testing.assert_array_equal(recipient.main.weights, manual.main.weights)

    def test_prefix_permutation_cannot_read_future_or_change_marginals(self):
        rows = [{"partner": 0}, {"partner": 1}]
        trace = np.zeros((10, 16))
        trace[:, 0] = np.arange(10) % 2
        trace[:, 1] = [100, 100, 100, 100, 5, 5, 5, 5, 2, 2]
        order, _ = permutation(trace, rows, 78201, (0, 0), "wording", 8)
        self.assertEqual(sorted(order.tolist()), list(range(8)))
        np.testing.assert_array_equal(trace[:8, 0], trace[order, 0])
        np.testing.assert_array_equal(
            np.digitize(trace[:8, 1], [6, 15]), np.digitize(trace[order, 1], [6, 15])
        )

    def test_wording_block_preserves_task_partner_frame_and_layout(self):
        rows = contexts()
        ids, _, _ = tape(78201, (0, 0), "wording", rows)
        self.assertEqual({(rows[i]["task"], rows[i]["wording"]) for i in ids}, {(0, 0)})
        held = {
            (rows[i]["partner"], rows[i]["frame"], rows[i]["layout"]): rows[i]
            for i in selected(rows, (0, 1))
        }
        for i in ids:
            r = rows[i]
            self.assertEqual(
                held[r["partner"], r["frame"], r["layout"]]["task"], r["task"]
            )
        self.assertEqual(len(held), 96)


if __name__ == "__main__":
    unittest.main()
