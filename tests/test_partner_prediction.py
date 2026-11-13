"""Checks for partner credit, chronological supervision and action-specific held wording scores."""

import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("scipy"), "scientific study uses SciPy")
class PartnerPredictionTests(unittest.TestCase):
    def test_wrong_credit_and_dropped_source(self):
        from experiments.lifetime_cooperation.design import ITEMS
        from experiments.partner_damage.design import swapped
        from experiments.partner_prediction.design import LANDMARKS, Forecast

        i = next(i for i in LANDMARKS if ITEMS[i]["partner"] == 3)
        j = swapped(i)
        x = np.zeros((len(ITEMS), 9))
        x[i, 0] = 1
        x[j, 1] = 1
        for arm, expected in (
            ("true", (True, False)),
            ("wrong", (False, True)),
            ("drop93", (False, False)),
            ("drop86", (True, False)),
            ("body", (False, False)),
        ):
            learner = Forecast(arm)
            learner.observe(x, i, 0, None)
            self.assertEqual(learner.updates, 0)
            learner.observe(x, i, 3, True)
            self.assertEqual(
                tuple(learner.read(x, id_)[2] > 0.5 for id_ in (i, j)), expected
            )
            self.assertTrue(np.array_equal(learner.w[:2], np.zeros((2, 9))))
            self.assertLessEqual(np.linalg.norm(learner.w), 0.4)

    def test_held_information_cannot_fit_feature_scales(self):
        from experiments.lifetime_cooperation.design import ITEMS
        from experiments.partner_prediction.design import FEATURE_IDS, features

        gen = np.random.default_rng(11)
        n = len(FEATURE_IDS)
        name, text = (gen.normal(size=(n, 9)) for _ in range(2))
        original, scale = features(name, text, FEATURE_IDS)
        held = np.array([ITEMS[i]["wording"] == 4 for i in FEATURE_IDS])
        text[held] *= -1
        changed, next_scale = features(name, text, FEATURE_IDS)
        self.assertEqual(scale, next_scale)
        src = np.asarray(FEATURE_IDS)[~held]
        self.assertTrue(np.array_equal(original[src], changed[src]))
        self.assertFalse(np.array_equal(original[FEATURE_IDS], changed[FEATURE_IDS]))

    def test_favorable_expression_is_partner_and_frame_specific(self):
        from experiments.individual_life.design import offers
        from experiments.lifetime_cooperation.design import ITEMS, PANEL, rules
        from experiments.partner_prediction.analysis import evaluate_panel
        from experiments.partner_prediction.design import ARMS, MODES, SHADOWS

        seed = 77101
        probs = np.zeros((len(MODES), 2, 4, 32, 4))
        true = np.empty((2, 32, 3))
        for w, ids in enumerate(PANEL):
            for j, i in enumerate(ids):
                rate, favor = offers(ITEMS[i], rules(seed, "stable", 0))
                true[w, j] = rate[1:]
                probs[:, w, :, j, favor] = 1
        archive = {arm: true for arm in (*ARMS, *SHADOWS)}
        policy, pred = evaluate_panel(seed, {**archive, "policy": probs}, "replay")
        self.assertTrue(
            all(
                abs(r["cooperation"] - 90) < 1e-10
                and r["advantage"] == 100
                and r["healthy"] == 100
                for r in policy
            )
        )
        self.assertTrue(all(r["error"] == 0 and r["rank"] == 100 for r in pred))
