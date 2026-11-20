"""Decision gates run on independent lives and preserve task identity across wording."""

import unittest

try:
    import scipy
except ImportError:
    scipy = None


@unittest.skipUnless(scipy, "study analysis requires SciPy")
class LifetimeCooperationTests(unittest.TestCase):
    def test_maintenance_and_fatal_boundaries(self):
        import numpy as np
        from experiments.lifetime_cooperation.analysis import scores
        from experiments.lifetime_cooperation.design import PANEL
        from experiments.lifetime_cooperation.audit import check_word_pairs

        check_word_pairs()
        uniform = np.full((3, 2, 4, PANEL.shape[1], 4), 0.25)
        for row in scores(uniform, (0, 1)):
            self.assertAlmostEqual(row["cooperation"], 0.25)
            self.assertAlmostEqual(row["death"], 0.9625)
            self.assertAlmostEqual(row["healthy"], 0.5)
        uniform[:, :, 0, :, :] = [0.0, 1.0, 0.0, 0.0]
        for row in scores(uniform, (1, 0)):
            self.assertAlmostEqual(row["death"], 0.85)

    def test_require_all_gates_and_equal_weight_for_each_seed(self):
        from experiments.lifetime_cooperation.analysis import gate
        from experiments.lifetime_cooperation.design import PILOT

        lives = [
            dict(
                seed=s,
                kind="qwen",
                world="stable",
                learning=b,
                healthy_tasks=4,
                healthy_rounds=5,
                low_then_acted=0,
            )
            for s in PILOT
            for b in (True, False)
        ]
        panels = []
        for s in PILOT:
            for family in ("qwen", "simple_on_qwen"):
                for role in ("current", "birth", "wrong"):
                    for word in ("familiar", "held"):
                        active = role == "current" and family == "qwen"
                        panels.append(
                            dict(
                                seed=s,
                                checkpoint="last_living",
                                world="stable",
                                family=family,
                                partner="all",
                                role=role,
                                split=word,
                                cooperation=0.3 + 0.06 * active,
                                death=0.96 - 0.03 * active,
                                healthy=0.51,
                            )
                        )
        self.assertTrue(gate(lives, panels, PILOT, "qwen")["passed"])
        for r in panels:
            if (
                r["family"] == "qwen"
                and r["role"] == "current"
                and r["split"] == "held"
            ):
                r["death"] = 0.97
        self.assertFalse(gate(lives, panels, PILOT, "qwen")["passed"])

    def test_exogenous_change_shares_original_stream(self):
        from experiments.lifetime_cooperation.design import (
            PILOT,
            rules,
            schedule,
            ITEMS,
        )

        indices, _, _, _ = schedule(PILOT[0])
        rows = [ITEMS[i] for i in indices[:24]]
        self.assertEqual(len({r["layout"] for r in rows}), 24)
        self.assertTrue(all(r["wording"] == 3 for r in rows))
        for k in range(0, 24, 8):
            self.assertEqual(
                len({(r["partner"], r["frame"], r["task"]) for r in rows[k : k + 8]}), 8
            )
        self.assertEqual(rules(PILOT[0], "stable", 23), rules(PILOT[0], "changed", 23))
        self.assertEqual((1, 0), rules(PILOT[0], "changed", 24))


if __name__ == "__main__":
    unittest.main()
