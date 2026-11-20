"""Direct read-path, crossed identity and nonconfounded site evaluation regression checks."""

from importlib.util import find_spec
import unittest
import numpy as np


@unittest.skipUnless(find_spec("scipy"), "study requires SciPy")
class PartnerDamageDesign(unittest.TestCase):
    def test_maintenance_label_and_damage_can_be_crossed(self):
        from experiments.partner_damage.design import factorial, DOSES, CELLS

        class Body:
            @staticmethod
            def rul(h):
                return h / 5

        e = dict(action=2, health=180.0, noise=[0.0] * 5, baseline=-1.0)
        x = factorial(Body(), e)
        self.assertEqual(list(x), list(CELLS))
        for label, (social, light) in zip(CELLS, DOSES, strict=True):
            self.assertEqual(x[label]["maintenance"], social)
            self.assertEqual(
                x[label]["signal"],
                x["accept_light" if light else "accept_heavy"]["signal"],
            )
        self.assertGreater(x["accept_light"]["signal"], 0.0)
        self.assertLess(x["accept_heavy"]["signal"], 0.0)
        self.assertLess(x["accept_light"]["damage"], x["accept_heavy"]["damage"])
        with self.assertRaises(ValueError):
            factorial(Body(), dict(e, action=0))

    def test_swapping_ids_preserves_semantics(self):
        from experiments.partner_damage.design import GRID, PAIRS, SMALL, swapped
        from experiments.lifetime_cooperation.design import ITEMS

        self.assertTrue(np.array_equal(PAIRS[PAIRS], np.arange(32)))
        for i in GRID:
            j = swapped(i)
            self.assertEqual(i, swapped(j))
            self.assertNotEqual(ITEMS[i]["partner"], ITEMS[j]["partner"])
            self.assertEqual(ITEMS[i]["text"], ITEMS[j]["text"])
        self.assertEqual(len(SMALL), 16)

    def test_spill_uses_the_actual_sources_side(self):
        from experiments.partner_damage.analysis import panel_tv
        from experiments.partner_damage.design import PARTNER

        p = np.full((4, 32, 4), 0.25)
        q = p.copy()
        q[:, PARTNER == 3, 0] += 0.1
        q[:, PARTNER == 3, 1] -= 0.1
        for source in (2, 3):
            d = panel_tv(p, q, source)
            self.assertAlmostEqual(d["source" if source == 3 else "other"], 10.0)
            self.assertAlmostEqual(d["other" if source == 3 else "source"], 0.0)
