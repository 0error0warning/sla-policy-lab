import unittest

import numpy as np

from experiments.individual_life.design import rows
from experiments.snapshot_transfer.analysis import effects
from experiments.snapshot_transfer.inputs import checks, row_key


class SnapshotTransferTests(unittest.TestCase):
    def test_contexts_change_exactly_the_requested_fields(self):
        items = rows()
        lookup = {row_key(r): i for i, r in enumerate(items)}
        i, j = lookup[(2, 0, 4, 3, 5)], lookup[(3, 1, 4, 3, 13)]
        stages = checks(i, j, 5.4, 5.21, items, lookup)
        self.assertEqual(stages[:3], [[i, 5.4], [i, 5.21], [j, 5.21]])
        expected = [(3, 0, 4, 3, 5), (2, 1, 4, 3, 5), (2, 0, 4, 3, 13)]
        for (k, h), key in zip(stages[3:], expected, strict=True):
            self.assertEqual(h, 5.21)
            self.assertEqual(row_key(items[k]), key)

    def test_penalizing_one_answer_can_leave_total_answer_mass_constant(self):
        # The original choice was trade, regardless of its display position at the next request.
        pair = np.array([[0.125, 0.125, 0.25, 0.5], [0.125, 0.125, 0.375, 0.375]])
        data = effects(pair, 3, -0.02)
        self.assertEqual(data["chosen_direction"], "correct")
        self.assertEqual(data["chosen_pp"], -12.5)
        self.assertEqual(data["sibling_pp"], 12.5)
        self.assertEqual(data["answer_direction"], "tied")
        self.assertEqual(effects(pair, 3, +0.02)["chosen_direction"], "opposite")
