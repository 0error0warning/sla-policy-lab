import unittest

import numpy as np

from experiments.consequence_policy.study import ORDERS
from experiments.order_transfer.analysis import probability_row
from experiments.order_transfer.design import rotations


class OrderTransferTests(unittest.TestCase):
    def test_rotations_balance_every_semantic_action_and_leave_twenty_common_orders(
        self,
    ):
        for factual in range(24):
            layouts = rotations(factual)
            self.assertEqual(layouts[0], factual)
            self.assertEqual(len(set(layouts)), 4)
            self.assertEqual(len(set(range(24)) - set(layouts)), 20)
            for a in range(4):
                self.assertEqual(
                    sorted(ORDERS[i].index(a) for i in layouts), list(range(4))
                )

    def test_same_semantic_action_and_answer_mass_are_compared_between_arms(self):
        c = {
            "seed": 71101,
            "birth": "blank",
            "wave": "pilot",
            "chosen": 3,
            "signal": -0.1,
            "next_context": {"layout": 5},
            "train_layouts": [0, 1, 2, 3],
        }
        q = np.array(
            [
                [0.25, 0.125, 0.125, 0.5],
                [0.25, 0.125, 0.25, 0.375],
                [0.375, 0.125, 0.25, 0.25],
            ]
        )
        x = probability_row(c, "held", q)
        self.assertTrue(x["next_order_unseen"])
        self.assertEqual(x["repeat_minus_pre_chosen_pp"], -12.5)
        self.assertEqual(x["repeat_minus_pre_answer_pp"], 0.0)
        self.assertEqual(x["rotate_minus_repeat_chosen_pp"], -12.5)
        self.assertEqual(x["rotate_minus_repeat_answer_pp"], -12.5)
