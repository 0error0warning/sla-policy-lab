import json
import unittest
from itertools import pairwise

from experiments.body_prediction.design import build
from experiments.self_future.design import fixtures


class BodyPrediction(unittest.TestCase):
    def test_old_plan_messages_unchanged_and_answers_separate(self):
        rows, keys = build()
        old, old_keys = fixtures()
        self.assertEqual(
            [r["messages"] for r in rows if r["task"] == "plan"],
            [r["messages"] for r in old],
        )
        self.assertEqual(keys["plans"], old_keys)
        self.assertNotIn('"answer":', json.dumps(rows))

    def test_nonterminal_history_identifies_costs_and_zero_is_absorbing(self):
        rows, keys = build()
        step = {r["id"]: r for r in rows if r["task"] == "step"}
        for cid, k in keys["steps"].items():
            data = json.loads(step[cid]["messages"][1]["content"])
            obs = data["observations"]
            self.assertEqual(
                {r["action"]: r["after"] - r["before"] for r in obs}, k["effects"]
            )
            self.assertTrue(all(r["before"] > 2 and r["after"] > 2 for r in obs))
            self.assertTrue(
                all(a["after"] == b["before"] for a, b in pairwise(obs))
            )
            if k["health"] == 0:
                self.assertEqual(k["answer"], {"next_health": 0, "can_act_next": False})
        self.assertEqual(len(step), 72)
        self.assertEqual(keys["steps"]["w2-h0-R-history"]["answer"]["next_health"], 0)
        self.assertEqual(keys["steps"]["w2-h1-R-history"]["answer"]["next_health"], 3)
