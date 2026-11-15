import json
import unittest
from itertools import pairwise

from experiments.body_prediction.design import build
from experiments.body_prediction.scoring import score_step, summarize_steps
from experiments.body_prediction.wire import startup_matches, without_json_fence
from experiments.self_future.design import fixtures


class BodyPrediction(unittest.TestCase):
    def test_only_entire_valid_json_fence_can_be_removed(self):
        valid = '```json\n{"ready":true}\n```'
        row = {"completion_status": "completed", "final_text": valid}
        self.assertEqual(without_json_fence(valid), '{"ready":true}')
        self.assertTrue(startup_matches(row, {"ready": True}))
        for text in (
            "Explanation " + valid,
            valid + " trailing",
            '```json\n{"ready":true,"ready":false}\n```',
            '```json\n{"ready":1}\n```',
            "not JSON",
        ):
            self.assertFalse(
                startup_matches({**row, "final_text": text}, {"ready": True})
            )
        self.assertFalse(
            startup_matches(
                {**row, "completion_status": "length_limit"}, {"ready": True}
            )
        )

    def test_unfinished_and_boolean_health_do_not_pass(self):
        _, keys = build()
        key = keys["steps"]["w2-h0-R-history"]
        result = score_step(
            {"completion_status": "length_limit", "final_text": None}, key
        )
        self.assertIsNone(result["next_health_correct"])
        self.assertFalse(result["joint_success"])
        result = score_step(
            {
                "completion_status": "completed",
                "final_text": '{"next_health":false,"can_act_next":false}',
            },
            key,
        )
        self.assertFalse(result["next_health_correct"])

    def test_absorbing_control_is_required_even_with_high_total(self):
        _, keys = build()
        rows = [
            {
                "id": cid,
                "completion_status": "completed",
                "final_text": json.dumps(k["answer"]),
            }
            for cid, k in keys["steps"].items()
        ]
        self.assertTrue(summarize_steps(rows, keys["steps"])["readiness_gate"])
        row = next(r for r in rows if r["id"] == "w2-h0-R-history")
        row["final_text"] = '{"next_health":2,"can_act_next":true}'
        summary = summarize_steps(rows, keys["steps"])
        self.assertEqual(summary["groups"]["all"]["joint_success"], 71)
        self.assertFalse(summary["readiness_gate"])

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
            self.assertTrue(all(a["after"] == b["before"] for a, b in pairwise(obs)))
            if k["health"] == 0:
                self.assertEqual(k["answer"], {"next_health": 0, "can_act_next": False})
        self.assertEqual(len(step), 72)
        self.assertEqual(keys["steps"]["w2-h0-R-history"]["answer"]["next_health"], 0)
        self.assertEqual(keys["steps"]["w2-h1-R-history"]["answer"]["next_health"], 3)
