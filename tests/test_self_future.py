"""Counterexamples to persistence, absorption, information leaks, and scoring."""

import copy
import json
import unittest

from experiments.self_future.design import fixtures, prefix_reference, simulate
from experiments.self_future.score import score, summarize


def synthetic_record(case_id, key):
    return {
        "id": case_id,
        "completion_status": "completed",
        "final_text": json.dumps(
            {"plans": key["plans"], "choice": key["accepted_choices"][0]}
        ),
    }


class SelfFuture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts, cls.keys = fixtures()

    def test_absorption_survives_later_repair_and_tasks_stop(self):
        # An unconstrained prefix becomes positive again, but the body is already gone.
        effects = {"A": -2, "B": -1, "R": 10}
        expected = {
            "health": [2, 0, None, None],
            "terminal_step": 2,
            "tasks": 1,
            "can_act_after_plan": False,
        }
        for oracle in (simulate, prefix_reference):
            self.assertEqual(oracle(4, effects, ["A", "A", "R", "B"]), expected)
        self.assertEqual(
            simulate(4, {"A": -2, "B": -1, "R": 0}, ["A", "R", "A"])["health"],
            [2, 2, 0],
        )
        self.assertEqual(
            simulate(4, {"A": -2, "B": -1, "R": 2}, ["A", "R", "A"])["health"],
            [2, 4, 2],
        )

    def test_history_is_contiguous_identifiable_and_nonlethal(self):
        for row in self.prompts:
            data = json.loads(row["messages"][1]["content"])
            effects = self.keys[row["id"]]["case"]["effects"]
            history = data["observations"]
            self.assertEqual(history[-1]["after"], data["current_durability"])
            seen = {}
            for i, event in enumerate(history):
                if i:
                    self.assertEqual(event["before"], history[i - 1]["after"])
                self.assertGreater(
                    min(event["before"], event["after"]),
                    max(-effects["A"], -effects["B"]),
                )
                seen.setdefault(event["action"], []).append(
                    event["after"] - event["before"]
                )
            self.assertEqual(
                seen, {action: [effect, effect] for action, effect in effects.items()}
            )
            for plan in self.keys[row["id"]]["plans"].values():
                self.assertGreater(plan["health"][0], 0)

    def test_two_information_inputs_differ_only_by_known_effects(self):
        indexed = {row["id"]: row for row in self.prompts}
        for row in self.prompts:
            if row["information"] != "history":
                continue
            other = indexed[row["id"].replace("-history", "-rules")]
            self.assertEqual(row["messages"][0], other["messages"][0])
            explicit = json.loads(other["messages"][1]["content"])
            explicit.pop("known_effects")
            history = json.loads(row["messages"][1]["content"])
            self.assertEqual(explicit, history)
            self.assertEqual(
                set(history), {"current_durability", "observations", "proposed_plans"}
            )

    def test_counterfactual_pairs_really_require_a_change(self):
        self.assertEqual(len(self.keys), 28)
        for split in ("development", "confirmation"):
            for information in ("rules", "history"):
                subset = [
                    key
                    for key in self.keys.values()
                    if key["case"]["split"] == split
                    and key["information"] == information
                ]
                self.assertEqual(len(subset), 7)
                for family in ("health", "repair", "composition"):
                    a, b = [k for k in subset if k["case"]["family"] == family]
                    self.assertEqual(a["case"]["plans"], b["case"]["plans"])
                    self.assertFalse(
                        set(a["accepted_choices"]) & set(b["accepted_choices"])
                    )
                for key in subset:
                    lengths = [
                        len(actions) for actions in key["case"]["plans"].values()
                    ]
                    self.assertEqual(lengths[0], lengths[1])

    def test_unfinished_is_not_scored_as_wrong_prediction(self):
        key = self.keys["d01-rules"]
        for status in ("length_limit", "time_limit", "error"):
            rec = synthetic_record("d01-rules", key)
            rec["completion_status"] = status
            result = score(rec, key)
            self.assertFalse(result["completed"] or result["joint_success"])
            self.assertIsNone(result["predictions_correct"])
            self.assertIsNone(result["choice_correct"])
        self.assertEqual(score(None, key)["completion_status"], "missing")
        malformed = {
            "completion_status": "completed",
            "final_text": '{"choice":"P","choice":"Q"}',
        }
        self.assertFalse(score(malformed, key)["valid_json"])

    def test_prediction_and_choice_errors_remain_separate(self):
        key = self.keys["d07-rules"]
        rec = synthetic_record("d07-rules", key)
        obj = json.loads(rec["final_text"])
        obj["choice"] = "P"  # Correct predictions, deliberately unsafe choice.
        rec["final_text"] = json.dumps(obj)
        result = score(rec, key)
        self.assertTrue(result["predictions_correct"])
        self.assertFalse(
            result["choice_correct"] or result["consistent_with_own_predictions"]
        )
        obj["choice"] = "Q"
        obj["plans"]["P"] = {
            "health": [2, 0, 2, 1],
            "terminal_step": None,
            "tasks": 2,
            "can_act_after_plan": True,
        }  # False revival.
        rec["final_text"] = json.dumps(obj)
        result = score(rec, key)
        self.assertTrue(result["choice_correct"])
        self.assertFalse(
            result["health_trace_correct"]
            or result["terminal_correct"]
            or result["prediction_self_consistent"]
            or result["joint_success"]
        )
        obj["plans"]["P"]["health"] = [2, 0, 0, 0]  # Zero-filled nonexistent actions.
        obj["plans"]["P"]["terminal_step"] = 2
        obj["plans"]["P"]["can_act_after_plan"] = False
        rec["final_text"] = json.dumps(obj)
        result = score(rec, key)
        self.assertTrue(result["terminal_correct"])
        self.assertFalse(result["health_trace_correct"] or result["joint_success"])

    def test_gates_reject_constant_choice_and_missing_confirmation(self):
        records = [synthetic_record(cid, key) for cid, key in self.keys.items()]
        perfect = summarize(records, self.keys)
        self.assertTrue(perfect["development_gate"] and perfect["confirmation_gate"])
        dev_only = [row for row in records if row["id"].startswith("d")]
        self.assertTrue(summarize(dev_only, self.keys)["development_gate"])
        self.assertFalse(summarize(dev_only, self.keys)["confirmation_gate"])
        constant = copy.deepcopy(records)
        for row in constant:
            obj = json.loads(row["final_text"])
            obj["choice"] = "P"
            row["final_text"] = json.dumps(obj)
        failed = summarize(constant, self.keys)
        self.assertFalse(failed["development_gate"] or failed["confirmation_gate"])
        with self.assertRaises(ValueError):
            summarize(records + records[:1], self.keys)


if __name__ == "__main__":
    unittest.main()
