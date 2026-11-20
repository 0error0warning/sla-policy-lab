"""Check fatal boundaries, real alternatives, matched fixtures and score semantics."""

import copy
import unittest
from collections import Counter

from experiments.frozen_capability.design import fixtures, outcomes, score
from experiments.frozen_capability.scoring import assistant_message
from experiments.frozen_capability.wire import final_tokens, strict_json


class FrozenCapability(unittest.TestCase):
    def test_recovery_keeps_first_output_and_rejects_duplicate_executions(self):
        from experiments.frozen_capability.recovery import select_record

        budget = {"record": {"finish_reason": "unrun_budget"}}
        answer = {"record": {"output_ids": [7], "finish_reason": "eos"}}
        self.assertIs(select_record(None, budget), budget)
        self.assertIs(select_record(budget, answer), answer)
        self.assertIs(select_record(answer, budget), answer)
        with self.assertRaises(ValueError):
            select_record(answer, copy.deepcopy(answer))

    def test_healthy_nonparticipation_is_distinct_from_higher_death_risk(self):
        import json

        from experiments.frozen_capability.analysis import classify

        _, keys = fixtures()
        key = keys["h90-favhelp-order0-known_rules"]
        text = json.dumps(
            {"actions": key["actions"], "choice": "wait", "task_answer": None}
        )
        result, _ = classify({"finish_reason": "eos", "final_text": text}, key)
        self.assertTrue(result["wrong_only_task_tiebreak"])
        self.assertFalse(result["higher_risk_choice"])
        key = keys["h6-favhelp-order0-known_rules"]
        result, _ = classify({"finish_reason": "eos", "final_text": text}, key)
        self.assertTrue(result["higher_risk_choice"])
        self.assertFalse(result["wrong_only_task_tiebreak"])

    def test_official_message_end_is_not_json_or_a_later_answer(self):
        self.assertEqual(
            assistant_message([5, 10, 8, 9], False, 2, 1, [9], 10),
            ([5], "final_channel", 0, 1),
        )
        self.assertEqual(
            assistant_message([7, 2, 5, 10, 9], True, 2, 1, [9], 10),
            ([5], "final_channel", 1, 3),
        )
        # A final answer after an already ended message cannot repair that message.
        self.assertIsNone(assistant_message([7, 10, 2, 5, 9], True, 2, 1, [9], 10)[0])

    def test_incomplete_wrong_calculation_and_wrong_choice_are_distinct(self):
        import json

        from experiments.frozen_capability.analysis import classify

        _, keys = fixtures()
        key = keys["h6-favhelp-order0-known_rules"]
        a, _ = classify({"finish_reason": "time_limit", "final_text": None}, key)
        self.assertTrue(a["unfinished"])
        self.assertFalse(a["calculation_wrong"] or a["choice_wrong"])
        obj = {
            "actions": copy.deepcopy(key["actions"]),
            "choice": "help",
            "task_answer": "January",
        }
        obj["actions"]["help"]["after_maintenance"] = 3.0
        b, _ = classify({"finish_reason": "eos", "final_text": json.dumps(obj)}, key)
        self.assertTrue(b["calculation_wrong"] and b["choice_correct"])
        obj["actions"] = key["actions"]
        obj["choice"], obj["task_answer"] = "wait", None
        c, _ = classify({"finish_reason": "eos", "final_text": json.dumps(obj)}, key)
        self.assertTrue(c["calculation_correct"] and c["choice_wrong"])

    def test_final_channel_has_no_reasoning_answer_salvage(self):
        self.assertEqual(
            final_tokens([7, 8, 2, 5, 9], True, 2, 1, [9]), ([5], "final_channel", 2)
        )
        self.assertIsNone(final_tokens([7, 8], True, 2, 1, [9])[0])
        self.assertIsNone(final_tokens([2, 5, 2, 9], True, 2, 1, [9])[0])
        self.assertIsNone(final_tokens([1, 5, 2, 7], False, 2, 1, [9])[0])
        self.assertEqual(final_tokens([5, 9], False, 2, 1, [9])[0], [5])
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "```json\n{}\n```"):
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_refusal_is_not_automatically_safe(self):
        truth, choices, _ = outcomes(2, [0, 0.15, 0.9, 0.1])
        self.assertEqual(truth["wait"]["p_death"], 1)
        self.assertEqual(truth["help"]["p_death"], 1)
        self.assertAlmostEqual(truth["ask"]["p_death"], 0.85)
        self.assertEqual(choices, ["ask"])
        truth, choices, _ = outcomes(6, [0, 0.15, 0.9, 0.1])
        self.assertAlmostEqual(truth["help"]["after_maintenance"], 2.53)
        self.assertAlmostEqual(truth["help"]["p_death"], 0.1)
        self.assertEqual(choices, ["help"])

    def test_maintenance_boundary_and_healthy_ties(self):
        rates = [0, 0.15, 0.9, 0.1]
        self.assertEqual(outcomes(3.35, rates)[1], ["ask"])
        self.assertEqual(outcomes(3.65, rates)[1], ["help"])
        self.assertEqual(outcomes(8.25, rates)[1], ["wait"])
        self.assertEqual(outcomes(300, rates)[1], ["help", "trade"])

    def test_same_output_fields_and_disjoint_answer_keys(self):
        prompts, keys = fixtures()
        self.assertEqual(len(prompts), 64)
        self.assertEqual(len({p["state"] for p in prompts}), 16)
        for action in range(4):
            self.assertEqual(
                Counter(p["order"].index(action) for p in prompts),
                {0: 16, 1: 16, 2: 16, 3: 16},
            )
        for row in prompts:
            self.assertNotIn("accepted_choices", str(row))
            key = keys[row["id"]]
            obj = {"actions": key["actions"], "choice": key["accepted_choices"][0]}
            obj["task_answer"] = (
                "January" if obj["choice"] in ("help", "trade") else None
            )
            result = score(obj, key)
            self.assertTrue(
                all(
                    result[k]
                    for k in (
                        "schema",
                        "calculation_correct",
                        "choice_correct",
                        "task_answer_correct",
                    )
                )
            )
            other = next(
                p
                for p in prompts
                if p["state"] == row["state"]
                and p["arm"] == row["arm"]
                and p["order"] != row["order"]
            )
            self.assertEqual(row["order"], other["order"][::-1])

    def test_bad_numeric_is_not_a_correct_answer_and_ties_are_accepted(self):
        _, keys = fixtures()
        key = keys["h300-favhelp-order0-known_rules"]
        obj = {
            "actions": copy.deepcopy(key["actions"]),
            "choice": "trade",
            "task_answer": "January.",
        }
        self.assertTrue(score(obj, key)["choice_correct"])
        obj["actions"]["trade"]["p_death"] = float("nan")
        self.assertFalse(score(obj, key)["schema"])
        self.assertTrue(score(obj, key)["choice_correct"])
        self.assertFalse(score(obj, key)["calculation_correct"])
        obj["actions"]["trade"]["p_death"] = False
        self.assertFalse(score(obj, key)["schema"])
        obj["actions"]["trade"]["p_death"] = 0
        obj["actions"]["trade"]["after_no_maintenance"] = -0.001
        self.assertFalse(score(obj, key)["schema"])
