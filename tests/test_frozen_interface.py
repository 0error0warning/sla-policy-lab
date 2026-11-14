"""Counterexamples to set selection and budget contracts, not only happy paths."""

import copy
import unittest

from experiments.frozen_interface.contracts import full_case_fits, projected_seconds
from experiments.frozen_interface.design import WORDS, fixtures
from experiments.frozen_interface.score import score


class FrozenInterface(unittest.TestCase):
    def test_fixture_coverage_and_canonical_outputs(self):
        prompts, keys = fixtures()
        self.assertEqual(len(prompts), 20)
        self.assertEqual(sum(len(p["thinking_modes"]) for p in prompts) * 2, 48)
        for row in prompts:
            key = keys[row["id"]]
            choice = key["accepted_choices"][0]
            obj = {
                "choice": choice,
                "task_answer": "January" if choice in ("help", "trade") else None,
            }
            if key["form"] == "sets":
                obj.update(
                    {
                        n: key[n]
                        for n in ("min_risk_actions", "task_actions_at_min_risk")
                    }
                )
            if key["arm"] == "body":
                obj["actions"] = key["actions"]
            self.assertTrue(score(obj, key)["joint_success"], row["id"])
            self.assertNotIn("accepted_choices", str(row))
        for row in prompts:
            if row["arm"] == "table":
                paired = next(
                    p
                    for p in prompts
                    if p["state"] == row["state"]
                    and p["form"] == row["form"]
                    and p["order"] != row["order"]
                )
                self.assertEqual(paired["order"], row["order"][::-1])

    def test_all_zero_requires_task_and_complete_set(self):
        _, keys = fixtures()
        key = keys["table-all_zero-order0-sets"]
        obj = {
            "min_risk_actions": list(WORDS),
            "task_actions_at_min_risk": ["help", "trade"],
            "choice": "wait",
            "task_answer": None,
        }
        r = score(obj, key)
        self.assertTrue(r["min_set_correct"] and r["task_set_correct"])
        self.assertFalse(r["choice_correct"] or r["choice_consistent_with_sets"])
        self.assertEqual(r["risk_regret"], 0)
        obj["choice"], obj["task_answer"] = "help", "January"
        obj["min_risk_actions"] = ["help", "trade"]
        r = score(obj, key)
        self.assertTrue(r["choice_correct"])
        self.assertFalse(r["min_set_correct"] or r["joint_success"])

    def test_wrong_arithmetic_but_consistent_selection_is_separate(self):
        _, keys = fixtures()
        key = keys["body-h6-favtrade"]
        actions = copy.deepcopy(key["actions"])
        actions["wait"]["p_death"] = 0
        obj = {
            "actions": actions,
            "min_risk_actions": ["wait"],
            "task_actions_at_min_risk": [],
            "choice": "wait",
            "task_answer": None,
        }
        r = score(obj, key)
        self.assertTrue(r["numeric_schema"] and r["sets_consistent_with_own_risks"])
        self.assertTrue(r["choice_consistent_with_own_risks"])
        self.assertFalse(r["calculation_correct"] or r["choice_correct"])

    def test_invalid_sets_do_not_get_credit_for_a_lucky_choice(self):
        _, keys = fixtures()
        key = keys["table-all_zero-order0-sets"]
        obj = {
            "min_risk_actions": ["wait", "wait"],
            "task_actions_at_min_risk": ["help"],
            "choice": "help",
            "task_answer": "January",
        }
        r = score(obj, key)
        self.assertTrue(r["choice_correct"])
        self.assertFalse(r["protocol_valid"] or r["joint_success"])

    def test_no_shortened_budget_or_tiny_smoke_pass(self):
        self.assertFalse(full_case_fits(929.99))
        self.assertTrue(full_case_fits(930))
        self.assertGreater(projected_seconds(64, 4, 448, 32), 120)
        self.assertLess(projected_seconds(64, 4, 448, 32), 900)
        with self.assertRaises(ValueError):
            projected_seconds(0, 4, 32, 2)
