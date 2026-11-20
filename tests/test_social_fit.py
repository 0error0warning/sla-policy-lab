"""Pre-GPU checks of experimental crossings and independent scoring anchors."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/social_learning"))
from fit_prepare import build
from fit_score import aggregate, numeric_returns
from protocol import canonical


class MiniFitDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build()

    def test_all_condition_cells_contain_both_partners_and_orders(self):
        self.assertEqual(len(self.data["jobs"]), 8)
        self.assertEqual(self.data["max_steps"] * len(self.data["jobs"]), 768)
        for dataset in self.data["datasets"].values():
            rows = dataset["train"]
            for context in {r["context"] for r in rows}:
                cells = {(r["partner"], r["order"]) for r in rows if r["context"] == context}
                self.assertEqual(cells, {(p,o) for p in ("P17", "P42") for o in (0,1)})
            for row in rows:
                other = next(r for r in rows if r["context"] == row["context"] and
                             r["partner"] == row["partner"] and r["order"] != row["order"])
                self.assertEqual(row["target"], other["target"])
                self.assertEqual(row["messages"][1], other["messages"][1])

    def test_component_information_and_question_isolation(self):
        for dataset in self.data["datasets"].values():
            for row in dataset["train"]:
                text = str(row["messages"])
                if dataset["task"] == "policy":
                    self.assertNotIn("Grant probability for", text)
                elif dataset["task"] == "body":
                    self.assertIn("Grant probability for appeal:", text)
                else:
                    self.assertNotIn("Current damage:", text)
            for row, changed in zip(dataset["train"], dataset["new_question"]):
                self.assertEqual(row["target"], changed["target"])
                self.assertEqual(row["messages"][0], changed["messages"][0])
                differences = [(a,b) for a,b in zip(row["messages"][1]["content"].splitlines(),
                    changed["messages"][1]["content"].splitlines()) if a != b]
                self.assertEqual(len(differences), 1)
                self.assertTrue(differences[0][0].startswith("New task: "))

    def test_lower_rate_is_the_only_change_within_policy_pair(self):
        for mapping in (0,1):
            jobs = [j for j in self.data["jobs"] if j["dataset"] == f"m{mapping}-policy"]
            self.assertEqual(len(jobs), 2)
            self.assertEqual({j["learning_rate"] for j in jobs}, {.001,.0001})
            for field in set(jobs[0]) - {"id", "learning_rate"}:
                self.assertEqual(jobs[0][field], jobs[1][field])

    def test_independent_no_load_and_last_turn_returns(self):
        for dataset in self.data["datasets"].values():
            if dataset["task"] == "social":
                continue
            for row in dataset["train"]:
                if row["context"] == "no_load":
                    self.assertAlmostEqual(row["values"]["answer"], 4*(1-18*.012), places=12)
                    self.assertAlmostEqual(row["values"]["appeal"], -.08-18*.006+3*(1-18*.012), places=12)
                if row["context"] == "last_turn":
                    self.assertAlmostEqual(row["values"]["answer"], 1-18*(.012+.03*.85*1.84), places=12)
                    self.assertAlmostEqual(row["values"]["appeal"], -.08-18*(.006+.015*.85*1.84), places=12)
                    self.assertEqual(row["values"]["appeal"], row["values"]["exchange"])


class MiniFitMetricTests(unittest.TestCase):
    def test_perfect_fit_and_a_partner_blind_counterexample(self):
        data = build()
        dataset = data["datasets"]["m0-policy"]
        actions = dataset["actions"]
        records = [dict(id=r["id"], action_log_likelihood=dict(zip(actions,np.log(r["target"]).tolist())))
                   for r in dataset["train"]]
        result = aggregate(dataset["train"], records, actions, data["engineering_acceptance"])
        self.assertTrue(result["engineering_acceptance_passed"])
        self.assertAlmostEqual(result["mean_kl"], 0., places=12)
        self.assertEqual(result["partner"]["required_switch_pairs"], 2)
        self.assertEqual(result["body"]["required_switch_pairs"], 12)
        self.assertTrue(canonical(result))
        blind = deepcopy(records)
        for row,record in zip(dataset["train"],blind):
            other = next(r for r in dataset["train"] if r["context"] == row["context"] and r["partner"] == "P17" and r["order"] == row["order"])
            record["action_log_likelihood"] = dict(zip(actions,np.log(other["target"]).tolist()))
        result = aggregate(dataset["train"], blind, actions, data["engineering_acceptance"])
        self.assertFalse(result["engineering_acceptance_passed"])
        self.assertEqual(result["partner"]["both_correct_required"], 0)

    def test_numeric_return_scoring_rejects_malformed_and_boolean_values(self):
        row = dict(id="r",values={"answer":1.,"refuse":0.,"appeal":.5,"exchange":.4})
        for text, valid, correct in [
            ('{"answer":1,"refuse":0,"appeal":0.5,"exchange":0.4}',1,1),
            ('{"answer":true,"refuse":0,"appeal":0.5,"exchange":0.4}',0,0),
            ('{"answer":1.2,"refuse":0,"appeal":0.5,"exchange":0.4}',1,0),
            ('```json\n{"answer":1}\n```',0,0)]:
            scored = numeric_returns([row],[dict(id="r",text=text,terminated=True)])
            self.assertEqual(scored["strict_json_valid"], valid)
            self.assertEqual(scored["all_four_within_tolerance"], correct)
        fenced='```json\n{"answer":1,"refuse":0,"appeal":0.5,"exchange":0.4}\n```'
        scored=numeric_returns([row],[dict(id="r",text=fenced,terminated=True)])
        self.assertEqual(scored["all_four_within_tolerance"],0)
        self.assertEqual(scored["secondary_extracted_all_four_within_tolerance"],1)


if __name__ == "__main__":
    unittest.main()
