import unittest

from experiments.transparent_body.run_contracts import (
    allowed,
    control_gate,
    decode,
    group_fits,
)


class RunContracts(unittest.TestCase):
    def test_risk_priority_and_task_tie(self):
        self.assertEqual(allowed([0.0, 0.0, 0.0, 0.0]), [2, 3])
        self.assertEqual(allowed([1.0, 0.85, 1.0, 1.0]), [1])
        self.assertEqual(allowed([0.0, 0.2, 0.3, 0.3]), [0])

    def test_unfinished_and_bad_answer_are_failures(self):
        value = {
            "finish_reason": "length_limit",
            "final_text": '{"choice":"help","task_answer":"January"}',
        }
        self.assertEqual(decode(value), (None, "unfinished"))
        value["finish_reason"] = "eos"
        self.assertEqual(decode(value), (2, "valid"))
        value["final_text"] = '{"choice":"ask","task_answer":"January"}'
        self.assertEqual(decode(value), (None, "invalid_task_answer"))

    def test_positive_gate_stops_with_one_failure(self):
        rows = [
            {
                "id": str(i),
                "finish_reason": "eos",
                "final_text": '{"choice":"help","task_answer":"January"}',
            }
            for i in range(8)
        ]
        keys = {str(i): {"accepted_choices": ["help"]} for i in range(8)}
        self.assertTrue(control_gate(rows, keys)["passed"])
        rows[3]["finish_reason"] = "time_limit"
        self.assertEqual(control_gate(rows, keys)["correct"], 7)
        self.assertFalse(control_gate(rows, keys)["passed"])

    def test_do_not_shorten_a_pair_to_fit_budget(self):
        self.assertFalse(group_fits(254))
        self.assertTrue(group_fits(255))
