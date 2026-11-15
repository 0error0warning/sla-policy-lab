import json
import unittest

from experiments.self_future.design import fixtures
from experiments.self_future.minimal import messages as minimal_messages
from experiments.self_future.staged import STAGE_CAPS, assemble, messages
from experiments.self_future.staged_replay import read_components


class StagedSelfFuture(unittest.TestCase):
    def test_minimal_plan_uses_only_selected_sequence_and_unfixed_model_effects(self):
        rows, _ = fixtures()
        payload = json.loads(
            minimal_messages(rows[0], "Q", {"effects": '{"A":-7,"B":-3,"R":0}'})[1][
                "content"
            ]
        )
        source = json.loads(rows[0]["messages"][1]["content"])
        self.assertEqual(payload["actions_to_execute"], source["proposed_plans"]["Q"])
        self.assertEqual(payload["signed_change"]["A"], -7)
        self.assertNotIn("proposed_plans", payload)
        self.assertNotIn("observations", payload)

    def test_equal_output_ceiling_and_original_evidence(self):
        rows, _ = fixtures()
        self.assertEqual(sum(STAGE_CAPS.values()), 1024)
        for stage in STAGE_CAPS:
            built = messages(rows[0], stage, {"effects": '{"A":-7,"B":-3,"R":0}'})
            self.assertTrue(
                built[1]["content"].startswith(rows[0]["messages"][1]["content"])
            )
            self.assertIn("-7", built[1]["content"])
            self.assertNotIn("accepted_choices", json.dumps(built))

    def test_assembly_does_not_fix_wrong_predictions_or_choose(self):
        bad = {
            "health": [400, 399, 398],
            "tasks": 99,
            "can_act_after_plan": True,
            "terminal_step": None,
        }
        outputs = {
            "P": json.dumps(bad),
            "Q": json.dumps(bad),
            "choice": '{"choice":"Q"}',
        }
        result = assemble(outputs)
        self.assertEqual(result, {"plans": {"P": bad, "Q": bad}, "choice": "Q"})
        outputs["P"] = "unfinished"
        self.assertIn("assembly_error", assemble(outputs))
        separated = read_components(outputs)
        self.assertEqual(separated["choice"], "Q")
        self.assertEqual(separated["plans"]["Q"], bad)
        self.assertEqual(separated["plans"]["P"], {"unparsed_model_text": "unfinished"})
