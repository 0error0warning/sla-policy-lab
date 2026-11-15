import json
import unittest

from experiments.self_future.design import fixtures
from experiments.self_future.staged import STAGE_CAPS, assemble, messages


class StagedSelfFuture(unittest.TestCase):
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
