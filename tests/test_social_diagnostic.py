"""Experimental-isolation checks before any saved-weight diagnostic is run."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/social_learning"))
from diagnose_prepare import build
from diagnose_score import metrics, pairing
from protocol import ACTIONS


class DiagnosticDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.key = build()

    def test_existing_checkpoints_and_frozen_coverage(self):
        jobs = self.public["jobs"]
        self.assertEqual(len(jobs), 9)
        self.assertEqual(set(jobs[0]["stimulus_ids"]), set(self.public["stimuli"]))
        self.assertLessEqual(len(jobs[0]["stimulus_ids"]), 416)
        self.assertEqual(sum(len(j["stimulus_ids"]) for j in jobs[1:]), 1024)
        self.assertTrue(all(j["adapter_member"] and j["adapter_tensor_sha256"] for j in jobs[1:]))

    def test_each_condition_changes_only_its_named_input(self):
        stimuli = self.public["stimuli"]
        for dataset in self.key["datasets"].values():
            groups = dataset["groups"]
            for group in ("partner_only", "order_only", "question_only"):
                for original, changed in zip(groups["train_original"], groups[group]):
                    a, b = (stimuli[r["stimulus_id"]]["messages"] for r in (original, changed))
                    self.assertEqual(original["state"], changed["state"])
                    if group == "order_only":
                        self.assertEqual(a[1], b[1])
                        self.assertEqual(a[0]["content"].splitlines()[0], b[0]["content"].splitlines()[0])
                        self.assertEqual(a[0]["content"].splitlines()[1:], list(reversed(b[0]["content"].splitlines()[1:])))
                    else:
                        self.assertEqual(a[0], b[0])
                        differing = [(x, y) for x, y in zip(a[1]["content"].splitlines(), b[1]["content"].splitlines()) if x != y]
                        self.assertEqual(len(differing), 1)
                        self.assertTrue(differing[0][0].startswith("Partner: " if group == "partner_only" else "New task: "))
                    if group != "partner_only":
                        np.testing.assert_allclose(original["target"], changed["target"], rtol=0, atol=1e-12)

    def test_partner_targets_follow_changed_identity(self):
        for dataset in self.key["datasets"].values():
            originals = dataset["groups"]["train_original"]
            for i, changed in enumerate(dataset["groups"]["partner_only"]):
                counterpart = originals[i ^ 1]
                self.assertEqual(changed["partner"], counterpart["partner"])
                np.testing.assert_allclose(changed["target"], counterpart["target"], rtol=0, atol=1e-12)

    def test_aliases_are_deduplicated_without_losing_pairings(self):
        for dataset in self.key["datasets"].values():
            groups = dataset["groups"]
            partner = {r["stimulus_id"] for r in groups["partner_only"]}
            order = {r["stimulus_id"] for r in groups["order_only"]}
            self.assertEqual(partner, order)
            union = {r["stimulus_id"] for rows in groups.values() for r in rows}
            self.assertEqual(len(union), 96)


class DiagnosticMetricTests(unittest.TestCase):
    @staticmethod
    def record(probabilities):
        return {"actions": {a: {"full_log_likelihood": float(np.log(p)), "word_log_likelihood": float(np.log(p))}
                            for a, p in zip(ACTIONS, probabilities)}}

    def test_identical_distribution_has_zero_kl(self):
        q = [.7, .1, .1, .1]
        case = dict(stimulus_id="x", target=q, values=dict(zip(ACTIONS, q)))
        result = metrics([case], {"x": self.record(q)}, np.array(q))
        self.assertAlmostEqual(result["mean_kl"], 0., places=13)
        self.assertEqual(result["teacher_argmax_correct"], 1)
        swapped = metrics([case], {"x": self.record([.1, .7, .1, .1])}, np.array(q))
        self.assertAlmostEqual(swapped["mean_kl"], .6 * np.log(7), places=13)

    def test_partner_switch_requires_both_corresponding_decisions(self):
        a = dict(source_id="same", stimulus_id="a", target=[.1, .1, .7, .1], values=dict(zip(ACTIONS, [.1, .1, .7, .1])))
        b = dict(source_id="same", stimulus_id="b", target=[.1, .1, .1, .7], values=dict(zip(ACTIONS, [.1, .1, .1, .7])))
        records = {"a": self.record(a["target"]), "b": self.record(a["target"])}
        result = pairing([a], [b], records)
        self.assertEqual(result["teacher_requires_switch"], 1)
        self.assertEqual(result["required_switch_both_correct"], 0)
        records["b"] = self.record(b["target"])
        self.assertEqual(pairing([a], [b], records)["required_switch_both_correct"], 1)

    def test_tied_teacher_actions_are_not_arbitrarily_penalized(self):
        case = dict(stimulus_id="x", target=[.1, .1, .4, .4], values=dict(zip(ACTIONS, [.1, .1, .4, .4])))
        result = metrics([case], {"x": self.record([.1, .1, .3, .5])}, np.array(case["target"]))
        self.assertEqual(result["teacher_argmax_correct"], 1)
        self.assertEqual(result["mean_reference_regret"], 0.)


if __name__ == "__main__":
    unittest.main()
