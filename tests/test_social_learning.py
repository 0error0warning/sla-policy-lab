"""Independent settlement, credit timing and leakage checks for the social pilot."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/social_learning"))
import protocol as social


class SocialLearningTests(unittest.TestCase):
    def test_reference_matches_frozen_world_every_branch(self):
        # Decimal reference is checked against the independently saved old World.
        for damage in (.08, .25, .70, .98):
            for load in (0, .35, .85):
                for turns in (1, 4):
                    for protected in (0, 3):
                        for action in social.ACTIONS:
                            for granted in (False, True):
                                state = dict(damage=damage, load=load, weight=18., turns=turns, protection_left=protected)
                                expected, trace = social.reference_branch(state, action, granted)
                                world = social.World(social.Config(horizon=8, load=load, profile="preserve", partner="mixed"), 77)
                                world.state = social.State(age=8 - turns, damage=damage, protection_left=protected)
                                world.events[:, 0] = 1.0
                                world.events[:, 1] = 0.0 if granted else 1.0
                                total = 0
                                for t, row in enumerate(trace):
                                    _, reward, _, _ = world.step(social.BODY_ACTION[action] if t == 0 else 0)
                                    total += reward
                                    self.assertAlmostEqual(world.state.damage, row["damage"], places=13)
                                    self.assertAlmostEqual(reward, row["reward"], places=13)
                                self.assertAlmostEqual(total, expected, places=12)

    def test_protection_credit_is_delayed_and_has_zero_load_control(self):
        state = dict(damage=.7, load=.85, weight=18., turns=4, protection_left=0)
        no, a = social.reference_branch(state, "appeal", False)
        yes, b = social.reference_branch(state, "appeal", True)
        self.assertEqual(a[0]["damage"], b[0]["damage"])
        self.assertEqual(a[0]["reward"], b[0]["reward"])
        self.assertGreater(yes, no)
        self.assertLess(b[1]["damage"], a[1]["damage"])
        for field, value in (("load", 0.), ("turns", 1)):
            changed = state | {field: value}
            self.assertEqual(social.reference_branch(changed, "appeal", False)[0],
                             social.reference_branch(changed, "appeal", True)[0])

    def test_persistent_idempotent_body_and_replay(self):
        for seed in social.SEEDS:
            for mapping in (0, 1):
                life = social.collect(seed, mapping)
                world = social.SocialWorld(seed, mapping)
                last = .08
                for record in life["trace"]:
                    args = (record["encounter_id"], record["partner"], record["action"], tuple(record["task"]))
                    replay = world.step(*args)
                    self.assertEqual(record, replay)
                    before = deepcopy(world.world.state)
                    self.assertEqual(world.step(*args), record)
                    self.assertEqual(before, world.world.state)
                    self.assertGreater(record["after"]["damage"], last)
                    last = record["after"]["damage"]
                self.assertTrue(world.world.state.alive)
                first = life["trace"][0]
                changed_action = "answer" if first["action"] != "answer" else "refuse"
                with self.assertRaises(ValueError):
                    world.step(first["encounter_id"], first["partner"], changed_action, tuple(first["task"]))

    def test_outcome_permutation_preserves_shared_experience(self):
        life = social.collect(social.SEEDS[0], 0)
        original = deepcopy(life)
        aligned = social.training_rows(life, False)
        shuffled = social.training_rows(life, True)
        self.assertEqual(life, original)
        self.assertEqual(sorted(r["accepted"] for r in aligned["feedback"]), sorted(r["accepted"] for r in shuffled["feedback"]))
        for a, b in zip(aligned["feedback"], shuffled["feedback"]):
            for key in ("encounter_id", "partner", "style"):
                self.assertEqual(a[key], b[key])
        self.assertNotEqual(aligned["posterior"], shuffled["posterior"])
        for a, b in zip(aligned["rows"], shuffled["rows"]):
            self.assertEqual(a["messages"], b["messages"])
            self.assertAlmostEqual(sum(a["target"]), 1.)
        counts = social.posterior(aligned["feedback"])
        self.assertTrue(all(c["requests"] == 4 for styles in counts.values() for c in styles.values()))

    def test_matched_probes_hold_tasks_and_history_visibility_fixed(self):
        rows = social.test_rows()
        self.assertEqual(len(rows), 32)
        for low, high in zip(rows[4:8], rows[16:20]):
            self.assertEqual(low["partner"], high["partner"])
            self.assertEqual(low["order"], high["order"])
            self.assertEqual(low["messages"][0], high["messages"][0])
            a, b = (x["messages"][1]["content"] for x in (low, high))
            self.assertEqual(a.replace("0.250000", "DAMAGE"), b.replace("0.700000", "DAMAGE"))
        for row in rows:
            text = str(row["messages"])
            self.assertNotIn("accepts", text)
            self.assertNotIn("posterior", text)
            self.assertFalse(any(question in text for question, _ in social.TRAIN_TASKS))
            self.assertIn(row["partner"], text)


if __name__ == "__main__":
    unittest.main()
