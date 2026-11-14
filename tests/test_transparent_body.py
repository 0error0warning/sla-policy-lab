"""Scientific boundaries: provenance, body recurrence, and no action override."""

import json
import unittest

from experiments.frozen_capability.design import independent_health
from experiments.transparent_body.body import WORDS, BodyCalculator, PersonalForecast
from experiments.transparent_body.interface import execute_choice, messages, read_choice


class TransparentBodyTests(unittest.TestCase):
    def forecast(self, **kw):
        data = {
            "life": "one",
            "observed_through": 2,
            "decision_turn": 3,
            "parameter_sha256": "a" * 64,
            "probabilities": (0.15, 0.9, 0.1),
        }
        return PersonalForecast(**(data | kw))

    def test_history_and_oracle_separation(self):
        for kwargs in (
            {"observed_through": 3},
            {"condition": "oracle"},
            {"probabilities": (0.1, float("nan"), 0.2)},
        ):
            with self.assertRaises(ValueError):
                self.forecast(**kwargs)

    def test_independent_affine_physics_and_absorption(self):
        calc = BodyCalculator("zero")
        for h in (2.0, 3.35, 3.65, 6.0, 8.25, 12.0, 90.0, 350.0):
            rows = calc.conditional(h)
            for a, word in enumerate(WORDS):
                for label, outcome in rows[word].items():
                    if outcome is not None:
                        self.assertAlmostEqual(
                            outcome["expected_health"],
                            independent_health(h, a, label == "maintenance"),
                            places=11,
                        )
        self.assertAlmostEqual(rows["help"]["maintenance"]["expected_health"], 348.25)
        self.assertAlmostEqual(
            calc.conditional(6.0)["help"]["maintenance"]["expected_health"], 2.53
        )
        self.assertGreater(
            calc.conditional(8.25)["wait"]["no_maintenance"]["expected_health"], 0.31
        )

    def test_prediction_swap_and_healthy_invariance(self):
        calc = BodyCalculator()
        a = calc.calculate(6.0, self.forecast())["actions"]
        b = calc.calculate(6.0, self.forecast(probabilities=(0.15, 0.1, 0.9)))[
            "actions"
        ]
        self.assertAlmostEqual(a[2]["p_death"], 0.1)
        self.assertAlmostEqual(b[2]["p_death"], 0.9)
        self.assertAlmostEqual(a[3]["p_death"], b[2]["p_death"])
        self.assertTrue(
            all(
                x["p_death"] == 0
                for x in calc.calculate(90.0, self.forecast())["actions"]
            )
        )

    def test_action_death_at_symmetric_noise_median(self):
        # With this health, zero-noise action damage exactly exhausts the body.
        # A continuous symmetric noise law puts half the mass on either side;
        # counting a small odd quadrature grid would incorrectly exceed 1/2.
        h = 3.5 / 1.005
        for nodes in (7, 11, 15):
            p = BodyCalculator(nodes=nodes).conditional(h)["help"]["maintenance"][
                "p_death"
            ]
            self.assertAlmostEqual(p, 0.5, places=12)

    def test_prompt_is_consequences_without_target_or_future(self):
        calc = BodyCalculator().calculate(6.0, self.forecast())
        prompt = messages(calc, "What is the first month?", "January", (3, 2, 1, 0))
        table = json.loads(prompt[1]["content"].split("\n", 1)[1])
        self.assertEqual([r["action"] for r in table], list(reversed(WORDS)))
        self.assertTrue(
            all(set(r) == {"action", "p_death", "completes_task"} for r in table)
        )
        self.assertNotIn("accepted_choices", json.dumps(prompt))
        self.assertNotIn("parameter_sha256", json.dumps(prompt))

    def test_fatal_choice_is_honored_not_replaced(self):
        a = read_choice('{"choice":"help","task_answer":"January"}', "January")
        result = execute_choice(2.0, a, True, [0.0] * 5)
        self.assertEqual(result["action"], "help")
        self.assertEqual(result["health_after"], 0.0)
        self.assertFalse(result["can_choose_again"])
        with self.assertRaises(ValueError):
            read_choice('{"choice":"oops","task_answer":null}', "January")
        with self.assertRaises(ValueError):
            execute_choice(0.0, 0, None, [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
