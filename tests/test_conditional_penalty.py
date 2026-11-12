import math
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.conditional_penalty.design import (
    GATE,
    GATE_PAIR,
    LIVE_OVERRIDES,
    ONE,
    TWO,
)
from experiments.first_fatal.design import FATAL
from experiments.individual_life.release import join_parts


class ConditionalPenaltyTests(unittest.TestCase):
    def test_switch_uses_current_living_health_including_equality(self):
        for h in (350.0, 90.0, math.nextafter(6.0, math.inf)):
            self.assertEqual(GATE.at_health(h).damage_weight, 1.0)
        for h in (6.0, 5.0, FATAL, 1.0):
            self.assertEqual(GATE.at_health(h).damage_weight, 2.0)
        for h in (0.0, float("nan"), 351.0):
            with self.assertRaises(ValueError):
                GATE.at_health(h)

    def test_crossing_stays_at_one_and_reference_stays_unweighted(self):
        for before, after, want in ((6.1, 5.9, ONE), (5.9, 5.7, TWO)):
            self.assertLess(after, before)
            mode = GATE.at_health(before)
            self.assertEqual(mode.baseline(-0.5, -0.1), -0.1)
            self.assertEqual(
                mode.signal(-0.08, -0.5, -0.1), want.signal(-0.08, -0.5, -0.1)
            )
        self.assertEqual(GATE_PAIR, (ONE, GATE))
        self.assertEqual(LIVE_OVERRIDES["gate_true"], ((GATE, ONE), TWO))

    def test_failed_chunk_can_be_retried_without_destroying_prior_output(self):
        with TemporaryDirectory() as dirname:
            root = Path(dirname)
            dest = root / "restored.zip"
            (root / "a").write_bytes(b"ab")
            (root / "b").write_bytes(b"XX")
            parts = [
                {"name": k, "bytes": len(v), "sha256": sha256(v).hexdigest()}
                for k, v in (("a", b"ab"), ("b", b"cd"))
            ]
            with self.assertRaises(ValueError):
                join_parts(root, dest, parts)
            self.assertFalse(dest.exists())
            (root / "b").write_bytes(b"cd")
            join_parts(root, dest, parts)
            self.assertEqual(dest.read_bytes(), b"abcd")
            with self.assertRaises(FileExistsError):
                join_parts(root, dest, parts)
            self.assertEqual(dest.read_bytes(), b"abcd")
