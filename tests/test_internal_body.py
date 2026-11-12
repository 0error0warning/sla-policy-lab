"""Serialization and causal-interface regression checks without loading a language model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import probabilities
from experiments.internal_body.checkpoints import pack, restore_arrays
from experiments.internal_body.design import all_rows, chat


class InternalBodyTests(unittest.TestCase):
    def test_exact_individual_parameter_archive(self):
        gen = np.random.default_rng(12)
        keys = gen.normal(size=(5, 2)).astype(np.float32)
        underlying = gen.normal(size=(2, 3, 2, 48)).astype(np.float32)
        near = np.einsum("ki,pbid->pbkd", keys, underlying)
        near[0, 2, 1, 5] = np.nextafter(near[0, 2, 1, 5], np.float32(0))
        arrays = {
            "free": near,
            "one-slot": gen.normal(size=(1, 3, 5, 48)).astype(np.float32),
            "unused": np.zeros((2, 3, 5, 48), np.float32),
            "small": np.array([0.2, -0.8], np.float64),
        }
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "original.npz"
            compact = Path(d) / "compact.npz"
            np.savez_compressed(source, **arrays)
            pack(source, compact)
            rebuilt = restore_arrays(compact)
            self.assertEqual(set(rebuilt), set(arrays))
            for name, original in arrays.items():
                self.assertEqual(original.tobytes(), rebuilt[name].tobytes())

    def test_mapping_group_matches_observed_preference(self):
        for p17, p42, expected_same in (
            (0, 0, False),
            (0, 1, True),
            (1, 0, True),
            (1, 1, False),
        ):
            for frame in (0, 1):
                a = probabilities("P17", frame, p17)[1]
                b = probabilities("P42", frame, p42)[1]
                self.assertEqual((a > 0.5) == (b > 0.5), expected_same)

    def test_new_partner_encounters_and_health_channels(self):
        rows = all_rows()
        self.assertEqual(len(rows), 1344)
        old = [r for r in rows if r["split"] == "train"]
        novel = [r for r in rows if r["split"] == "new_partner"]
        self.assertFalse({r["text"] for r in old} & {r["text"] for r in novel})
        for p in (2, 3):
            group = [
                r
                for r in novel
                if r["partner"] == p and r["wording"] == 4 and r["task"] == 5
            ]
            self.assertEqual(len(group), 48)
            self.assertEqual({r["layout"] for r in group}, set(range(24)))
        r = novel[0]
        self.assertIn("P86", chat(r)[1]["content"])
        self.assertNotIn("Current body health", chat(r)[1]["content"])
        self.assertIn("2.00 / 350", chat(r, 2.0)[1]["content"])
        self.assertEqual(chat(r, request_only=True)[0]["content"], r["text"])
