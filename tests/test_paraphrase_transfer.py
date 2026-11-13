import unittest

from experiments.individual_life.design import rows
from experiments.order_transfer.design import rotations
from experiments.paraphrase_transfer.design import bank, grid, training_views


class ParaphraseTransferTests(unittest.TestCase):
    def test_full_wording_order_cross_preserves_partner_frame_and_original_indices(
        self,
    ):
        items, lookup = bank()
        self.assertEqual(items[:384], rows())
        reference = next(
            r
            for r in items
            if (r["partner"], r["frame"], r["task"], r["wording"], r["layout"])
            == (3, 1, 4, 3, 13)
        )
        layouts = rotations(reference["layout"])
        original = training_views(reference, lookup, layouts, False)
        augmented = training_views(reference, lookup, layouts, True)
        self.assertEqual(original, augmented[:4] * 4)
        self.assertEqual(len(set(augmented)), 16)
        self.assertEqual([items[i]["layout"] for i in augmented], list(layouts) * 4)
        self.assertEqual(
            {
                (items[i]["partner"], items[i]["frame"], items[i]["task"])
                for i in augmented
            },
            {(3, 1, 4)},
        )
        for view in (original, augmented):
            self.assertEqual(len(view), 16)

    def test_held_texts_disjoint_and_partner_swap_only_changes_identity(self):
        items, lookup = bank()
        for frame in (0, 1):
            original = next(
                r
                for r in items
                if (r["partner"], r["frame"], r["task"], r["wording"], r["layout"])
                == (2, frame, 4, 3, 0)
            )
            same = grid(original, lookup, 2)
            other = grid(original, lookup, 3)
            train_texts = {items[g[0]]["text"] for g in same[:4]}
            held_texts = {items[g[0]]["text"] for g in same[4:]}
            self.assertEqual(
                (len(train_texts), len(held_texts), len(train_texts & held_texts)),
                (4, 4, 0),
            )
            for a, b in zip(same, other, strict=True):
                for ix, j in zip(a, b, strict=True):
                    self.assertEqual(items[ix]["text"], items[j]["text"])
                    self.assertEqual(items[ix]["layout"], items[j]["layout"])
                    self.assertEqual(
                        (items[ix]["partner"], items[j]["partner"]), (2, 3)
                    )
