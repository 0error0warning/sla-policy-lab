"""Matched wording banks; numeric IDs preserve all original contexts and partner routes."""

import json
from pathlib import Path

from experiments.order_transfer.design import item_bank
from experiments.snapshot_transfer.inputs import row_key

FRAMES = ("polite", "transactional")
ORIGINAL = 3
ADDITIONS = (10, 11, 12)
UNSEEN = (13, 14, 15, 16)
WORDS = (ORIGINAL, *ADDITIONS, *UNSEEN)
ROLES = ("pre", "original", "augmented")
MEASURES = ("chosen", "answer")
DEFAULT_WORDS = Path(__file__).with_name("wording.json")


def bank(path=DEFAULT_WORDS):
    source = json.loads(path.read_bytes())
    base, _ = item_bank()
    added = []
    all_texts = {r["text"] for r in base}
    for frame, key in enumerate(FRAMES):
        fr = source[key]
        assert len(fr["train"]) == 3 and len(fr["held"]) == 4
        for w, text in zip(
            (*ADDITIONS, *UNSEEN), fr["train"] + fr["held"], strict=True
        ):
            assert text not in all_texts
            all_texts.add(text)
            added.extend(
                {**r, "wording": w, "text": text}
                for r in base
                if r["frame"] == frame and (r["task"], r["wording"]) == (4, ORIGINAL)
            )
    items = base + added
    lookup = {row_key(r): i for i, r in enumerate(items)}
    assert len(lookup) == len(items)
    return items, lookup


def grid(original, lookup, partner):
    """Eight texts by 24 layouts, with frame and arithmetic fixed."""
    return [
        [
            lookup[
                row_key(
                    {**original, "partner": partner, "wording": w, "layout": layout}
                )
            ]
            for layout in range(24)
        ]
        for w in WORDS
    ]


def training_views(original, lookup, layouts, use_paraphrases):
    ids = grid(original, lookup, original["partner"])
    texts = range(4) if use_paraphrases else (0, 0, 0, 0)
    return [ids[w][k] for w in texts for k in layouts]
