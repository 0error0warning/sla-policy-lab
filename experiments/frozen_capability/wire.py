"""Channel and JSON validation without any task answers or body model."""

import json
import math


def final_tokens(ids, thinking, close_id, open_id, eos_ids):
    ids = list(ids)
    while ids and ids[-1] in eos_ids:
        ids.pop()
    closes = [i for i, token in enumerate(ids) if token == close_id]
    opens = [i for i, token in enumerate(ids) if token == open_id]
    if thinking:
        if len(closes) != 1 or opens:
            return None, "missing_or_ambiguous_thinking_end", len(ids)
        return ids[closes[0] + 1 :], "final_channel", closes[0]
    if closes or opens:
        return None, "unexpected_thinking_channel", 0
    return ids, "final_channel", 0


def strict_json(text):
    def pairs(values):
        obj = {}
        for key, value in values:
            if key in obj:
                raise ValueError("duplicate_json_key")
            obj[key] = value
        return obj

    def constant(_):
        raise ValueError("nonfinite_json_number")

    def floating(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("nonfinite_json_number")
        return value

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating
    )
