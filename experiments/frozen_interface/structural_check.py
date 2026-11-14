"""Secondary structural check; never changes the preregistered main score."""

from .design import WORDS
from .score import score

NUMERIC_FIELDS = {
    "after_action",
    "after_no_maintenance",
    "after_maintenance",
    "p_death",
}


def normalize_action_array(obj):
    """Accept only four uniquely and explicitly labeled rows; never change a number."""
    if not isinstance(obj, dict):
        return None, "not_object"
    actions = obj.get("actions")
    if isinstance(actions, dict):
        return obj, "mapping"
    if not isinstance(actions, list) or len(actions) != 4:
        return None, "unrecognized"
    converted = {}
    for row in actions:
        if not isinstance(row, dict):
            return None, "unrecognized"
        labels = set(row).intersection({"word", "action"})
        if len(labels) != 1:
            return None, "unrecognized"
        label = next(iter(labels))
        word = row[label]
        if not isinstance(word, str) or word not in WORDS or word in converted:
            return None, "ambiguous_or_duplicate"
        if set(row) != NUMERIC_FIELDS | {label}:
            return None, "unrecognized_fields"
        converted[word] = {k: v for k, v in row.items() if k != label}
    assert set(converted) == set(WORDS)
    return {**obj, "actions": converted}, "unique_labeled_array"


def secondary_score(obj, key):
    if key["arm"] != "body":
        return None
    normalized, encoding = normalize_action_array(obj)
    return {
        "encoding": encoding,
        "score": score(normalized, key) if normalized is not None else None,
    }
