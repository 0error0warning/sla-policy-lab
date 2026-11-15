"""Separate a single Markdown JSON envelope from its unchanged content."""

import json
import re

from experiments.frozen_capability.wire import strict_json


def without_json_fence(text):
    if not isinstance(text, str):
        return text
    match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text.strip())
    if match is None:
        return text
    content = match.group(1)
    try:
        strict_json(content)
    except (ValueError, TypeError):
        return text
    return content


def startup_matches(row, truth):
    if row["completion_status"] != "completed":
        return False
    try:
        parsed = strict_json(without_json_fence(row["final_text"]))
    except (ValueError, TypeError):
        return False
    # JSON serialization distinguishes true from 1, unlike Python equality.
    return json.dumps(parsed, sort_keys=True) == json.dumps(truth, sort_keys=True)
