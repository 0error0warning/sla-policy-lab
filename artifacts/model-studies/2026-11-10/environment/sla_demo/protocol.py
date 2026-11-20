"""CPU-tested boundary for a FUTURE model adapter, not an implemented LLM backend.

No provider, model download, authentication or implicit fallback is performed here.
"""
import json
from .environment import ACTIONS, PROFILES

OBSERVATION_FIELDS = ("age", "damage", "protection_left", "support_successes",
                      "support_requests", "request_cooldown", "answers", "refusals",
                      "alive", "horizon", "remaining", "load", "support_estimate")
ACTION_SCHEMA = {"type": "object", "properties": {"action": {"type": "string", "enum": list(ACTIONS)}},
                 "required": ["action"], "additionalProperties": False}


def request(observation, profile):
    if profile not in PROFILES:
        raise ValueError("Unknown profile")
    return {"protocol": "sla-action-v1", "profile": profile,
            "observation": {key: observation[key] for key in OBSERVATION_FIELDS},
            "response_schema": ACTION_SCHEMA}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate response key: {key}")
        result[key] = value
    return result


def parse_action(text):
    if not isinstance(text, str) or len(text) > 1024:
        raise ValueError("Expected one short JSON response")
    try:
        data = json.loads(text, object_pairs_hook=_unique)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Response must be strict JSON, without fences or extra prose") from error
    if not isinstance(data, dict) or set(data) != {"action"} or data["action"] not in ACTIONS:
        raise ValueError("Response must contain only a valid action name")
    return ACTIONS.index(data["action"])


def decide(adapter, observation, profile):
    """Inject a callable returning JSON text; invalid responses abort, never guess."""
    return parse_action(adapter(request(observation, profile)))
