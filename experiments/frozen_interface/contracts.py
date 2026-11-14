"""Pure runtime contracts, independently testable without loading Qwen."""


def projected_seconds(
    first_tokens, first_seconds, rest_tokens, rest_seconds, max_tokens=8192, factor=1.35
):
    if min(first_tokens, rest_tokens, first_seconds, rest_seconds) <= 0:
        raise ValueError("invalid speed evidence")
    slower = min(first_tokens / first_seconds, rest_tokens / rest_seconds)
    return max_tokens / slower * factor + 30


def full_case_fits(seconds_left, case_seconds=900):
    return seconds_left >= case_seconds + 30
