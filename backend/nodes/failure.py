"""Deterministic failure and no-match responses.

Neither failure nor no-match calls an LLM, ensuring zero drift into hallucinated advice.
"""

from typing import Optional


def build_failure_response(error_message: Optional[str] = None) -> str:
    """Return deterministic error response when weather or intake fails."""
    detail = f" ({error_message})" if error_message else ""
    return (
        f"I couldn't retrieve the weather information needed to provide a "
        f"weather-based advisory for this request{detail}. "
        f"Please verify the location and try again."
    )


def build_no_match_response() -> str:
    """Return deterministic response when no SOP applies."""
    return (
        "I don't have guidance for that activity under the current SOPs. "
        "Our safety policy currently covers outdoor exercise (cycling, running), "
        "travel/commute, vulnerable groups (children, seniors, pets), and outdoor picnics."
    )
