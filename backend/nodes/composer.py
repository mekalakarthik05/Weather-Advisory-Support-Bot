"""Composer node: formats grounded user-facing response citing matched SOPs."""

from typing import Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import get_llm
from backend.models import SOP, UserIntent, WeatherData

COMPOSER_SYSTEM_PROMPT = """You are a policy-grounded Weather Advisory Communication Assistant.
Your sole job is to communicate the provided safety policy clearly to the user.

CRITICAL NON-NEGOTIABLES:
1. Grounding: Use ONLY the provided live weather numbers. NEVER round, alter, or invent numbers.
   For example, if wind speed is 42.0 km/h, write "42.0 km/h" or "42 km/h". Never say "about 45 km/h" or "around 50 km/h".
2. Safety Policy: Use ONLY the advice text from the supplied SOP. NEVER invent rules, precautions, or safety thresholds.
3. Citation: ALWAYS cite the exact SOP ID and title provided.
4. Response Format:
   Weather: <exact weather numbers>
   Advisory: <clear explanation grounded in SOP advice>
   Policy: <SOP citation(s)>
"""


def format_weather_summary(weather: WeatherData) -> str:
    """Format exact live weather metrics without estimation or rounding drift."""
    return (
        f"{weather.temperature_2m}°C, "
        f"{weather.wind_speed_10m} km/h wind, "
        f"{weather.precipitation_probability:.0f}% precipitation probability, "
        f"{weather.precipitation} mm rain, "
        f"UV index {weather.uv_index}"
    )


def composer_node(
    intent: UserIntent,
    weather: WeatherData,
    primary_sop: SOP,
    matched_sops: List[SOP],
    situational_override: bool = False,
    llm: Any = None,
) -> str:
    """Compose user-facing response grounded in SOP policy and live weather numbers."""
    if not primary_sop:
        from backend.nodes.failure import build_no_match_response
        return build_no_match_response()

    weather_text = format_weather_summary(weather)

    # Collect citations
    citations = [primary_sop.cite_as]
    for s in matched_sops:
        if s.id != primary_sop.id and s.cite_as not in citations:
            citations.append(s.cite_as)
    citations_text = " | ".join(citations)

    # If situational override applies, explicitly format emergency precedence
    if situational_override:
        return (
            f"⚠️ SITUATIONAL OVERRIDE: SEVERE WEATHER ALERT\n\n"
            f"Weather ({weather.location_name}):\n"
            f"{weather_text}\n\n"
            f"Advisory:\n"
            f"{primary_sop.advice.strip()}\n\n"
            f"Policy:\n"
            f"{primary_sop.cite_as} [Severity: {primary_sop.severity.upper()}]"
        )

    # Attempt to format with LLM for natural phrasing while strictly bounded
    if llm is None:
        llm = get_llm()

    # If running with mock LLM or offline, format directly to ensure 100% adherence
    if hasattr(llm, "_generate_response"):
        return (
            f"Weather ({weather.location_name}):\n"
            f"{weather_text}\n\n"
            f"Advisory:\n"
            f"{primary_sop.advice.strip()}\n\n"
            f"Policy:\n"
            f"{citations_text} [Severity: {primary_sop.severity.upper()}]"
        )

    # For live LLM: provide strict context prompt
    user_prompt = (
        f"User Query: {intent.original_query}\n"
        f"Location: {weather.location_name}\n"
        f"Live Weather Numbers: {weather_text}\n"
        f"Applicable SOP Advice: {primary_sop.advice}\n"
        f"SOP Citation: {citations_text}\n\n"
        f"Format the final response strictly with:\n"
        f"Weather:\n"
        f"Advisory:\n"
        f"Policy:\n"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=COMPOSER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        # Ensure policy citation is present in response
        if primary_sop.id not in content:
            content += f"\n\nPolicy:\n{citations_text}"
        return content.strip()
    except Exception:
        # Fallback to direct deterministic formatting
        return (
            f"Weather ({weather.location_name}):\n"
            f"{weather_text}\n\n"
            f"Advisory:\n"
            f"{primary_sop.advice.strip()}\n\n"
            f"Policy:\n"
            f"{citations_text} [Severity: {primary_sop.severity.upper()}]"
        )
