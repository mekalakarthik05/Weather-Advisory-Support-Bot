"""Intake node: extracts user intent into validated Pydantic model."""

import json
import re
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import get_llm
from backend.models import UserIntent

INTAKE_SYSTEM_PROMPT = """You are an intent extraction engine for a weather advisory safety system.
Analyze the user's message and extract:
- location: The target city or location (string). If not mentioned, look at session context facts or return empty string.
- activity: The activity or action mentioned (e.g. cycling, running, picnic, travel, walking, park visit).
- person: Who the activity is for if specified (e.g. child, elderly, pet, adult, or null).
- time_window: The time mentioned (e.g. today, this evening, tomorrow morning, or null).
- original_query: The user's exact query.

CRITICAL INSTRUCTIONS:
- You must reply with ONLY a single valid JSON object.
- Do NOT wrap in markdown ticks (e.g. no ```json).
- Never invent locations not mentioned in the query or context.
"""


def extract_json_from_text(text: str) -> dict:
    """Safely extract JSON object from LLM response text."""
    text = text.strip()
    # Strip markdown block if model included it
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def intake_node(
    message: str,
    prior_facts: Optional[Dict[str, Any]] = None,
    llm: Any = None,
) -> UserIntent:
    """Extract structured intent from user message and prior session facts.

    Raises ValueError if parsing or validation fails.
    """
    if not message or not message.strip():
        raise ValueError("Empty user message cannot be processed.")

    if llm is None:
        llm = get_llm()

    context_str = ""
    if prior_facts:
        context_str = f"\nPrior established facts in this session: {json.dumps(prior_facts)}"

    prompt = f"User message: {message}{context_str}\n\nExtract intent as JSON with keys 'location', 'activity', 'person', 'time_window', 'original_query':"

    try:
        response = llm.invoke([
            SystemMessage(content=INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = extract_json_from_text(raw_text)
    except Exception as e:
        raise ValueError(f"Failed to extract structured intent: {e}") from e

    # If location or activity missing in current turn, check prior_facts
    if prior_facts:
        if (not parsed.get("location") or parsed.get("location") == "Unknown") and prior_facts.get("location"):
            parsed["location"] = prior_facts["location"]
        if (not parsed.get("activity") or parsed.get("activity") == "general") and prior_facts.get("activity"):
            parsed["activity"] = prior_facts["activity"]
        if not parsed.get("person") and prior_facts.get("person"):
            parsed["person"] = prior_facts["person"]

    parsed["original_query"] = message

    # Validate with Pydantic
    return UserIntent(**parsed)
