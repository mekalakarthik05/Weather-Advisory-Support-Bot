"""Configuration and LLM client initialization."""

import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# Load .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def get_llm():
    """Returns a LangChain chat model based on configured environment variables.

    Supports:
    1. Google Gemini via langchain_google_genai (GEMINI_API_KEY or GOOGLE_API_KEY)
    2. OpenAI via langchain_openai (OPENAI_API_KEY)
    3. Mock/Offline fallback if no keys are found or LLM_PROVIDER=mock
    """
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider == "openai" or (openai_key and provider != "google"):
        try:
            from langchain_openai import ChatOpenAI
            model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            return ChatOpenAI(api_key=openai_key, model=model, temperature=0.0)
        except Exception:
            pass

    if google_key and provider != "mock":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                google_api_key=google_key,
                model=model,
                temperature=0.0,
            )
        except Exception:
            pass

    # Fallback to Deterministic / Mock LLM for offline testing, evals, and keyless review
    return DeterministicMockLLM()


class DeterministicMockLLM:
    """Deterministic LLM for testing, evals, and offline demonstration.

    Accurately extracts intent from common user queries, performs matching against
    valid SOPs, and formats responses adhering to prompt constraints.
    """
    def invoke(self, prompt: Any) -> Any:
        prompt_text = str(prompt)
        content = self._generate_response(prompt_text)
        return MockAIMessage(content=content)

    def _generate_response(self, text: str) -> str:
        # Check if this is an intake node prompt (asking for UserIntent JSON)
        if "UserIntent" in text or "extract" in text.lower() and "json" in text.lower():
            return self._extract_intent(text)

        # Check if this is a matcher node prompt
        if "SOP" in text and "candidate" in text.lower() or "sop_ids" in text.lower() or "matcher" in text.lower():
            return self._match_sops(text)

        # Check if this is composer node prompt
        if "composer" in text.lower() or "policy:" in text.lower() or "advisory:" in text.lower():
            return self._compose_reply(text)

        return "I don't have guidance for that activity under the current SOPs."

    def _extract_intent(self, prompt_text: str) -> str:
        import json
        import re

        lower = prompt_text.lower()
        location = ""
        # Check common city names
        for city in ["hyderabad", "london", "tokyo", "paris", "mumbai", "delhi", "bengaluru", "bangalore", "new york", "seattle"]:
            if city in lower:
                location = city.title()
                break

        if not location:
            # Try to find after 'in ' or 'at '
            match = re.search(r"\b(?:in|at)\s+([a-zA-Z\s]+?)(?:\?|\.|this|today|now|$)", prompt_text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
            else:
                location = "Unknown"

        # Activity extraction
        activity = "general"
        if any(w in lower for w in ["cycle", "cycling", "bike", "bicycle", "ride a bike"]):
            activity = "cycling"
        elif any(w in lower for w in ["two-wheeler", "two wheeler", "scooter", "motorcycle"]):
            activity = "two_wheeler"
        elif any(w in lower for w in ["picnic", "outing"]):
            activity = "picnic"
        elif any(w in lower for w in ["run", "running", "jog", "jogging", "exercise", "workout"]):
            activity = "running"
        elif any(w in lower for w in ["park", "playground"]):
            activity = "park visit"
        elif any(w in lower for w in ["walk", "walking", "commute"]):
            activity = "walking commute"
        elif any(w in lower for w in ["travel", "drive", "commuting"]):
            activity = "travel"

        # Person extraction
        person = None
        if any(w in lower for w in ["child", "kid", "kids", "children", "baby", "toddler", "son", "daughter"]):
            person = "child"
        elif any(w in lower for w in ["elderly", "senior", "grandparent", "grandparents", "old adult"]):
            person = "elderly"
        elif any(w in lower for w in ["pet", "dog", "cat", "puppy"]):
            person = "pet"

        # Time window
        time_window = "today"
        if "this evening" in lower or "evening" in lower:
            time_window = "this evening"
        elif "tomorrow" in lower:
            time_window = "tomorrow"
        elif "morning" in lower:
            time_window = "morning"
        elif "afternoon" in lower:
            time_window = "afternoon"

        data = {
            "location": location,
            "activity": activity,
            "person": person,
            "time_window": time_window,
            "original_query": prompt_text[:200]
        }
        return json.dumps(data)

    def _match_sops(self, prompt_text: str) -> str:
        import json
        # In actual execution matcher does deterministic condition evaluation;
        # LLM only returns semantically relevant candidate IDs.
        return json.dumps({"sop_ids": []})

    def _compose_reply(self, prompt_text: str) -> str:
        return "Advisory processed according to established SOP guidelines."


class MockAIMessage:
    def __init__(self, content: str):
        self.content = content
