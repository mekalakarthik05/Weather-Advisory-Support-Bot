"""LangGraph Agent Workflow with genuine conditional branching.

Defines the StateGraph with distinct failure, no-match, and composed response paths.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from backend.loader import load_sops_from_dir
from backend.models import SOP, UserIntent, WeatherData
from backend.weather import get_weather_for_location, LocationUnresolved, WeatherAPIError
from backend.nodes.intake import intake_node
from backend.nodes.matcher import match_sops
from backend.nodes.composer import composer_node
from backend.nodes.failure import build_failure_response, build_no_match_response
from backend.memory import get_session_memory, update_session_facts

# Load SOPs once at module level (fails loudly at startup if malformed)
SOPS_DIR = Path(__file__).resolve().parent.parent / "sops"
LOADED_SOPS: List[SOP] = load_sops_from_dir(SOPS_DIR)


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    intent: Optional[Dict[str, Any]]
    weather: Optional[Dict[str, Any]]
    matched_sops: List[Dict[str, Any]]
    primary_sop: Optional[Dict[str, Any]]
    situational_override: bool
    error: Optional[str]
    response: str


# Node Functions

def step_intake(state: AgentState) -> AgentState:
    """Intake node: parses user message into validated UserIntent."""
    msg = state.get("user_message", "").strip()
    session_id = state.get("session_id", "")
    prior_facts = get_session_memory(session_id).get("facts", {}) if session_id else {}

    try:
        intent_obj = intake_node(msg, prior_facts=prior_facts)
        state["intent"] = intent_obj.model_dump()
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        state["intent"] = None

    return state


def step_weather(state: AgentState) -> AgentState:
    """Weather node: resolves geocoding and fetches live Open-Meteo weather."""
    if state.get("error"):
        return state

    intent_dict = state.get("intent")
    if not intent_dict or not intent_dict.get("location"):
        state["error"] = "No location could be identified in the request."
        return state

    location_query = intent_dict["location"]

    try:
        weather_obj = get_weather_for_location(location_query)
        state["weather"] = weather_obj.model_dump()
        # Save established facts in session memory for future turns
        session_id = state.get("session_id", "")
        if session_id:
            update_session_facts(
                session_id,
                location=weather_obj.location_name,
                activity=intent_dict.get("activity"),
                person=intent_dict.get("person"),
            )
    except LocationUnresolved as e:
        state["error"] = str(e)
    except WeatherAPIError as e:
        state["error"] = f"Weather service unavailable: {e}"
    except Exception as e:
        state["error"] = f"Unexpected weather fetch error: {e}"

    return state


def step_matcher(state: AgentState) -> AgentState:
    """Matcher node: evaluates loaded SOPs against live weather and intent."""
    if state.get("error"):
        return state

    intent_dict = state.get("intent")
    weather_dict = state.get("weather")

    if not intent_dict or not weather_dict:
        state["error"] = "Missing intent or weather data for policy matching."
        return state

    intent_obj = UserIntent(**intent_dict)
    weather_obj = WeatherData(**weather_dict)

    matched, primary, override = match_sops(intent_obj, weather_obj, LOADED_SOPS)

    state["matched_sops"] = [s.model_dump() for s in matched]
    state["primary_sop"] = primary.model_dump() if primary else None
    state["situational_override"] = override

    return state


def step_failure_node(state: AgentState) -> AgentState:
    """Deterministic failure response with zero LLM hallucination risk."""
    state["response"] = build_failure_response(state.get("error"))
    return state


def step_no_match_node(state: AgentState) -> AgentState:
    """Deterministic no-match response when no SOP policy applies."""
    state["response"] = build_no_match_response()
    return state


def step_composer(state: AgentState) -> AgentState:
    """Composer node: produces user reply strictly citing matched SOPs and live weather."""
    intent_obj = UserIntent(**state["intent"])
    weather_obj = WeatherData(**state["weather"])
    primary_sop = SOP(**state["primary_sop"]) if state.get("primary_sop") else None
    matched_sops = [SOP(**s) for s in state.get("matched_sops", [])]
    override = state.get("situational_override", False)

    state["response"] = composer_node(
        intent=intent_obj,
        weather=weather_obj,
        primary_sop=primary_sop,
        matched_sops=matched_sops,
        situational_override=override,
    )
    return state


# Routing Decision

def decide_next_step(state: AgentState) -> str:
    """Branching logic enforcing strict safety routing:

    1. Error -> failure_node
    2. Situational override -> composer (with override alert)
    3. Matched SOPs -> composer
    4. Zero matched SOPs -> no_match_node
    """
    if state.get("error"):
        return "failure"
    if not state.get("primary_sop"):
        return "no_match"
    return "composer"


# Construct the StateGraph

def create_advisory_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add processing nodes
    workflow.add_node("intake", step_intake)
    workflow.add_node("weather", step_weather)
    workflow.add_node("matcher", step_matcher)

    # Add destination branch nodes
    workflow.add_node("failure_node", step_failure_node)
    workflow.add_node("no_match_node", step_no_match_node)
    workflow.add_node("composer", step_composer)

    # Connect sequential pipeline
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "weather")
    workflow.add_edge("weather", "matcher")

    # Add conditional branching from matcher
    workflow.add_conditional_edges(
        "matcher",
        decide_next_step,
        {
            "failure": "failure_node",
            "no_match": "no_match_node",
            "composer": "composer",
        },
    )

    # All branches route to END
    workflow.add_edge("failure_node", END)
    workflow.add_edge("no_match_node", END)
    workflow.add_edge("composer", END)

    return workflow.compile()


# Singleton compiled graph
ADVISORY_GRAPH = create_advisory_graph()


def run_advisory_agent(user_message: str, session_id: str = "default_session") -> Dict[str, Any]:
    """Execute the compiled LangGraph agent for a user message."""
    initial_state: AgentState = {
        "session_id": session_id,
        "user_message": user_message,
        "intent": None,
        "weather": None,
        "matched_sops": [],
        "primary_sop": None,
        "situational_override": False,
        "error": None,
        "response": "",
    }
    return ADVISORY_GRAPH.invoke(initial_state)
