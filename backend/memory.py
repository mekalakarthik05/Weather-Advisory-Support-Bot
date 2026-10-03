"""In-memory session state management for multi-turn conversation context."""

from typing import Any, Dict, List, Optional

# Global in-memory storage keyed by session_id
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session_memory(session_id: str) -> Dict[str, Any]:
    """Retrieve or initialize session state for given session_id."""
    if not session_id:
        session_id = "default_session"

    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {
            "session_id": session_id,
            "messages": [],  # List of {"role": "user"|"assistant", "content": str}
            "facts": {},     # {"location": str, "activity": str, "person": Optional[str]}
        }
    return _SESSIONS[session_id]


def update_session_facts(
    session_id: str,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    person: Optional[str] = None,
) -> None:
    """Store or update established facts in session memory."""
    mem = get_session_memory(session_id)
    if location:
        mem["facts"]["location"] = location
    if activity:
        mem["facts"]["activity"] = activity
    if person:
        mem["facts"]["person"] = person


def add_session_message(session_id: str, role: str, content: str) -> None:
    """Append a user or assistant message to session history."""
    mem = get_session_memory(session_id)
    mem["messages"].append({"role": role, "content": content})


def clear_session(session_id: str) -> None:
    """Reset session memory for a given session."""
    if session_id in _SESSIONS:
        _SESSIONS[session_id] = {
            "session_id": session_id,
            "messages": [],
            "facts": {},
        }


def get_all_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Retrieve chronological messages for session."""
    return get_session_memory(session_id).get("messages", [])
