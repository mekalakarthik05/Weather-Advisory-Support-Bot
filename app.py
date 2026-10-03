"""Streamlit Chat UI for Weather-Advisory Support Bot.

SOP-Grounded LangGraph Weather Safety Assistant.
"""

import os
import uuid
import streamlit as st
from backend.graph import run_advisory_agent, LOADED_SOPS
from backend.memory import (
    get_session_memory,
    add_session_message,
    clear_session,
)

# Page configuration
st.set_page_config(
    page_title="Weather Advisory Support Bot",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize unique session_id in Streamlit session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:8]}"

session_id = st.session_state["session_id"]

# Sidebar
with st.sidebar:
    st.title("🌤️ Weather Advisory Bot")
    st.markdown(
        "An **SOP-Grounded LangGraph Agent** that answers outdoor-activity "
        "safety questions using live weather data from Open-Meteo and strictly "
        "enforces written safety standard operating procedures."
    )

    st.divider()

    st.subheader("System Architecture")
    st.markdown("""
    - **Engine**: LangGraph StateGraph
    - **Weather**: Open-Meteo API (Live)
    - **Policy**: Grounded YAML SOPs
    - **Memory**: Multi-turn Context Store
    """)

    # Optional API key configuration for Streamlit Community Cloud reviewers
    st.subheader("LLM Key Configuration")
    provider = st.selectbox("LLM Provider", ["Auto / Mock / Local", "Google Gemini", "OpenAI"])
    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", help="Enter Google Gemini API key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["LLM_PROVIDER"] = "google"
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter OpenAI API key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["LLM_PROVIDER"] = "openai"

    st.divider()

    st.caption(f"**Session ID:** `{session_id}`")
    if st.button("🔄 Reset Conversation Memory", use_container_width=True):
        clear_session(session_id)
        st.session_state["chat_history"] = []
        st.rerun()

    with st.expander(f"📋 Loaded Policies ({len(LOADED_SOPS)} SOPs)"):
        for sop in LOADED_SOPS:
            st.markdown(f"**{sop.id}** — *{sop.category}* ({sop.severity})")
            st.caption(sop.description)

# Main UI Header
st.title("Weather-Advisory Support Bot")
st.markdown("##### SOP-Grounded Weather Safety Assistant")

# Status Bar
col1, col2, col3 = st.columns(3)
with col1:
    st.success("● Weather: Open-Meteo Live", icon="📡")
with col2:
    st.info(f"● Policy: {len(LOADED_SOPS)} Active SOPs", icon="📜")
with col3:
    st.warning("● Agent: LangGraph Workflow", icon="⚡")

# Example Questions Expandable
with st.expander("💡 Click to view example safety queries"):
    st.markdown("""
    - *Can I cycle to work in Hyderabad today?*
    - *Should I take my 7-year-old child to the park this evening in Hyderabad?*
    - *Is today suitable for an outdoor picnic in London?*
    - *Is it safe to ride my two-wheeler in high wind?*
    - *What about this evening instead?* *(tests multi-turn session memory)*
    """)

st.divider()

# Initialize chat history in state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display conversation messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask about weather and an outdoor activity (e.g. Can I cycle in Hyderabad today?)...")

if user_query:
    # Display user message
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    add_session_message(session_id, "user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Consulting live weather and verifying safety SOPs..."):
            try:
                result = run_advisory_agent(user_query, session_id=session_id)
                response_text = result.get("response", "No response generated.")
            except Exception as e:
                response_text = f"An unexpected system error occurred: {e}"

            st.markdown(response_text)

    # Store assistant response
    st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
    add_session_message(session_id, "assistant", response_text)
