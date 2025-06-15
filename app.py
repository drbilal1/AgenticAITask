import streamlit as st
import os
from dotenv import load_dotenv

# Initialize first
load_dotenv()

# Verify API key before anything else
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing OpenAI API Key. Configure in Secrets (cloud) or .env (local)")
    st.stop()

# Only now import ResearchAssistant to prevent early failures
try:
    from research_assistant import ResearchAssistant
except ImportError as e:
    st.error(f"❌ Failed to import ResearchAssistant: {str(e)}")
    st.stop()

# Initialize session state
if "assistant" not in st.session_state:
    try:
        st.session_state.assistant = ResearchAssistant()
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"❌ Assistant initialization failed: {str(e)}")
        st.stop()

# Simple chat interface
st.title("🔍 Research Assistant")
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    try:
        response = st.session_state.assistant.query(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
