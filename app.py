import streamlit as st
from research_assistant import ResearchAssistant
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="🔍 Research Assistant",
    page_icon="🔍",
    layout="wide"
)

def initialize_session_state():
    """Initialize all session state variables with proper error handling"""
    if "initialized" not in st.session_state:
        try:
            # Verify API key is available
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API key not configured. Please set it in Secrets (cloud) or .env (local)")
                st.stop()

            # Initialize assistant only once
            st.session_state.assistant = ResearchAssistant()
            st.session_state.chat_history = []
            st.session_state.tool_usage = {"Search": 0, "Python_REPL": 0}
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize assistant: {str(e)}")
            st.stop()

def display_chat():
    """Display chat messages with avatar icons"""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🔍" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

def display_sidebar():
    """Sidebar with stats and controls"""
    with st.sidebar:
        st.title("Controls")
        
        # API status
        st.markdown(f"**API Status:** {'✅ Connected' if os.getenv('OPENAI_API_KEY') else '❌ Disconnected'}")
        
        # Tool usage stats
        st.subheader("Tool Usage")
        for tool, count in st.session_state.tool_usage.items():
            st.progress(count % 100, text=f"{tool}: {count} uses")
        
        # Clear conversation button
        if st.button("♻️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()

def track_tools(response: str):
    """Track tool usage from agent response"""
    if "used Search" in response:
        st.session_state.tool_usage["Search"] += 1
    if "used Python_REPL" in response:
        st.session_state.tool_usage["Python_REPL"] += 1

def main():
    """Main app function"""
    initialize_session_state()
    
    # Header
    st.title("🔍 Research Assistant")
    st.caption("Powered by LangChain + OpenAI")
    
    # Layout
    display_sidebar()
    display_chat()
    
    # User input
    if prompt := st.chat_input("Ask me to research or calculate something..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Process with assistant
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.assistant.query(prompt)
                track_tools(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Sorry, I encountered an error. Please try again."
                })

if __name__ == "__main__":
    main()
