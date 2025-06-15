import streamlit as st
from research_assistant import ResearchAssistant
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all necessary session state variables."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tool_usage" not in st.session_state:
        st.session_state.tool_usage = {"Search": 0, "Python_REPL": 0}

def display_chat_history():
    """Display the chat history in the Streamlit app."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_sidebar():
    """Display the sidebar with additional information."""
    with st.sidebar:
        st.title("About")
        st.markdown("""
        **Research Assistant** is an AI agent that can:
        - Search for current information online
        - Perform calculations using Python
        - Remember our conversation
        """)
        
        st.divider()
        
        st.subheader("Tool Usage Statistics")
        for tool, count in st.session_state.tool_usage.items():
            st.write(f"{tool}: {count} uses")
        
        st.divider()
        
        st.subheader("System Information")
        st.write(f"Model: {st.session_state.assistant.llm.model_name}")
        st.write("Memory: Enabled")
        
        st.divider()
        
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.tool_usage = {"Search": 0, "Python_REPL": 0}
            st.rerun()

def track_tool_usage(response: str):
    """Track which tools were used in the response."""
    if "used Search" in response:
        st.session_state.tool_usage["Search"] += 1
    if "used Python_REPL" in response:
        st.session_state.tool_usage["Python_REPL"] += 1

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    
    st.title("🔍 Research Assistant")
    st.caption("A task-oriented AI agent with search, calculation, and memory capabilities")
    
    display_sidebar()
    display_chat_history()
    
    # User input
    if prompt := st.chat_input("What would you like to research or analyze?"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.spinner("Researching..."):
            try:
                response = st.session_state.assistant.query(prompt)
                track_tool_usage(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()

if __name__ == "__main__":
    main()
