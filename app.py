import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure page before any other Streamlit commands
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

def initialize_assistant():
    """Safely initialize the ResearchAssistant"""
    try:
        from research_assistant import ResearchAssistant
        return ResearchAssistant()
    except ImportError as e:
        st.error(f"Failed to import ResearchAssistant: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Assistant initialization failed: {str(e)}")
        st.stop()

def main():
    # Initialize session state with error handling
    if "assistant" not in st.session_state:
        st.session_state.assistant = initialize_assistant()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.title("🔍 Research Assistant")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        try:
            response = st.session_state.assistant.query(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Sorry, I encountered an error. Please try again."
            })

if __name__ == "__main__":
    # Additional safety check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Configure in Secrets (cloud) or .env (local)")
        st.stop()
    
    main()
