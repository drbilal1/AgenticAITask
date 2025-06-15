import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

class ResearchAssistant:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the Research Assistant with tools and memory.
        
        Args:
            model_name: Name of the LLM to use (default: "gpt-3.5-turbo")
            temperature: Creativity parameter (0-1, default: 0.7)
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize embeddings (for potential future memory expansion)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        self.agent = self._create_agent()

    def _initialize_tools(self) -> List[Tool]:
        """Initialize and return the tools for the agent."""
        # Search tool (DuckDuckGo)
        search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(
            name="Search",
            func=search.run,
            description=(
                "Useful for when you need to answer questions about current events or find recent information. "
                "Input should be a search query. "
                "Prefer this tool for factual queries or when asked for recent/current information."
            )
        )
        
        # Python REPL tool
        python_repl = PythonREPLTool()
        python_repl_tool = Tool(
            name="Python_REPL",
            func=python_repl.run,
            description=(
                "Useful for when you need to execute Python code to solve math problems, "
                "analyze data, or perform calculations. "
                "Input should be valid Python code. "
                "Always check your code for errors before execution."
            )
        )
        
        return [search_tool, python_repl_tool]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with tools and memory."""
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly capable research and analysis assistant. 
            Your goal is to help users find information, analyze data, and solve problems.
            
            You have access to the following tools:
            - Search: For finding current information online
            - Python_REPL: For executing Python code to solve problems
            
            Always follow these rules:
            1. Think step by step before answering
            2. Use tools when needed, especially for factual or computational queries
            3. Be concise but thorough in your responses
            4. If a query requires multiple steps, break it down and explain your reasoning
            5. When using the Python REPL, always check your code for errors before execution
            6. When using Search, make sure to extract the most relevant information
            7. Always mention which tool you're using when you use one
            
            Current conversation:
            {chat_history}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5  # Prevent infinite loops
        )
        
        return agent_executor
    
    def add_to_memory(self, text: str, is_user: bool = True):
        """Add a message to the conversation memory."""
        if is_user:
            self.memory.chat_memory.add_user_message(text)
        else:
            self.memory.chat_memory.add_ai_message(text)
    
    def query(self, input_text: str) -> str:
        """
        Process a user query and return the agent's response.
        
        Args:
            input_text: The user's query
            
        Returns:
            The agent's response
        """
        try:
            response = self.agent.invoke({
                "input": input_text,
                "chat_history": self.memory.buffer_as_messages
            })
            return response["output"]
        except Exception as e:
            error_msg = (
                "I encountered an error while processing your request. "
                f"Please try again or rephrase your question. Error: {str(e)}"
            )
            return error_msg
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()

# For testing purposes
if __name__ == "__main__":
    # Initialize the assistant
    assistant = ResearchAssistant()
    
    print("Research Assistant initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Get the assistant's response
        response = assistant.query(user_input)
        print(f"\nAssistant: {response}")
