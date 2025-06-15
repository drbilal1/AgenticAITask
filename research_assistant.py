import os
from typing import List
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class ResearchAssistant:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Verify API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.tools = self._setup_tools()
            self.agent = self._create_agent()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize assistant: {str(e)}")

    def _setup_tools(self) -> List[Tool]:
        """Initialize and configure tools"""
        search = DuckDuckGoSearchAPIWrapper()
        return [
            Tool(
                name="Search",
                func=search.run,
                description="For searching current information online"
            ),
            Tool(
                name="Python_REPL",
                func=PythonREPLTool().run,
                description="For executing Python code"
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Configure the agent with proper prompt and tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful research assistant..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def query(self, input_text: str) -> str:
        """Handle user queries with error protection"""
        try:
            response = self.agent.invoke({"input": input_text})
            return response["output"]
        except Exception as e:
            return f"Error processing your request: {str(e)}"
