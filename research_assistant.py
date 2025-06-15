import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = [
            Tool(
                name="Search",
                func=DuckDuckGoSearchAPIWrapper().run,
                description="For current information"
            ),
            Tool(
                name="Python",
                func=PythonREPLTool().run,
                description="For calculations"
            )
        ]
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=ChatPromptTemplate.from_messages([
                ("system", "You're a helpful assistant..."),
                ("human", "{input}")
            ])
        )

    def query(self, input_text: str) -> str:
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        ).invoke({"input": input_text})["output"]
