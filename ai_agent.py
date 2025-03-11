import os 

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults


# Initialize environment variables
_ = load_dotenv()
huggingfacehub_api_token = os.environ.get("HUGGINGFACE_API_KEY")


# Initialize Tavily Search
tool = TavilySearchResults(max_results=1) #increased number of results
print(type(tool))
print(tool.name)


# Define AI Agent class

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}



prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

print("Initialize LLM model...")
#model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
llm_model = "microsoft/Phi-3.5-mini-instruct"
llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            task="text-generation",
            temperature = 0.7,
            max_new_tokens = 1024,
            top_k = 1,
            do_sample=False,
            huggingfacehub_api_token=huggingfacehub_api_token,
        )
model = ChatHuggingFace(llm=llm)


print("Initialize AI agent...")
abot = Agent(model, [tool], system=prompt)



print("AI agent - inference...")
messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})

print(result['messages'][-1].content)


