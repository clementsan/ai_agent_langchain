"""
Chatbot with search AI agent capabilities
"""

import os
import sys
import argparse

import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver


# Define AI Agent State
class AgentState(TypedDict):
    """Define AI Agent State"""
    messages: Annotated[list[AnyMessage], operator.add]


# Define AI Agent Class
class Agent:
    """Define AI Agent"""

    def __init__(self, model, tools, checkpointer, system=""):
        """Initialization with cyclic graph"""
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        """Check if action exists (conditional edge)"""
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        """Call llm"""
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        """Call Tavily tool"""
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t["name"] in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


def arg_parser():
    """Parse arguments"""

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Chatbot with search AI agent capabilities"
    )
    # Add arguments
    parser.add_argument(
        "--llm_type",
        type=str,
        help="LLM type: use of OpenAI (openai) vs Hugging Face (hf)",
        choices=["openai", "hf"],
        default="openai",
        required=False,
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        help="LLM model for OpenAI",
        default="gpt-3.5-turbo",
        required=False,
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        help="LLM model for HugginFace",
        default="meta-llama/Llama-3.2-3B-Instruct",
        required=False,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM temperature",
        default=0.7,
        required=False,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Max new tokens",
        default=1028,
        required=False,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Number of results for agent search",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    return parser


def main(args=None):
    """Main function"""

    args = arg_parser().parse_args(args)

    # Initialize environment variables
    _ = load_dotenv()
    huggingfacehub_api_token = os.environ.get("HUGGINGFACE_API_KEY")

    # System prompt
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    print("Initialize LLM model...")

    chat_model = None
    if args.llm_type == "openai":
        chat_model = ChatOpenAI(
            # reduce inference cost
            model=args.openai_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            timeout=None,
        )
    # WARNING: Issue with HF API and tool call
    # URL: https://discuss.huggingface.co/t/function-calling-not-working-with-inference-clients-on-seemingly-any-model/138581/13
    elif args.llm_type == "hf":
        chat_model = HuggingFaceEndpoint(
            repo_id=args.hf_model,
            task="text-generation",
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            do_sample=False,
            huggingfacehub_api_token=huggingfacehub_api_token,
        )
        chat_model = ChatHuggingFace(llm=llm)
    else:
        raise argparse.ArgumentTypeError("Invalid argument!")

    # Initialize Tavily Search tool
    print("Initialize Tavily Search Tool...")
    tool = TavilySearchResults(max_results=1)

    # Initialize AI Agent
    print("Initialize AI agent...")
    # Add persistence (in-memory database)
    with SqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(chat_model, [tool], checkpointer=memory, system=prompt)

        print("\nAI Chatbot")
        prompt = input(" Enter your prompt: ")
    
        messages = [HumanMessage(content=prompt)]

        # No streaming
        thread = {"configurable": {"thread_id": "1"}}
        result = abot.graph.invoke({"messages": messages}, thread)
        print("\n\nFull result: ", result)

        # With streaming
        # for event in abot.graph.stream({"messages": messages}, thread):
        #     for v in event.values():
        #         print(v['messages'])

        # final_answer = result["messages"][-1].content
        # print("\n\nFinal chatbot answer: ", final_answer)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
