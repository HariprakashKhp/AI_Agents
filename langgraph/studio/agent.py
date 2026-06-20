import getpass
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def multiply(a: int, b: int) -> int:
    """
        Multiply a and b.

        Args:
            a: first int
            b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """
        Add a and b.

        Args:
            a: first int
            b: second int
    """
    return a + b

def divide(a: int, b: int) -> int:
    """
        Divide a and b.

        Args:
            a: first int
            b: second int
    """
    return a / b

tools = [multiply, add, divide]

llm_with_tools = llm.bind_tools(tools)

sys_message = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

class MessagesState(MessagesState):
    pass

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant" , tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()
