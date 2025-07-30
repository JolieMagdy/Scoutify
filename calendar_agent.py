# gmail_agent.py
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
import os
import openai

load_dotenv()  # loads variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# calendar_agent.py
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

flow = InstalledAppFlow.from_client_secrets_file(
    'C:\\Users\\Julie\\OneDrive\\Desktop\\Scoutify_Final\\credentials.json',
    SCOPES
)
creds = flow.run_local_server(port=0, open_browser=False)
print("Login successful!")

from langchain_google_community import CalendarToolkit
toolkit = CalendarToolkit()

from langchain_openai import ChatOpenAI
tools = toolkit.get_tools()
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools=tools)

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

class GraphState(TypedDict):
    messages: list[AnyMessage]

def tool_calling_llm(state: GraphState):
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing calendar related events")
    new_message = llm.invoke([sys_msg] + state["messages"])

    return {"messages": [new_message]}

def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)
builder = StateGraph(GraphState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", should_continue)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

# Optional GUI display in notebook
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

messages = [HumanMessage(content="Can we name the calendar event on Saturday noon 27/07 uni event ?")]
initial_state = {"messages": messages}

new_state = graph.invoke(initial_state)
for message in new_state["messages"]:
    message.pretty_print()
# calendar_agent.py

def run_calendar_agent(user_message: str):
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content="You are a helpful assistant tasked with performing calendar-related events."),
        HumanMessage(content=user_message)
    ]
    state = {"messages": messages}
    result = graph.invoke(state)
    return result["messages"][-1].content  # or format as needed
