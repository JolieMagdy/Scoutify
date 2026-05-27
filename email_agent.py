# gmail_agent.py
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
import os
import openai

load_dotenv()  # loads variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")


SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]

flow = InstalledAppFlow.from_client_secrets_file(
    'path',
    SCOPES
)
creds = flow.run_local_server(port=0, open_browser=False)
print("Login successful!")

from googleapiclient.discovery import build
service = build('gmail', 'v1', credentials=creds)

from langchain_google_community import GmailToolkit
toolkit = GmailToolkit(api_resource=service)

from langchain_google_community.gmail.send_message import GmailSendMessage
send_tool = GmailSendMessage(api_resource=toolkit.api_resource)

# response = send_tool.invoke({
# #     "to": "....",
# #     "subject": "akheran eshta8al",
# #     "message": "2adae w lataf"
# # })
# # print(response)
# })
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools=toolkit.get_tools())
tools = toolkit.get_tools()

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

class GraphState(TypedDict):
    messages: list[AnyMessage]

def tool_calling_llm(state: GraphState):
    new_message = llm.invoke(state["messages"])
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

# Optional: display diagram if run in Jupyter
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

# messages = [HumanMessage(content="Can we send an email confirming the meeting tommorow ?")]
# initial_state = {"messages": messages}

# new_state = graph.invoke(initial_state)
# for message in new_state["messages"]:
#     message.pretty_print()


from langchain_core.messages import SystemMessage

def run_gmail_agent(user_message: str):
    messages = [
        SystemMessage(content="You are an assistant that can use tools to send emails and read calendar events."),
        HumanMessage(content=user_message)
    ]
    new_state = graph.invoke({"messages": messages})
    return new_state["messages"]
