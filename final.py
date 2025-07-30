from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
import os
# router_app.py
from filtering_agent import run_graph  # ✅ Import your agent
from email_agent import run_gmail_agent  # 👈 Import the actual email agent logic
from calendar_agent import run_calendar_agent  
from email_agent import graph as email_graph
from calendar_agent import graph as calendar_graph
from filtering_agent import graph as filtering_graph


# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Define app state
class AppState(TypedDict):
    user_input: str
    selected_agent: str
    final_output: str

# Agents (You must replace the logic with your real agents)
def filtering_agent(input_text: str) -> str:
    # You need to split the input_text into resume + job text.
    # This can be done using a delimiter, like "::", or through preprocessing.

    if "::" not in input_text:
        return "❌ Please provide resume and job description separated by '::'."

    resume_text, job_text = input_text.split("::", 1)

    try:
        result = run_graph(resume_text.strip(), job_text.strip())
        score = result["score"]
        title = result["job"].title if result.get("job") else "N/A"
        decision = result["recommendation"]
        return f"✅ Match Result:\n- Job Title: {title}\n- Score: {score:.2f}\n- Decision: {decision}"
    except Exception as e:
        return f"❌ Error running filtering agent: {str(e)}"
    
def email_agent(input_text: str) -> str:
    try:
        messages = run_gmail_agent(input_text)
        return "\n".join([msg.content for msg in messages if msg.content])
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"
    
def calendar_tool(input: str) -> str:
    """Handles calendar-related requests like creating or reading events."""
    return run_calendar_agent(input)

def transcribe_agent(input_text: str) -> str:
    return f"[Transcription Agent] Transcribed audio: {input_text}"

def approval_agent(input_text: str) -> str:
    return f"[Approval Agent] Approved/Rejected days off request: {input_text}"

def policy_chatbot_agent(input_text: str) -> str:
    return f"[Policy Bot] Answer: {input_text}"


def email_agent_node(state: AppState):
    user_input = state["user_input"]
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = email_graph.invoke(initial_state)
    # Extract output content (adjust as per your agent's output format)
    output = result["messages"][-1].content
    return {"final_output": output}

def calendar_agent_node(state: AppState):
    user_input = state["user_input"]
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = calendar_graph.invoke(initial_state)
    output = result["messages"][-1].content
    return {"final_output": output}

def filtering_agent_node(state: AppState):
    user_input = state["user_input"]
    # Use your existing filtering function or wrap filtering_graph similarly
    result = run_graph(user_input)  # Or wrap filtering graph if exists
    # Format output as before
    score = result["score"]
    title = result["job"].title if result.get("job") else "N/A"
    decision = result["recommendation"]
    output = f"✅ Match Result:\n- Job Title: {title}\n- Score: {score:.2f}\n- Decision: {decision}"
    return {"final_output": output}

# Routing LLM setup
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = JsonOutputParser(pydantic_schema=TypedDict("RouterOutput", {"agent": str}))

router_prompt = PromptTemplate.from_template("""
You are a smart HR assistant router. Decide which agent should handle the user's request. If you can't process the request reply with this is out of my scope.

Available agents:
- filtering_agent: To analyze/filter CVs and match them with job descriptions.
- email_agent: To send confirmation or rejection emails.
- calendar_agent: To schedule interviews and manage the HR's calendar.
- transcribe_agent: To transcribe interview recordings.
- approval_agent: To approve or reject employee days off.
- policy_chatbot_agent: To answer employee questions about company policies.

Return only JSON in this format:
{{"agent": "<chosen_agent>"}}

User input:
{user_input}
""")

router_chain = (
    router_prompt | router_llm | parser
)

# Define nodes
def route_agent(state: AppState):
    result = router_chain.invoke({"user_input": state["user_input"]})
    return {"selected_agent": result["agent"]}

def run_selected_agent(state: AppState):
    agent = state["selected_agent"]
    user_input = state["user_input"]
    
    if agent == "filtering_agent":
        output = filtering_agent(user_input)
    elif agent == "email_agent":
        output = email_agent(user_input)
    elif agent == "calendar_agent":
        output = calendar_agent(user_input)
    elif agent == "transcribe_agent":
        output = transcribe_agent(user_input)
    elif agent == "approval_agent":
        output = approval_agent(user_input)
    elif agent == "policy_chatbot_agent":
        output = policy_chatbot_agent(user_input)
    else:
        output = "Agent not recognized."

    return {"final_output": output}

# Build graph
builder = StateGraph(AppState)

builder.add_node("router", route_agent)
builder.add_node("agent_runner", run_selected_agent)

builder.set_entry_point("router")
builder.add_edge("router", "agent_runner")
builder.add_edge("agent_runner", END)

graph = builder.compile()

# Example run
if __name__ == "__main__":
    user_prompt = input("💬 Hello HR, How can I Help You Today: ")

    output = graph.invoke({
        "user_input": user_prompt
    })

    print("\n✅ Final Output:")
    print(output["final_output"])
