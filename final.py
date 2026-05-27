

# router_app.py

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict
import os
import pandas as pd
from typing import TypedDict, Optional

#Import actual agent LangGraphs
from filtering_agent import run_graph  # A callable, not LangGraph (yet)
from email_agent import get_email_graph
from calendar_agent import get_calendar_graph

#from filtering_agent import graph as filtering_graph  # Optional if filtering becomes a LangGraph

# Set OpenAI API Key


# Define app state
class AppState(TypedDict):
    user_input: Optional[str]  # Optional, general user input
    resume_text: Optional[str]
    job_text: Optional[str]
    selected_agent: Optional[str]
    final_output: Optional[str]
    # Add any other shared state fields you want to preserve for context

# Individual non-graph agents (still callable functions)
def transcribe_agent(input_text: str) -> str:
    return f"[Transcription Agent] Transcribed audio: {input_text}"

def approval_agent(input_text: str) -> str:
    return f"[Approval Agent] Approved/Rejected days off request: {input_text}"

def policy_chatbot_agent(input_text: str) -> str:
    return f"[Policy Bot] Answer: {input_text}"

# Batch processing function for filtering agent
def run_batch_filtering() -> str:
    """Run filtering on all resumes and job descriptions from CSV files"""
    try:
        # Load the CSV files (update paths as needed)
        jobs_df = pd.read_csv("job_title_des.csv")
        resumes_df = pd.read_csv("sampled_resumes.csv")
        
        results = []
        processed_count = 0
        
        for _, rrow in resumes_df.iterrows():
            resume_text = rrow.get("Resume", "")
            if not isinstance(resume_text, str) or not resume_text.strip():
                continue
                
            for _, jrow in jobs_df.iterrows():
                job_text = jrow.get("Job Description", "")
                if not isinstance(job_text, str) or not job_text.strip():
                    continue

                try:
                    state = run_graph(resume_text, job_text)
                    results.append({
                        "resume_preview": resume_text[:100] + "...",
                        "job_title": state.get("job").title if state.get("job") else jrow.get("Job Title", ""),
                        "score": state.get("score", 0.0),
                        "recommendation": state.get("recommendation", "")
                    })
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing pair: {str(e)}")
                    continue
        
        # Filter for matches and get top results
        matches = [r for r in results if r["recommendation"] == "Matched"]
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        if not matches:
            return "❌ No matches found between the resumes and job descriptions."
        
        # Format results
        output = f"✅ Batch Processing Complete!\n"
        output += f"📊 Processed {processed_count} resume-job pairs\n"
        output += f"🎯 Found {len(matches)} matches\n\n"
        output += "🏆 Top Matches:\n"
        
        for i, match in enumerate(matches[:10], 1):  # Show top 10
            output += f"{i}. Job: {match['job_title']}\n"
            output += f"   Score: {match['score']:.3f}\n"
            output += f"   Resume: {match['resume_preview']}\n\n"
        
        return output
        
    except FileNotFoundError as e:
        return f"❌ CSV files not found. Please ensure 'job_title_des.csv' and 'sampled_resumes.csv' are in the current directory. Error: {str(e)}"
    except Exception as e:
        return f"❌ Error during batch processing: {str(e)}"

# Routing LLM setup
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Output parser
parser = JsonOutputParser(pydantic_schema=TypedDict("RouterOutput", {"agent": str}))

# Prompt to select agent
router_prompt = PromptTemplate.from_template("""
You are a smart HR assistant router. Decide which agent should handle the user's request. If you can't process the request reply with "this is out of my scope".

Available agents:
- filtering_agent: To analyze/filter CVs and match them with job descriptions (batch processing from CSV files).
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

# Router chain
router_chain = router_prompt | router_llm | parser

# Router logic node
def route_agent(state: AppState) -> AppState:
    # Check if specific resume and job texts are provided
    if state.get("resume_text") and state.get("job_text"):
        agent = "filtering_agent"
    else:
        if not state.get("user_input"):
            return state  # no input, keep as is

        result = router_chain.invoke({"user_input": state["user_input"]})
        agent = result.get("agent", "unknown")

    return {
        **state,
        "selected_agent": agent,
        "final_output": ""
    }

# Agent execution node
def run_selected_agent(state: AppState) -> AppState:
    agent = state["selected_agent"]
    user_input = state["user_input"]
    message_context = {"messages": [HumanMessage(content=user_input)]}

    try:
        if agent == "email_agent":
            result = get_email_graph().invoke(message_context)
            final = result["messages"][-1].content

        elif agent == "calendar_agent":
            result = get_calendar_graph().invoke(message_context)
            final = result["messages"][-1].content

        elif agent == "filtering_agent":
            # Check if we have specific resume and job texts
            if state.get("resume_text") and state.get("job_text"):
                # Run filtering graph with specific resume and job texts
                result_state = run_graph(state["resume_text"], state["job_text"])
                final = (
                    f"✅ Match Result:\n"
                    f"- Job Title: {result_state['job'].title if result_state.get('job') else 'N/A'}\n"
                    f"- Score: {result_state['score']:.2f}\n"
                    f"- Recommendation: {result_state['recommendation']}"
                )
            else:
                # Run batch processing from CSV files
                final = run_batch_filtering()

        elif agent == "transcribe_agent":
            final = transcribe_agent(user_input)
        elif agent == "approval_agent":
            final = approval_agent(user_input)
        elif agent == "policy_chatbot_agent":
            final = policy_chatbot_agent(user_input)
        else:
            final = "❌ Agent not recognized."
    except Exception as e:
        final = f"❌ Error running agent: {str(e)}"

    return {
        **state,
        "final_output": final
    }

# Build the router graph
builder = StateGraph(AppState)

builder.add_node("router", route_agent)
builder.add_node("agent_runner", run_selected_agent)

builder.set_entry_point("router")
builder.add_edge("router", "agent_runner")
builder.add_edge("agent_runner", END)

# Compile
graph = builder.compile()

# Example run
if __name__ == "__main__":
    user_prompt = input("💬 Hello HR, how can I help you today? ")

    output = graph.invoke({
        "user_input": user_prompt
    })

    print("\n✅ Final Output:")
    print(output["final_output"])
