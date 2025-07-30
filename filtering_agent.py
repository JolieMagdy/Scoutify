import os
import pandas as pd
import sqlite3
from typing import TypedDict, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import START, END, StateGraph
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool


import torch
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os
import openai

load_dotenv()  # loads variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- Models ----------
class JobRequirements(BaseModel):
    title: str
    required_skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]

class ResumeData(BaseModel):
    name: str
    skills: List[str]
    experience: List[str]
    education: List[str]

class AgentState(TypedDict):
    resume: ResumeData
    job: JobRequirements
    score: float
    recommendation: str

class Config(BaseModel):
    resume_text: str
    job_text: str

# ---------- Initialize LLM ----------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# ---------- Parser Chains ----------
job_parser = JsonOutputParser(pydantic_object=JobRequirements)
job_prompt = PromptTemplate(
    template=(
        "You are an assistant that extracts structured job info.\n\n"
        "{format_instructions}\n\nJob description:\n{job_text}"
    ),
    input_variables=["job_text"],
    partial_variables={"format_instructions": job_parser.get_format_instructions()}
)
job_chain = job_prompt | llm | job_parser

resume_parser = JsonOutputParser(pydantic_object=ResumeData)
resume_prompt = PromptTemplate(
    template=(
        "Extract structured information from this resume:\n\n"
        "{format_instructions}\n\nResume Text:\n{resume_text}"
    ),
    input_variables=["resume_text"],
    partial_variables={"format_instructions": resume_parser.get_format_instructions()}
)
resume_chain = resume_prompt | llm | resume_parser

def parse_job_description(job_text: str) -> JobRequirements:
    data = job_chain.invoke({"job_text": job_text})
    return JobRequirements.model_validate(data)

def parse_resume_text(resume_text: str) -> ResumeData:
    data = resume_chain.invoke({"resume_text": resume_text})
    return ResumeData.model_validate(data)

# ---------- Initialize OpenAI Embeddings ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------- Normalize Utility ----------
def normalize_skill(skill: str) -> str:
    return skill.lower().split('(')[0].strip()

# ---------- Matching Logic ----------
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def score_match(resume: ResumeData, job: JobRequirements) -> float:
    # Normalize and deduplicate
    res_skills = [normalize_skill(s) for s in resume.skills]
    job_skills = [normalize_skill(s) for s in job.required_skills]

    score_exact = len(set(res_skills).intersection(set(job_skills))) / max(1, len(job_skills))

    if not job_skills or not res_skills:
        return score_exact

    # Turn resume skills into docs
    resume_docs = [Document(page_content=s) for s in res_skills]

    # Create in-memory vector store
    store = InMemoryVectorStore.from_documents(resume_docs, embedding=embeddings)

    # For each job skill, find best matching resume skill
    matches = 0
    for job_skill in job_skills:
        result = store.similarity_search_with_score(job_skill, k=1)
        if result:
            _, score = result[0]
            # OpenAI cosine distance → 0 = identical, 1 = far → similarity = 1 - distance
            similarity = 1 - score
            if similarity >= 0.6:
                matches += 1

    score_sem = matches / len(job_skills)
    return max(score_exact, score_sem)

# ---------- Workflow Agents ----------
def resume_agent(state: AgentState, config: dict):
    state["resume"] = parse_resume_text(config["configurable"]["resume_text"])
    return state

def job_agent(state: AgentState, config: dict):
    state["job"] = parse_job_description(config["configurable"]["job_text"])
    return state

def match_agent(state: AgentState):
    state["score"] = score_match(state["resume"], state["job"])
    print(f"Score: {state['score']:.3f} · Job: {state['job'].title}")
    return state

def recruiter_agent(state: AgentState):
    state["recommendation"] = "Matched" if state["score"] >= 0.5 else "Reject"
    return state

# ---------- LangGraph Setup ----------
graph = StateGraph(AgentState)
graph.add_node("ResumeAgent", resume_agent)
graph.add_node("JobAgent", job_agent)
graph.add_node("MatchAgent", match_agent)
graph.add_node("RecruiterAgent", recruiter_agent)
graph.set_entry_point("ResumeAgent")
graph.add_edge("ResumeAgent", "JobAgent")
graph.add_edge("JobAgent", "MatchAgent")
graph.add_edge("MatchAgent", "RecruiterAgent")
graph.add_edge("RecruiterAgent", END)
workflow = graph.compile(debug=True)


# ❌ Remove this:
# @tool()

# ✅ Keep it plain:
def run_graph(resume_text: str, job_text: str):
    """Runs the resume-job matching graph and returns a score and recommendation."""
    return workflow.invoke(
        {"score": 0.0, "recommendation": "", "resume": None, "job": None},
        config={"configurable": {"resume_text": resume_text, "job_text": job_text}}
    )

if __name__ == "__main__":
    # ---------- Batch Processing ----------
    jobs_df = pd.read_csv("C:\\Users\\Julie\\OneDrive\\Desktop\\Scoutify-2\\job_title_des.csv")
    resumes_df = pd.read_csv("C:\\Users\\Julie\\OneDrive\\Desktop\\Scoutify-2\\sampled_resumes.csv")

    all_results = []
    for _, rrow in resumes_df.iterrows():
        resume_text = rrow["Resume"]
        for _, jrow in jobs_df.iterrows():
            job_text = jrow["Job Description"]
            state = run_graph(resume_text, job_text)
            all_results.append({
                "resume": resume_text[:50] + "...",
                "job_title": state["job"].title if state.get("job") else jrow.get("Job Title", ""),
                "score": state["score"],
                "recommendation": state["recommendation"]
            })

    # ---------- Save & Query ----------
    conn = sqlite3.connect("matches.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            resume TEXT, job_title TEXT, score REAL, recommendation TEXT
        )
    """)
    for rec in all_results:
        conn.execute("INSERT INTO matches VALUES (?,?,?,?)", 
                    (rec["resume"], rec["job_title"], rec["score"], rec["recommendation"]))
    conn.commit()

    df = pd.read_sql("SELECT * FROM matches", conn)
    print("Recommendation counts:\n", df['recommendation'].value_counts())
    best = df[df['recommendation'] == 'Matched'].sort_values(['job_title','score'], ascending=[True,False])
    print(best.groupby('job_title').head(1))

    conn.close()
