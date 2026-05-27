# 🤖 Scoutify — AI-Powered HR Assistant

[![Hackathon](https://img.shields.io/badge/Generate%20AI-Berlin%202025-orange?style=flat-square)](https://tinyurl.com/4mux2a5t)
[![Award](https://img.shields.io/badge/🏆-Awarded%20at%20Workshop-gold?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-framework-green?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--mini-purple?style=flat-square)](https://openai.com)

> Built in **48 hours** at the *Generate AI: From Data to Innovation* international hackathon in Berlin, 2025.  
> Awarded at the workshop. Live demo → [tinyurl.com/4mux2a5t](https://tinyurl.com/4mux2a5t)

---

## What is Scoutify?

Scoutify is a **router-agent multi-agent system** that automates core HR workflows using LLMs and a RAG pipeline. Instead of one monolithic agent, a central router classifies every incoming HR request and dispatches it to the right specialist sub-agent — keeping each agent focused, fast, and accurate.

---

## Features

| Workflow | Description |
|---|---|
| 📄 **CV Screening** | Parses and evaluates resumes against job requirements automatically |
| 📅 **Interview Scheduling** | Coordinates and books interview slots without manual back-and-forth |
| 🌴 **Leave Tracking** | Handles leave requests, balances, and approvals in natural language |
| 💬 **Policy Q&A (RAG)** | Answers questions from unstructured policy documents with high accuracy |

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ Router Agent │  ← classifies intent
└──────┬───────┘
       │
  ┌────┴─────────────────────────┐
  │                              │
  ▼                              ▼
Sub-agents                   RAG Pipeline
(CV, schedule,          (policy docs → chunks →
 leave, etc.)            embeddings → retrieval
                          → GPT-3.5 answer)
```

### RAG Pipeline

```
Policy docs (PDF/text)
    → Chunking & Embedding
    → Vector Store
    → Semantic Retrieval
    → GPT-3.5-mini Response
```

---

## Tech Stack

- **Python** — core language
- **LangChain** — agent orchestration and RAG chain
- **OpenAI API** (`gpt-3.5-turbo`) — LLM backbone
- **RAG** — retrieval-augmented generation over unstructured policy documents

---

## Project Structure

```
Scoutify/
├── agents/          # router agent + specialist sub-agents
├── rag/             # RAG pipeline: chunking, embedding, retrieval
├── data/            # sample policy documents, CVs
├── tools/           # scheduling, leave, and screening tool definitions
├── main.py          # entry point
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/JolieMagdy/Scoutify
cd Scoutify
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY=your_key_here

python main.py
```

---

## Demo

Live demo: [tinyurl.com/4mux2a5t](https://tinyurl.com/4mux2a5t)

---

## Hackathon Context

Built at **Generate AI: From Data to Innovation** (Berlin, 2025) in an international team setting.  
Designed, implemented, and demoed a fully working system within the 48-hour window.  
**Awarded at the Berlin workshop.**

---

## Authors

[Jolie Magdy & Lydia Ayyad](https://github.com/JolieMagdy) · and team

---

*Built with Python, LangChain, and OpenAI API · Berlin 2025*
