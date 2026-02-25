# Agentic JobHunt — Architecture

## Overview

Agentic JobHunt is a multi-agent system that automates the job search workflow for Engineering Managers. Each agent is implemented using a different Agentic AI framework — deliberately chosen to demonstrate multi-framework fluency.

## Agent Design

### 1. Orchestrator — Google ADK

**Why ADK?** Google ADK natively supports hierarchical multi-agent architectures where a primary agent delegates to specialized sub-agents. The orchestrator understands intent from user preferences and routes tasks in the correct sequence.

**Flow:**

```
User Preferences → Orchestrator → find_jobs → match_resume → draft_outreach → track_applications → Summary
```

### 2. Job Finder — CrewAI

**Why CrewAI?** CrewAI excels at multi-agent collaboration with role-based agents working sequentially. The Job Finder uses two agents: a Scraper and an Analyst — mimicking a real recruiter workflow.

**Flow:**

```
Search Query → Scraper Agent (Tavily API) → Raw Listings → Analyst Agent → Filtered & Ranked Jobs
```

### 3. Resume Matcher — LangGraph

**Why LangGraph?** LangGraph provides graph-based state management with conditional edges — ideal for a multi-step scoring pipeline that loops over multiple job descriptions. The graph-based approach makes the pipeline inspectable and debuggable.

**Graph:**

```
load_resume → score_job → [loop until all jobs scored] → compile_results → END
```

**RAG Pipeline:**

- Resume is chunked and embedded into ChromaDB
- For each JD, the most relevant resume sections are retrieved via similarity search
- The LLM scores the match using retrieved context (not the full resume)

### 4. Outreach — OpenAI Agents SDK

**Why OpenAI Agents SDK?** The SDK's tool-calling primitives and built-in tracing make it ideal for a drafting agent that needs to call multiple tools (profile retrieval, message drafting, letter drafting) in a controlled, inspectable way.

### 5. Tracker — LangChain + MCP

**Why LangChain?** LangChain's tool-calling agent pattern with SQLAlchemy integration makes it straightforward to build a database-backed tracking agent. The MCP tool definitions make it easy to extend with external integrations (calendar, email, Slack).

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│                   User / FastAPI                     │
└────────────────────┬────────────────────────────────┘
                     │ SearchPreferences
                     ▼
┌─────────────────────────────────────────────────────┐
│            Orchestrator (Google ADK)                 │
│                                                     │
│  1. find_jobs()      → Job Finder (CrewAI)          │
│  2. match_resume()   → Resume Matcher (LangGraph)   │
│  3. draft_outreach() → Outreach (OpenAI SDK)        │
│  4. track_jobs()     → Tracker (LangChain)          │
└─────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
    Tavily API      ChromaDB        SQLite
    (job search)   (embeddings)    (tracker)
```

## Key Design Decisions

| Decision                | Rationale                                                                     |
| ----------------------- | ----------------------------------------------------------------------------- |
| One framework per agent | Demonstrates multi-framework fluency; each framework chosen for its strengths |
| ChromaDB for RAG        | Lightweight, local, no additional infra needed                                |
| SQLite for tracking     | Simple, file-based, no Docker dependency for dev                              |
| FastAPI wrapper         | Makes the system accessible via REST for future UI integration                |
| LangSmith tracing       | Single observability layer across all frameworks                              |

## Observability

All agents are traced through LangSmith. Set `LANGCHAIN_TRACING_V2=true` to enable. This gives:

- Full trace of each agent's tool calls
- Token usage per agent
- Latency breakdown
- Error visibility

## Extension Points

- Add a **UI layer** (Streamlit or React) on top of the FastAPI
- Add **email/calendar MCP tools** to the Tracker for automated follow-ups
- Add a **Negotiation Coach agent** that prepares you for salary discussions
- Add **interview prep agent** that generates likely questions per JD
