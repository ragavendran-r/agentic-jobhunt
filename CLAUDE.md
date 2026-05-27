# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic JobHunt is a multi-agent AI system that autonomously finds, evaluates, and helps apply for Engineering Manager roles. It intentionally uses multiple agent frameworks in one project as a hands-on learning showcase.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env  # then fill in API keys

# Run full pipeline (CLI)
python -m app_agents.orchestrator

# Run API server
uvicorn api.main:app --reload

# Docker
docker-compose up

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_job_finder.py -v

# Run a specific test
pytest tests/test_job_finder.py::TestJobFinderAgent::test_run_job_finder_returns_dict -v

# Format code
black .

# Test individual agents standalone
python -m app_agents.job_finder
python -m app_agents.resume_matcher
python -m app_agents.outreach
python -m app_agents.tracker
python -m memory.vector_store
python -m tools.search_tool
```

## Architecture

The pipeline runs in strict sequential order, orchestrated by Google ADK:

```
User Preferences
      │
      ▼
Orchestrator (Google ADK, Gemini 2.5 Flash)
      │
      ├─► Job Finder (CrewAI, Gemini 2.5 Flash)
      │     Two CrewAI agents: Scraper (Tavily search) → Analyst (filter/rank)
      │     Returns: {jobs: [...], total_found, top_matches}
      │
      ├─► Resume Matcher (LangGraph, Gemini 2.5 Flash)
      │     LangGraph state machine: load_resume → score_job (loop) → compile_results
      │     RAG: resume chunked with ChromaDB + Gemini embeddings, in-memory per run
      │     Returns: {matched_jobs: [...], total_scored, recommended}
      │
      ├─► Outreach Drafter (OpenAI Agents SDK, GPT-4o-mini)
      │     Single agent with function tools for LinkedIn messages + cover letters
      │     Processes top 5 matched jobs only
      │     Returns: {outreach: [...], total_drafted}
      │
      └─► Tracker (LangChain + SQLAlchemy, Gemini 2.5 Flash)
            LangChain @tool decorators over SQLite via SQLAlchemy
            Statuses: To Apply → Applied → Phone Screen → Interview → Offer/Rejected/Withdrawn
            Returns: {logged, summary}
```

### Key architectural decisions

- **`config/settings.py`** is the single config source — all agents import `settings` from there. All three required API keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`) must be set; the app will fail to start without them.
- **LLM model constants** are in `settings.py`: `gemini_model` (direct), `gemini_model_crew` (prefixed `gemini/` for LiteLLM/CrewAI), `gemini_model_embedding`, `openai_model`.
- **The Resume Matcher creates a new in-memory ChromaDB** per invocation (inside `score_job`) rather than using the persistent store in `memory/vector_store.py`. The `memory/vector_store.py` module exists as a standalone utility but is not wired into the live pipeline.
- **The Orchestrator wraps sub-agents as plain Python function tools** (`find_jobs`, `match_resume`, `draft_outreach`, `track_applications`) and passes them to the Google ADK `Agent`. The ADK agent controls execution order via its instruction prompt.
- **`draft_outreach` is async** (uses OpenAI Agents SDK `Runner.run`); the synchronous wrappers in `orchestrator.py` use `asyncio.run()` for CLI entry, while `api/main.py` uses `await` directly.
- **SQLite DB** at `data/jobs.db` (configurable via `DB_PATH` env var). Schema auto-created on first run via `Base.metadata.create_all`.
- **LangSmith tracing** is enabled by default when `LANGCHAIN_API_KEY` is set and `LANGCHAIN_TRACING_V2=true`.

### Data layer

- `data/jobs.db` — SQLite application tracker
- `data/chroma/` — Persistent ChromaDB (used by `memory/vector_store.py`; not used by the live matcher)
- `resume.pdf` — Expected at repo root (path configurable via `RESUME_PATH`); falls back to a hardcoded sample string if missing

### API endpoints

- `GET /health` — liveness check
- `POST /search` — trigger full pipeline (async, uses `run_async`)
- `GET /applications` — all tracked applications
- `POST /applications/status` — update application status
- `GET /reminders` — follow-up reminders for active applications
