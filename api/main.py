"""
FastAPI entrypoint for Agentic JobHunt
Exposes REST endpoints to trigger the multi-agent pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app_agents.orchestrator import JobHuntOrchestrator
from app_agents.tracker import (
    update_application_status,
    get_application_summary,
    get_followup_reminders,
)

app = FastAPI(
    title="Agentic JobHunt",
    description="Multi-agent job search assistant for Engineering Managers",
    version="1.0.0",
)

orchestrator = JobHuntOrchestrator()


# ── Request Models ────────────────────────────────────────────────────────────


class SearchPreferences(BaseModel):
    role: str = "Engineering Manager"
    location: str = "Chennai, Remote"
    tech_stack: list[str] = ["Golang", "AWS", "Kubernetes"]
    min_salary: int = 7000000
    resume_path: str = "resume.pdf"
    candidate_name: str = "Ragavendran Ramalingam"


class StatusUpdate(BaseModel):
    company: str
    new_status: str
    notes: Optional[str] = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "service": "Agentic JobHunt"}


@app.post("/search")
async def run_job_search(prefs: SearchPreferences):
    """Trigger the full multi-agent job search pipeline."""
    try:
        result = await orchestrator.run_async(prefs.model_dump())
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/applications")
def get_applications():
    """Get all tracked job applications and summary."""
    return {"summary": get_application_summary.invoke({})}


@app.post("/applications/status")
def update_status(update: StatusUpdate):
    """Update the status of a tracked application."""
    result = update_application_status.invoke(
        {
            "company": update.company,
            "new_status": update.new_status,
            "notes": update.notes,
        }
    )
    return {"result": result}


@app.get("/reminders")
def get_reminders():
    """Get follow-up reminders for active applications."""
    return {"reminders": get_followup_reminders.invoke({})}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
