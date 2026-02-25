"""
Tracker Agent — LangChain + MCP
Logs job applications to SQLite and tracks stages via MCP tool integration.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json
import os

from config.settings import settings


# ── Database Setup ────────────────────────────────────────────────────────────

Base = declarative_base()


class JobApplication(Base):
    __tablename__ = "job_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(200))
    title = Column(String(200))
    url = Column(String(500))
    location = Column(String(200))
    salary = Column(String(100))
    match_score = Column(Integer)
    status = Column(String(50), default="To Apply")
    applied_date = Column(DateTime)
    follow_up_date = Column(DateTime)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db_session():
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{settings.db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


# ── LangChain MCP Tools ───────────────────────────────────────────────────────

@tool
def log_job_application(
    company: str,
    title: str,
    url: str,
    match_score: int,
    salary: str = "",
    location: str = "",
    notes: str = ""
) -> str:
    """Log a job application to the SQLite tracking database."""
    session = get_db_session()
    try:
        app = JobApplication(
            company=company,
            title=title,
            url=url,
            match_score=match_score,
            salary=salary,
            location=location,
            status="To Apply",
            applied_date=None,
            follow_up_date=None,
            notes=notes,
            created_at=datetime.utcnow(),
        )
        session.add(app)
        session.commit()
        return f"✅ Logged: {title} at {company} (Match: {match_score}%) — ID: {app.id}"
    except Exception as e:
        return f"❌ Error logging job: {str(e)}"
    finally:
        session.close()


@tool
def update_application_status(company: str, new_status: str, notes: str = "") -> str:
    """
    Update the status of a job application.
    Valid statuses: To Apply, Applied, Phone Screen, Interview, Offer, Rejected, Withdrawn
    """
    valid_statuses = ["To Apply", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"]
    if new_status not in valid_statuses:
        return f"❌ Invalid status. Choose from: {valid_statuses}"

    session = get_db_session()
    try:
        app = session.query(JobApplication).filter(
            JobApplication.company == company
        ).order_by(JobApplication.created_at.desc()).first()

        if not app:
            return f"❌ No application found for {company}"

        app.status = new_status
        if new_status == "Applied":
            app.applied_date = datetime.utcnow()
        if notes:
            app.notes = (app.notes or "") + f"\n[{datetime.utcnow().strftime('%Y-%m-%d')}] {notes}"
        session.commit()
        return f"✅ Updated {company} → {new_status}"
    finally:
        session.close()


@tool
def get_application_summary() -> str:
    """Get a summary of all tracked job applications and their statuses."""
    session = get_db_session()
    try:
        apps = session.query(JobApplication).order_by(JobApplication.match_score.desc()).all()
        if not apps:
            return "No applications tracked yet."

        summary = {"total": len(apps), "by_status": {}, "top_matches": []}

        for app in apps:
            summary["by_status"][app.status] = summary["by_status"].get(app.status, 0) + 1
            if app.match_score and app.match_score >= 75:
                summary["top_matches"].append({
                    "company": app.company,
                    "title": app.title,
                    "match_score": app.match_score,
                    "status": app.status,
                })

        return json.dumps(summary, indent=2)
    finally:
        session.close()


@tool
def get_followup_reminders() -> str:
    """Get list of applications that need follow-up action."""
    session = get_db_session()
    try:
        apps = session.query(JobApplication).filter(
            JobApplication.status.in_(["Applied", "Phone Screen", "Interview"])
        ).all()

        if not apps:
            return "No pending follow-ups."

        reminders = []
        for app in apps:
            applied_days = (datetime.utcnow() - app.applied_date).days if app.applied_date else 0
            if applied_days >= 5:
                reminders.append({
                    "company": app.company,
                    "status": app.status,
                    "days_since_action": applied_days,
                    "action": "Send follow-up message",
                })

        return json.dumps(reminders, indent=2) if reminders else "No follow-ups needed yet."
    finally:
        session.close()


# ── LangChain Agent ───────────────────────────────────────────────────────────

def build_tracker_agent() -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )

    tools = [log_job_application, update_application_status, get_application_summary, get_followup_reminders]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a job application tracker assistant.
            Your job is to log new applications, update statuses, and surface follow-up reminders.
            Always confirm what you've logged and provide a brief summary at the end.
            Be concise and organized.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ── Public Interface ──────────────────────────────────────────────────────────

def run_tracker(jobs: list[dict]) -> dict:
    """
    Log all matched jobs to the tracker database.

    Returns:
        {"logged": int, "summary": str}
    """
    if not jobs:
        return {"logged": 0, "summary": "No jobs to track."}

    executor = build_tracker_agent()

    jobs_text = "\n".join([
        f"- {j.get('title')} at {j.get('company')} | Score: {j.get('match_score', 'N/A')}% | URL: {j.get('url')} | Salary: {j.get('salary', 'N/A')}"
        for j in jobs[:10]
    ])

    result = executor.invoke({
        "input": f"Please log these job applications and then give me a summary:\n{jobs_text}"
    })

    return {
        "logged": len(jobs[:10]),
        "summary": result.get("output", ""),
    }


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("[bold blue]📊 Running Tracker Agent...[/bold blue]")

    sample_jobs = [
        {"title": "Engineering Manager", "company": "Freshworks", "url": "https://freshworks.com", "match_score": 85, "salary": "70-90 LPA"},
        {"title": "Engineering Manager", "company": "Chargebee", "url": "https://chargebee.com", "match_score": 78, "salary": "65-80 LPA"},
    ]

    result = run_tracker(sample_jobs)
    console.print(Panel(result["summary"], title="✅ Tracker Summary", style="green"))
