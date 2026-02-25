"""
Job Store — SQLite persistence layer
Stores and manages job application records.
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager

from config.settings import settings


# ── ORM Model ─────────────────────────────────────────────────────────────────

Base = declarative_base()


class JobApplication(Base):
    __tablename__ = "job_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(200), nullable=False)
    title = Column(String(200), nullable=False)
    url = Column(String(500))
    location = Column(String(200))
    salary = Column(String(100))
    source = Column(String(100))          # LinkedIn, Naukri, etc.
    match_score = Column(Float)           # 0-100
    matching_skills = Column(Text)        # JSON list
    missing_skills = Column(Text)         # JSON list
    status = Column(String(50), default="To Apply")
    applied_date = Column(DateTime)
    follow_up_date = Column(DateTime)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "company": self.company,
            "title": self.title,
            "url": self.url,
            "location": self.location,
            "salary": self.salary,
            "source": self.source,
            "match_score": self.match_score,
            "matching_skills": json.loads(self.matching_skills or "[]"),
            "missing_skills": json.loads(self.missing_skills or "[]"),
            "status": self.status,
            "applied_date": str(self.applied_date) if self.applied_date else None,
            "follow_up_date": str(self.follow_up_date) if self.follow_up_date else None,
            "notes": self.notes,
            "created_at": str(self.created_at),
        }


# ── Database Setup ────────────────────────────────────────────────────────────

def _get_engine():
    os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)), exist_ok=True)
    engine = create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session() -> Session:
    """Context manager for database sessions."""
    engine = _get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── CRUD Operations ───────────────────────────────────────────────────────────

def log_job_to_db(
    company: str,
    title: str,
    url: str = "",
    match_score: float = 0,
    salary: str = "",
    location: str = "",
    source: str = "",
    matching_skills: list = None,
    missing_skills: list = None,
    notes: str = "",
) -> dict:
    """
    Log a new job application to the database.

    Returns:
        The saved job application as a dict
    """
    with get_session() as session:
        # Check for duplicate
        existing = session.query(JobApplication).filter(
            JobApplication.company == company,
            JobApplication.title == title,
        ).first()

        if existing:
            return {"message": f"Already tracked: {title} at {company}", "id": existing.id}

        job = JobApplication(
            company=company,
            title=title,
            url=url,
            match_score=match_score,
            salary=salary,
            location=location,
            source=source,
            matching_skills=json.dumps(matching_skills or []),
            missing_skills=json.dumps(missing_skills or []),
            status="To Apply",
            notes=notes,
            created_at=datetime.utcnow(),
        )
        session.add(job)
        session.flush()
        return {"message": f"Logged: {title} at {company}", "id": job.id}


def update_job_status(company: str, new_status: str, notes: str = "") -> dict:
    """Update the status of a job application."""
    valid_statuses = ["To Apply", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"]
    if new_status not in valid_statuses:
        return {"error": f"Invalid status. Use one of: {valid_statuses}"}

    with get_session() as session:
        job = session.query(JobApplication).filter(
            JobApplication.company == company
        ).order_by(JobApplication.created_at.desc()).first()

        if not job:
            return {"error": f"No application found for {company}"}

        job.status = new_status
        job.updated_at = datetime.utcnow()

        if new_status == "Applied":
            job.applied_date = datetime.utcnow()

        if notes:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d")
            job.notes = f"{job.notes or ''}\n[{timestamp}] {notes}".strip()

        return {"message": f"Updated {company} → {new_status}"}


def get_all_jobs(status_filter: str = None) -> list[dict]:
    """Get all tracked jobs, optionally filtered by status."""
    with get_session() as session:
        query = session.query(JobApplication)
        if status_filter:
            query = query.filter(JobApplication.status == status_filter)
        jobs = query.order_by(JobApplication.match_score.desc()).all()
        return [j.to_dict() for j in jobs]


def get_jobs_summary() -> dict:
    """Get a summary of all applications grouped by status."""
    with get_session() as session:
        jobs = session.query(JobApplication).all()

        summary = {
            "total": len(jobs),
            "by_status": {},
            "top_matches": [],
            "avg_match_score": 0,
        }

        scores = []
        for job in jobs:
            status = job.status or "To Apply"
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            if job.match_score:
                scores.append(job.match_score)
            if job.match_score and job.match_score >= 75:
                summary["top_matches"].append({
                    "company": job.company,
                    "title": job.title,
                    "match_score": job.match_score,
                    "status": job.status,
                    "url": job.url,
                })

        summary["avg_match_score"] = round(sum(scores) / len(scores), 1) if scores else 0
        summary["top_matches"].sort(key=lambda x: x["match_score"], reverse=True)

        return summary


def get_pending_followups() -> list[dict]:
    """Get jobs that need follow-up (Applied/Interview for 5+ days)."""
    with get_session() as session:
        jobs = session.query(JobApplication).filter(
            JobApplication.status.in_(["Applied", "Phone Screen", "Interview"])
        ).all()

        reminders = []
        now = datetime.utcnow()

        for job in jobs:
            ref_date = job.applied_date or job.created_at
            if ref_date:
                days_elapsed = (now - ref_date).days
                if days_elapsed >= 5:
                    reminders.append({
                        "company": job.company,
                        "title": job.title,
                        "status": job.status,
                        "days_elapsed": days_elapsed,
                        "url": job.url,
                        "action": "Send follow-up message",
                    })

        return sorted(reminders, key=lambda x: x["days_elapsed"], reverse=True)


def delete_job(company: str) -> dict:
    """Delete a job application record."""
    with get_session() as session:
        job = session.query(JobApplication).filter(
            JobApplication.company == company
        ).first()
        if not job:
            return {"error": f"No record found for {company}"}
        session.delete(job)
        return {"message": f"Deleted record for {company}"}


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]💾 Testing Job Store...[/bold blue]")

    # Log test jobs
    log_job_to_db("Freshworks", "Engineering Manager", "https://freshworks.com", match_score=85, salary="70-90 LPA", location="Chennai")
    log_job_to_db("Chargebee", "Engineering Manager", "https://chargebee.com", match_score=78, salary="65-80 LPA", location="Chennai")
    log_job_to_db("Kissflow", "Engineering Manager", "https://kissflow.com", match_score=72, salary="60-75 LPA", location="Chennai")

    # Update a status
    update_job_status("Freshworks", "Applied", "Applied via LinkedIn referral")

    # Get summary
    summary = get_jobs_summary()
    console.print("\n[bold]Summary:[/bold]")
    print_json(data=summary)

    # Clean up test data
    for company in ["Freshworks", "Chargebee", "Kissflow"]:
        delete_job(company)
    console.print("\n[green]✅ Test data cleaned up[/green]")
