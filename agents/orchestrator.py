"""
Orchestrator Agent — Google ADK
Routes job search tasks to specialized sub-agents based on user preferences.
"""

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.job_finder import run_job_finder
from agents.resume_matcher import run_resume_matcher
from agents.outreach import run_outreach
from agents.tracker import run_tracker
from config.settings import settings


# ── Sub-agent tool wrappers ──────────────────────────────────────────────────


def find_jobs(role: str, location: str, tech_stack: list[str], min_salary: int) -> dict:
    """Find matching jobs from LinkedIn, Naukri, and Wellfound."""
    return run_job_finder(
        role=role, location=location, tech_stack=tech_stack, min_salary=min_salary
    )


def match_resume(job_descriptions: list[dict], resume_path: str) -> dict:
    """Score each job description against the candidate resume and identify gaps."""
    return run_resume_matcher(job_descriptions=job_descriptions, resume_path=resume_path)


def draft_outreach(matched_jobs: list[dict], candidate_name: str) -> dict:
    """Draft personalized cold outreach messages and cover letters for top matches."""
    return run_outreach(matched_jobs=matched_jobs, candidate_name=candidate_name)


def track_applications(jobs: list[dict]) -> dict:
    """Log applied roles and track application stages in the database."""
    return run_tracker(jobs=jobs)


# ── Orchestrator Agent ───────────────────────────────────────────────────────

orchestrator_agent = Agent(
    name="JobHuntOrchestrator",
    model=settings.gemini_model,
    description=(
        "An intelligent job search orchestrator that coordinates specialized "
        "sub-agents to find relevant Engineering Manager roles, match them "
        "against the candidate's resume, draft outreach messages, and track applications."
    ),
    instruction="""
        You are a job search assistant for an experienced Engineering Manager.
        
        When given job search preferences, follow this workflow in order:
        1. Call find_jobs to discover relevant openings matching the criteria
        2. Call match_resume to score each job against the candidate's resume
        3. Filter to jobs with a match score >= 60%
        4. Call draft_outreach to create personalized messages for top matches
        5. Call track_applications to log everything to the database
        6. Return a clear summary: jobs found, top matches, and next actions
        
        Always be concise and action-oriented in your final summary.
    """,
    tools=[find_jobs, match_resume, draft_outreach, track_applications],
)


# ── Runner ───────────────────────────────────────────────────────────────────


class JobHuntOrchestrator:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=orchestrator_agent,
            app_name="agentic-jobhunt",
            session_service=self.session_service,
        )

    def run(self, preferences: dict) -> dict:
        """
        Run the full job hunt pipeline.

        Args:
            preferences: {
                "role": "Engineering Manager",
                "location": "Chennai, Remote",
                "tech_stack": ["Golang", "AWS", "Kubernetes"],
                "min_salary": 7000000,
                "resume_path": "resume.pdf",
                "candidate_name": "Ragavendran Ramalingam"
            }
        Returns:
            dict with matched_jobs, outreach_messages, tracker_summary
        """
        session = self.session_service.create_session(
            app_name="agentic-jobhunt",
            user_id="user_001",
        )

        prompt = f"""
        Please run a complete job search with these preferences:
        - Role: {preferences.get('role', 'Engineering Manager')}
        - Location: {preferences.get('location', 'Chennai, Remote')}
        - Tech Stack: {preferences.get('tech_stack', [])}
        - Minimum Salary: ₹{preferences.get('min_salary', 6500000):,}
        - Resume: {preferences.get('resume_path', settings.resume_path)}
        - Candidate: {preferences.get('candidate_name', 'Candidate')}
        """

        result_text = ""
        for event in self.runner.run(
            user_id="user_001",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        result_text += part.text

        return {"summary": result_text}


# ── CLI Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    orchestrator = JobHuntOrchestrator()

    prefs = {
        "role": "Engineering Manager",
        "location": "Chennai, Remote",
        "tech_stack": ["Golang", "AWS", "Kubernetes", "DevSecOps"],
        "min_salary": 7000000,
        "resume_path": "resume.pdf",
        "candidate_name": "Ragavendran Ramalingam",
    }

    console.print(Panel("🤖 Agentic JobHunt Starting...", style="bold blue"))
    result = orchestrator.run(prefs)
    console.print(Panel(result["summary"], title="✅ Results", style="bold green"))
