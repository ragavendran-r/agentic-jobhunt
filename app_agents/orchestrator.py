"""
Orchestrator Agent — Google ADK
Routes job search tasks to specialized sub-agents based on user preferences.
"""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from requests import session

from app_agents.job_finder import run_job_finder
from app_agents.resume_matcher import run_resume_matcher
from app_agents.outreach import run_outreach
from app_agents.tracker import run_tracker
from config.settings import settings


# ── Sub-agent tool wrappers ──────────────────────────────────────────────────


def find_jobs(role: str, location: str, tech_stack: list[str], min_salary: int) -> dict:
    """Find matching jobs from LinkedIn, Naukri, and Wellfound."""
    result = run_job_finder(
        role=role, location=location, tech_stack=tech_stack, min_salary=min_salary
    )
    return result["jobs"]


def match_resume(job_descriptions: list[dict], resume_path: str) -> dict:
    """Score each job description against the candidate resume and identify gaps."""
    return run_resume_matcher(job_descriptions=job_descriptions, resume_path=resume_path)


async def draft_outreach(matched_jobs: list[dict], candidate_name: str) -> dict:
    """Draft personalized cold outreach messages and cover letters for top matches."""
    return await run_outreach(matched_jobs=matched_jobs, candidate_name=candidate_name)


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
        You are a job search assistant. You MUST call ALL four tools in strict order. Do not stop early.

        STEP 1: Call find_jobs to discover relevant openings matching the criteria
        STEP 2: Call match_resume to score each job from Step 1 against the candidate's resume in the resume_path from the user prompt.
        STEP 3: Call draft_outreach with the matched jobs from Step 2 and the candidate_name from the user prompt.
        STEP 4: Call track_applications with the final jobs list.
        STEP 5: Write a final summary covering:
        - How many jobs were found
        - Top matches with title, company, url, match score
        - For EACH job, include the FULL outreach message text returned by draft_outreach
          (LinkedIn message AND cover letter — copy the full content, do not summarize it)
        - Next actions

        YOU MUST COMPLETE ALL 5 STEPS. Do not return a response before calling all four tools.
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
        """Sync version for CLI."""
        return asyncio.run(self.run_async(preferences))

    async def run_async(self, preferences: dict) -> dict:
        """
        Run the full job hunt pipeline. Async version for FastAPI

        Args:
            preferences: {
                "role": "Engineering Manager",
                "location": "Chennai, Remote",
                "tech_stack": ["Golang", "AWS", "Kubernetes"],
                "min_salary": 7000000,
                "resume_path": settings.resume_path,
                "candidate_name": "Ragavendran Ramalingam"
            }
        Returns:
            dict with matched_jobs, outreach_messages, tracker_summary
        """
        session = await self.session_service.create_session(
            app_name="agentic-jobhunt",
            user_id="user_001",
        )

        prompt = f"""
        Please run a complete job search with these preferences:
        - Role: {preferences.get('role', 'Engineering Manager')}
        - Location: {preferences.get('location', 'Chennai, Remote')}
        - Tech Stack: {preferences.get('tech_stack', [])}
        - Minimum Salary: ₹{preferences.get('min_salary', 0):,}
        - Resume: {preferences.get('resume_path', settings.resume_path)}
        - Candidate: {preferences.get('candidate_name', 'Candidate')}
        """

        result_text = ""
        async for event in self.runner.run_async(
            user_id="user_001",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            print(
                f"[{event.author}] final={event.is_final_response()} | has_content={bool(event.content)}"
            )  # temporary debug
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        result_text += part.text or ""
                    if hasattr(part, "function_call") and part.function_call:
                        print(f"  → TOOL CALL: {part.function_call.name}")
                    if hasattr(part, "function_response") and part.function_response:
                        print(f"  ← TOOL RESP: {part.function_response.name} success")
        # After loop: if orchestrator gave nothing, collect from session state instead
        if not result_text:
            session = await self.session_service.get_session(
                app_name="agentic-jobhunt", user_id="user_001", session_id=session.id
            )
            if session:
                result_text = str(session.state.get("summary") or session.state)

        # Join the list to form the final string
        full_response = "".join(result_text).strip()
        # If result_text is still empty, it means the agent called tools
        # but never wrote a concluding summary.
        return {"summary": full_response or "Task complete, but no summary was generated."}


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
