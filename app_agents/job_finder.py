"""
Job Finder Agent — CrewAI
Scrapes LinkedIn, Naukri, and Wellfound for matching EM roles.
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from tavily import TavilyClient
from pydantic import BaseModel, Field
from config.settings import settings
import json


# ── Tavily Search Tool ───────────────────────────────────────────────────────


class JobSearchInput(BaseModel):
    query: str = Field(description="Job search query string")
    location: str = Field(description="Location to search jobs in")


class JobSearchTool(BaseTool):
    name: str = "Job Search Tool"
    description: str = "Search for job listings on the web using Tavily."
    args_schema: type[BaseModel] = JobSearchInput

    def _run(self, query: str, location: str) -> str:
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(
            query=f"{query} {location} job opening 2025",
            search_depth="advanced",
            max_results=10,
            include_domains=["linkedin.com", "naukri.com", "wellfound.com", "indeed.com"],
        )
        return json.dumps(results.get("results", []), indent=2)


# ── CrewAI Agents ────────────────────────────────────────────────────────────


def build_crew(role: str, location: str, tech_stack: list[str], min_salary: int) -> Crew:
    search_tool = JobSearchTool()

    # Agent 1: Scraper — finds raw job listings
    scraper = Agent(
        role="Job Listing Scraper",
        goal=f"Find all {role} job openings in {location} on LinkedIn, Naukri and Wellfound",
        backstory=(
            "You are an expert job market researcher with deep knowledge of "
            "tech hiring in India. You know exactly how to find senior engineering "
            "leadership roles on job portals."
        ),
        tools=[search_tool],
        llm=f"{settings.gemini_model_crew}",
        verbose=True,
    )

    # Agent 2: Analyst — filters and scores listings
    analyst = Agent(
        role="Job Fit Analyst",
        goal=(
            f"Filter job listings to only those matching: "
            f"role={role}, tech={tech_stack}, min_salary=₹{min_salary:,}, location={location}"
        ),
        backstory=(
            "You are a technical recruiter specializing in engineering leadership roles. "
            "You can quickly assess whether a job description matches a candidate's "
            "technical stack and seniority level."
        ),
        llm=f"{settings.gemini_model_crew}",
        verbose=True,
    )

    # Task 1: Scrape
    scrape_task = Task(
        description=(
            f"Search for '{role}' job openings in '{location}'. "
            f"Search on LinkedIn, Naukri, and Wellfound. "
            f"Collect job listings with these fields: title, company, location, URL, salary (put 'Not Available' if missing), description (use full description available). "
            f"Target software engineering leadership roles only — ignore hardware, civil, mechanical engineering. "
            f"Collect at least 10 listings."
        ),
        expected_output=(
            "A JSON list of job listings with fields: "
            "title, company, location, url, salary, description, source"
        ),
        agent=scraper,
    )

    # Task 2: Filter
    filter_task = Task(
        description=(
            f"From the scraped job listings, filter and rank those that match:\n"
            f"- Role: Must be Engineering Manager, Engineering Lead, Development Manager, or similar senior engineering leadership role\n"
            f"- Location: {location} — include Remote roles too\n"
            f"- Exclude clearly irrelevant roles: civil engineering, hardware, HVAC, campus internships, analyst roles\n"
            f"- DO NOT filter on tech stack — descriptions are too short to contain this info\n"
            f"- DO NOT filter on salary — almost never mentioned in listings\n"
            f"- Remove duplicate companies (keep best matching title per company)\n"
            f"Rank by seniority and relevance to software engineering leadership."
        ),
        expected_output=(
            "A ranked JSON list (max 10) of filtered jobs with fields: "
            "title, company, location, url, salary, description, source, fit_reason"
        ),
        agent=analyst,
        context=[scrape_task],
    )

    return Crew(
        agents=[scraper, analyst],
        tasks=[scrape_task, filter_task],
        process=Process.sequential,
        verbose=True,
    )


# ── Public Interface ─────────────────────────────────────────────────────────


def run_job_finder(role: str, location: str, tech_stack: list[str], min_salary: int) -> dict:
    """
    Find and filter job listings using CrewAI multi-agent workflow.

    Returns:
        {"jobs": [...], "total_found": int, "top_matches": int}
    """
    crew = build_crew(role, location, tech_stack, min_salary)
    result = crew.kickoff()

    try:
        jobs = json.loads(str(result))
    except json.JSONDecodeError:
        jobs = [{"raw_output": str(result)}]

    return {
        "jobs": jobs,
        "total_found": len(jobs),
        "top_matches": min(len(jobs), 10),
    }


# ── Standalone Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]🔍 Running Job Finder Agent...[/bold blue]")

    result = run_job_finder(
        role="Engineering Manager",
        location="Chennai, Remote India",
        tech_stack=["Golang", "AWS", "Kubernetes", "DevSecOps"],
        min_salary=7000000,
    )

    console.print(f"\n[green]Found {result['total_found']} jobs[/green]")
    print_json(data=result)
