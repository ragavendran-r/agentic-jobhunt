"""
Outreach Agent — OpenAI Agents SDK
Drafts personalized cold outreach messages and cover letters for matched jobs.
"""

from agents import Agent, Runner, function_tool
from config.settings import settings
import json
import asyncio


# ── Tools ─────────────────────────────────────────────────────────────────────


def _get_candidate_profile() -> str:
    """Retrieve the candidate's profile and key highlights for personalization."""
    return json.dumps(
        {
            "name": "Ragavendran Ramalingam",
            "title": "Engineering Manager",
            "experience_years": 21,
            "recent_company": "CloudBees",
            "key_skills": ["Golang", "AWS", "Kubernetes", "DevSecOps", "ReactJS"],
            "certifications": [
                "CKA",
                "AWS Solutions Architect Professional",
                "Google Cloud Engineer",
                "OCI GenAI",
            ],
            "highlights": [
                "Led 10-member team delivering SaaS DevSecOps product at CloudBees",
                "Built Golang microservices from scratch to production in 9 months",
                "Managed 40-member cross-functional account for Life Sciences customer",
                "Agentic AI enthusiast with hands-on MCP and Google ADK experience",
            ],
            "target_salary": "70-85 LPA",
            "location": "Chennai, open to remote",
        }
    )


get_candidate_profile = function_tool(_get_candidate_profile)


def _draft_linkedin_message(company: str, role: str, hiring_manager: str, fit_reason: str) -> str:
    """Draft a short LinkedIn cold outreach message to a hiring manager."""
    return f"""
    Hi {hiring_manager or 'there'},

    I came across the {role} opportunity at {company} and it aligns strongly with my background — 
    21 years in full-stack engineering with recent experience as EM at CloudBees, leading a 
    Golang/AWS/K8s team delivering a SaaS DevSecOps product.

    {fit_reason}

    I'd love to connect briefly to explore if there's a fit. Would you be open to a quick call?

    Best,
    Ragavendran
    """


@function_tool
def draft_linkedin_message(company: str, role: str, hiring_manager: str, fit_reason: str) -> str:
    """Draft a short LinkedIn cold outreach message to a hiring manager."""
    return _draft_linkedin_message(company, role, hiring_manager, fit_reason)


def _draft_cover_letter(
    company: str, role: str, job_description: str, matching_skills: list[str]
) -> str:
    """Draft a tailored cover letter for a specific job application."""
    skills_str = ", ".join(matching_skills[:5]) if matching_skills else "Golang, AWS, Kubernetes"
    return f"""
    Dear Hiring Team at {company},

    I am writing to express my interest in the {role} position. With 21+ years of 
    software engineering experience and a hands-on approach to leadership, I bring both 
    the technical depth and people management expertise your team needs.

    Most recently at CloudBees, I led a 10-member engineering team delivering a SaaS 
    DevSecOps product — staying hands-on in Golang and AWS every sprint while owning 
    quarterly roadmap planning and team growth. Prior to that, I architected and delivered 
    a Golang microservices platform from zero to production in 9 months.

    My skills in {skills_str} align directly with your requirements. I am also actively 
    exploring Agentic AI — having completed Google's 5-Day AI Agents Intensive and built 
    a Golang-based OpenAI Assistant with MCP function routing.

    I would welcome the opportunity to discuss how I can contribute to {company}'s engineering 
    mission. Thank you for your consideration.

    Warm regards,
    Ragavendran Ramalingam
    +91 9921382626 | ragaven.r@gmail.com
    """


draft_cover_letter = function_tool(_draft_cover_letter)

# ── Agent ─────────────────────────────────────────────────────────────────────

outreach_agent = Agent(
    name="OutreachDrafter",
    model=settings.openai_model,
    instructions="""
        You are a professional job application assistant specializing in engineering leadership roles.
        
        For each matched job provided:
        1. Call get_candidate_profile to understand the candidate's background
        2. Call draft_linkedin_message to create a concise, personalized LinkedIn outreach
        3. Call draft_cover_letter to create a tailored cover letter
        4. Return both drafts clearly labeled for each company
        
        Keep messages professional but warm. Highlight specific matches between the 
        candidate's experience and the job requirements. Never sound generic or templated.
        Each message should feel personally written.
    """,
    tools=[get_candidate_profile, draft_linkedin_message, draft_cover_letter],
)


# ── Public Interface ──────────────────────────────────────────────────────────


async def run_outreach(matched_jobs: list[dict], candidate_name: str) -> dict:
    """
    Draft outreach messages for all matched jobs.

    Returns:
        {"outreach": [...], "total_drafted": int}
    """
    if not matched_jobs:
        return {"outreach": [], "total_drafted": 0}

    jobs_summary = json.dumps(
        [
            {
                "company": j.get("company"),
                "title": j.get("title"),
                "description": j.get("description", "")[:300],
                "matching_skills": j.get("matching_skills", []),
                "fit_reason": j.get("strengths", ""),
            }
            for j in matched_jobs[:5]  # Top 5 only
        ],
        indent=2,
    )

    prompt = f"""
    Please draft personalized outreach materials for these {len(matched_jobs[:5])} job matches:
    
    {jobs_summary}
    
    For each job, provide:
    1. A LinkedIn message (under 200 words)
    2. A cover letter (under 300 words)
    """

    result = await Runner.run(outreach_agent, prompt)
    output_text = result.final_output

    outreach_list = []
    for job in matched_jobs[:5]:
        print(f"Processing outreach for {job.get('company')} - {job.get('title')}")
        outreach_list.append(
            {
                "company": job.get("company"),
                "title": job.get("title"),
                "url": job.get("url"),
                "match_score": job.get("match_score"),
                "outreach_content": output_text,
            }
        )

    print(f"Drafted outreach for jobs:  {outreach_list}")
    return {
        "outreach": outreach_list,
        "total_drafted": len(outreach_list),
    }


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("[bold blue]✉️  Running Outreach Agent...[/bold blue]")

    sample_matches = [
        {
            "company": "Freshworks",
            "title": "Engineering Manager",
            "url": "https://freshworks.com/careers",
            "match_score": 85,
            "matching_skills": ["Golang", "AWS", "Kubernetes", "SaaS"],
            "strengths": "Strong match on Golang SaaS leadership at CloudBees",
            "description": "EM role for SaaS product team using Golang and AWS",
        }
    ]

    result = asyncio.run(run_outreach(sample_matches, "Ragavendran Ramalingam"))
    console.print(
        Panel(
            result["outreach"][0]["outreach_content"] if result["outreach"] else "No output",
            title="✅ Outreach Draft",
            style="green",
        )
    )
