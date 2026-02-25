"""
Search Tool — Tavily API wrapper
Used by Job Finder agent to search for job listings.
"""

from tavily import TavilyClient
from config.settings import settings
import json


def search_jobs(query: str, location: str, max_results: int = 10) -> list[dict]:
    """
    Search for job listings using Tavily.

    Args:
        query: Job title or keywords e.g. "Engineering Manager Golang"
        location: Location string e.g. "Chennai" or "Remote India"
        max_results: Max number of results to return

    Returns:
        List of job result dicts with url, title, content, score
    """
    client = TavilyClient(api_key=settings.tavily_api_key)

    search_query = f"{query} {location} job opening hiring 2025"

    response = client.search(
        query=search_query,
        search_depth="advanced",
        max_results=max_results,
        include_domains=[
            "linkedin.com",
            "naukri.com",
            "wellfound.com",
            "indeed.com",
            "glassdoor.com",
            "instahyre.com",
            "cutshort.io",
        ],
        include_answer=False,
    )

    results = response.get("results", [])

    # Normalize results
    normalized = []
    for r in results:
        normalized.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("content", "")[:500],
            "source": _extract_source(r.get("url", "")),
            "score": r.get("score", 0),
        })

    # Sort by relevance score
    return sorted(normalized, key=lambda x: x["score"], reverse=True)


def search_hiring_manager(company: str, role: str = "Engineering Manager") -> list[dict]:
    """
    Search for hiring managers or engineering leaders at a company on LinkedIn.

    Args:
        company: Company name
        role: Role to look for e.g. "VP Engineering", "Director Engineering"

    Returns:
        List of potential contacts
    """
    client = TavilyClient(api_key=settings.tavily_api_key)

    response = client.search(
        query=f"{role} at {company} LinkedIn India",
        search_depth="basic",
        max_results=5,
        include_domains=["linkedin.com"],
    )

    return response.get("results", [])


def _extract_source(url: str) -> str:
    """Extract the source platform name from a URL."""
    if "linkedin.com" in url:
        return "LinkedIn"
    elif "naukri.com" in url:
        return "Naukri"
    elif "wellfound.com" in url:
        return "Wellfound"
    elif "indeed.com" in url:
        return "Indeed"
    elif "glassdoor.com" in url:
        return "Glassdoor"
    elif "instahyre.com" in url:
        return "Instahyre"
    elif "cutshort.io" in url:
        return "Cutshort"
    return "Other"


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]🔍 Testing Search Tool...[/bold blue]")

    results = search_jobs(
        query="Engineering Manager Golang DevSecOps",
        location="Chennai Remote India",
        max_results=5,
    )

    console.print(f"[green]Found {len(results)} results[/green]")
    print_json(data=results)
