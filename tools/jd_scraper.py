"""
JD Scraper — Job Description scraper
Fetches full job descriptions from job portal URLs.
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ── Headers to mimic a browser ────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_job_description(url: str, timeout: int = 10) -> dict:
    """
    Scrape a job description from a given URL.

    Args:
        url: Full URL to the job listing
        timeout: Request timeout in seconds

    Returns:
        {
            "url": str,
            "title": str,
            "company": str,
            "location": str,
            "description": str,
            "source": str,
            "success": bool,
            "error": str | None
        }
    """
    result = {
        "url": url,
        "title": "",
        "company": "",
        "location": "",
        "description": "",
        "source": _get_source(url),
        "success": False,
        "error": None,
    }

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        source = result["source"]

        if source == "LinkedIn":
            result.update(_parse_linkedin(soup))
        elif source == "Naukri":
            result.update(_parse_naukri(soup))
        elif source == "Wellfound":
            result.update(_parse_wellfound(soup))
        else:
            result.update(_parse_generic(soup))

        result["success"] = True

    except requests.exceptions.Timeout:
        result["error"] = f"Timeout after {timeout}s"
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP {e.response.status_code}"
    except Exception as e:
        result["error"] = str(e)

    return result


def scrape_multiple(urls: list[str]) -> list[dict]:
    """Scrape multiple job descriptions and return list of results."""
    results = []
    for url in urls:
        result = scrape_job_description(url)
        results.append(result)
        if result["success"]:
            print(f"✅ Scraped: {result['title']} at {result['company']}")
        else:
            print(f"❌ Failed: {url} — {result['error']}")
    return results


# ── Site-specific parsers ─────────────────────────────────────────────────────

def _parse_linkedin(soup: BeautifulSoup) -> dict:
    return {
        "title": _get_text(soup, "h1"),
        "company": _get_text(soup, ".topcard__org-name-link") or _get_text(soup, ".company-name"),
        "location": _get_text(soup, ".topcard__flavor--bullet"),
        "description": _get_text(soup, ".show-more-less-html__markup") or _get_text(soup, ".description__text"),
    }


def _parse_naukri(soup: BeautifulSoup) -> dict:
    return {
        "title": _get_text(soup, "h1.jd-header-title") or _get_text(soup, "h1"),
        "company": _get_text(soup, "a.jd-header-comp-name") or _get_text(soup, ".comp-name"),
        "location": _get_text(soup, ".loc") or _get_text(soup, ".location"),
        "description": _get_text(soup, ".job-desc") or _get_text(soup, "#job-desc"),
    }


def _parse_wellfound(soup: BeautifulSoup) -> dict:
    return {
        "title": _get_text(soup, "h1"),
        "company": _get_text(soup, ".company-name") or _get_text(soup, "h2"),
        "location": _get_text(soup, ".location"),
        "description": _get_text(soup, ".job-description") or _get_text(soup, ".description"),
    }


def _parse_generic(soup: BeautifulSoup) -> dict:
    """Generic parser for unknown job portals."""
    # Remove nav, footer, scripts
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    return {
        "title": _get_text(soup, "h1"),
        "company": "",
        "location": "",
        "description": _clean_text(soup.get_text(separator="\n"))[:2000],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_text(soup: BeautifulSoup, selector: str) -> str:
    """Safely get text from a CSS selector."""
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else ""


def _clean_text(text: str) -> str:
    """Remove excess whitespace from scraped text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _get_source(url: str) -> str:
    """Identify source platform from URL."""
    domain = urlparse(url).netloc.lower()
    if "linkedin" in domain:
        return "LinkedIn"
    elif "naukri" in domain:
        return "Naukri"
    elif "wellfound" in domain:
        return "Wellfound"
    elif "indeed" in domain:
        return "Indeed"
    elif "glassdoor" in domain:
        return "Glassdoor"
    return "Other"


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("[bold blue]🌐 Testing JD Scraper...[/bold blue]")

    # Test with a sample public URL
    test_url = "https://wellfound.com/jobs"
    result = scrape_job_description(test_url)

    if result["success"]:
        console.print(Panel(
            f"Title: {result['title']}\nCompany: {result['company']}\nDesc: {result['description'][:200]}",
            title="✅ Scraped Job",
            style="green"
        ))
    else:
        console.print(f"[red]Failed: {result['error']}[/red]")
