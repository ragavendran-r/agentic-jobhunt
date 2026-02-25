"""
Resume Parser — PDF and DOCX resume text extractor
Parses candidate resume into structured sections for RAG and matching.
"""

import os
import re
import PyPDF2
from docx import Document as DocxDocument
from config.settings import settings


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_pdf(path: str) -> str:
    """Extract raw text from a PDF resume."""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_docx(path: str) -> str:
    """Extract raw text from a DOCX resume."""
    doc = DocxDocument(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def parse_resume(path: str = None) -> str:
    """
    Parse a resume file (PDF or DOCX) and return raw text.

    Args:
        path: Path to the resume file. Defaults to settings.resume_path.

    Returns:
        Raw text content of the resume.
    """
    path = path or settings.resume_path

    if not os.path.exists(path):
        # Return a default profile for development/testing
        return _default_profile()

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(path)
    elif ext in [".docx", ".doc"]:
        return parse_docx(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use PDF or DOCX.")


def parse_resume_structured(path: str = None) -> dict:
    """
    Parse resume and return structured sections.

    Returns:
        {
            "full_text": str,
            "summary": str,
            "experience": str,
            "skills": str,
            "education": str,
            "certifications": str,
        }
    """
    raw_text = parse_resume(path)
    return {
        "full_text": raw_text,
        "summary": _extract_section(raw_text, ["summary", "profile", "about"]),
        "experience": _extract_section(raw_text, ["experience", "employment", "work history"]),
        "skills": _extract_section(raw_text, ["skills", "technical skills", "technologies"]),
        "education": _extract_section(raw_text, ["education", "academic"]),
        "certifications": _extract_section(raw_text, ["certifications", "certificates", "credentials"]),
    }


def extract_skills(resume_text: str) -> list[str]:
    """
    Extract a list of technical skills mentioned in the resume.
    Simple keyword extraction — no ML needed.
    """
    known_skills = [
        "Golang", "Go", "Python", "Java", "NodeJS", "ReactJS", "TypeScript",
        "AWS", "GCP", "Azure", "Kubernetes", "K8S", "Docker", "Terraform",
        "Kafka", "Spark", "Storm", "Pulsar", "NATS", "gRPC", "REST",
        "MongoDB", "PostgreSQL", "MySQL", "Elasticsearch", "Redis", "Cassandra",
        "DevSecOps", "CI/CD", "GitLab", "GitHub", "Jenkins",
        "LangChain", "LangGraph", "CrewAI", "OpenAI", "Gemini", "ADK", "MCP",
        "Microservices", "SaaS", "Agile", "Scrum",
    ]

    found = []
    text_lower = resume_text.lower()
    for skill in known_skills:
        if skill.lower() in text_lower:
            found.append(skill)

    return list(set(found))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_section(text: str, keywords: list[str]) -> str:
    """Extract a section from resume text based on heading keywords."""
    lines = text.split("\n")
    section_lines = []
    in_section = False

    section_headings = [
        "summary", "experience", "education", "skills",
        "certifications", "projects", "achievements", "publications"
    ]

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line is the section we want
        if any(kw in line_lower for kw in keywords) and len(line.strip()) < 50:
            in_section = True
            continue

        # Check if we've hit a new section
        if in_section and any(h in line_lower for h in section_headings) and len(line.strip()) < 50:
            if not any(kw in line_lower for kw in keywords):
                break

        if in_section and line.strip():
            section_lines.append(line)

    return "\n".join(section_lines[:30])  # Cap at 30 lines


def _default_profile() -> str:
    """Fallback profile for testing when no resume file is present."""
    return """
    Ragavendran Ramalingam — Engineering Manager
    21+ years of experience in full-stack and data engineering.
    Currently: Engineering Manager at CloudBees — SaaS DevSecOps product.
    Leading 10-member team in Golang, ReactJS, AWS, Kubernetes.
    Previous: Built Golang microservices from scratch to production in 9 months.
    Managed 40-member cross-functional account for Life Sciences customer.
    Skills: Golang, Java, NodeJS, ReactJS, AWS, Kubernetes, Kafka, Spark, MongoDB, DevSecOps.
    Certifications: CKA, AWS Solutions Architect Professional, Google Cloud Engineer, OCI GenAI.
    Agentic AI: Google 5-Day AI Agents course (ADK), OpenAI MCP chatbot in Golang.
    """


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]📄 Testing Resume Parser...[/bold blue]")

    text = parse_resume()
    skills = extract_skills(text)

    console.print(f"[green]Resume length: {len(text)} chars[/green]")
    console.print(f"[green]Skills found: {skills}[/green]")

    structured = parse_resume_structured()
    console.print("\n[bold]Structured Sections:[/bold]")
    for section, content in structured.items():
        if section != "full_text":
            console.print(f"\n[yellow]{section.upper()}:[/yellow]\n{content[:200]}")
