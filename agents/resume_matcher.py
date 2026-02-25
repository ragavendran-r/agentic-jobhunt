"""
Resume Matcher Agent — LangGraph
Scores job descriptions against candidate resume using RAG + graph-based workflow.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
import PyPDF2
import json
import os

from config.settings import settings


# ── State Definition ─────────────────────────────────────────────────────────

class MatcherState(TypedDict):
    resume_text: str
    job_descriptions: list[dict]
    resume_chunks: list[str]
    scored_jobs: list[dict]
    current_job_index: int
    final_results: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_resume_text(resume_path: str) -> str:
    """Extract text from a PDF resume."""
    if not os.path.exists(resume_path):
        return "Sample resume: Engineering Manager with 21 years experience in Golang, AWS, Kubernetes, DevSecOps."
    with open(resume_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() for page in reader.pages)


def build_vector_store(resume_text: str) -> Chroma:
    """Chunk resume and store in ChromaDB for RAG retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_path,
    )


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def load_resume(state: MatcherState) -> MatcherState:
    """Node 1: Load and parse the resume PDF."""
    resume_text = extract_resume_text(settings.resume_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    return {**state, "resume_text": resume_text, "resume_chunks": chunks}


def score_job(state: MatcherState) -> MatcherState:
    """Node 2: Score the current job description against the resume using RAG + LLM."""
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )

    idx = state["current_job_index"]
    job = state["job_descriptions"][idx]
    jd_text = job.get("description", "")

    # RAG: retrieve most relevant resume sections for this JD
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )
    vector_store = Chroma.from_texts(
        texts=state["resume_chunks"],
        embedding=embeddings,
    )
    relevant_chunks = vector_store.similarity_search(jd_text, k=4)
    resume_context = "\n".join([doc.page_content for doc in relevant_chunks])

    prompt = f"""
    You are a technical recruiter. Score how well this candidate's resume matches the job description.
    
    RESUME (relevant sections):
    {resume_context}
    
    JOB DESCRIPTION:
    {jd_text}
    
    Respond ONLY in this JSON format:
    {{
        "match_score": <0-100>,
        "matching_skills": ["skill1", "skill2"],
        "missing_skills": ["skill1", "skill2"],
        "strengths": "One sentence on candidate strengths for this role",
        "gaps": "One sentence on key gaps",
        "recommendation": "Apply" | "Consider" | "Skip"
    }}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        analysis = json.loads(response.content)
    except json.JSONDecodeError:
        analysis = {"match_score": 50, "recommendation": "Consider", "raw": response.content}

    scored_job = {**job, **analysis}
    scored_jobs = state.get("scored_jobs", []) + [scored_job]

    return {
        **state,
        "scored_jobs": scored_jobs,
        "current_job_index": idx + 1,
    }


def compile_results(state: MatcherState) -> MatcherState:
    """Node 3: Sort results by match score and filter to recommendations."""
    scored = state["scored_jobs"]
    filtered = [j for j in scored if j.get("match_score", 0) >= 60]
    sorted_jobs = sorted(filtered, key=lambda x: x.get("match_score", 0), reverse=True)
    return {**state, "final_results": sorted_jobs}


# ── Conditional Edge ──────────────────────────────────────────────────────────

def should_continue_scoring(state: MatcherState) -> str:
    """Continue scoring jobs or move to compile results."""
    if state["current_job_index"] < len(state["job_descriptions"]):
        return "score_job"
    return "compile_results"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_matcher_graph() -> StateGraph:
    graph = StateGraph(MatcherState)

    graph.add_node("load_resume", load_resume)
    graph.add_node("score_job", score_job)
    graph.add_node("compile_results", compile_results)

    graph.set_entry_point("load_resume")
    graph.add_edge("load_resume", "score_job")
    graph.add_conditional_edges("score_job", should_continue_scoring)
    graph.add_edge("compile_results", END)

    return graph.compile()


# ── Public Interface ──────────────────────────────────────────────────────────

def run_resume_matcher(job_descriptions: list[dict], resume_path: str) -> dict:
    """
    Score all job descriptions against the resume.

    Returns:
        {"matched_jobs": [...], "total_scored": int, "recommended": int}
    """
    graph = build_matcher_graph()

    initial_state: MatcherState = {
        "resume_text": "",
        "job_descriptions": job_descriptions,
        "resume_chunks": [],
        "scored_jobs": [],
        "current_job_index": 0,
        "final_results": [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "matched_jobs": final_state["final_results"],
        "total_scored": len(final_state["scored_jobs"]),
        "recommended": len(final_state["final_results"]),
    }


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]📄 Running Resume Matcher Agent...[/bold blue]")

    sample_jobs = [
        {
            "title": "Engineering Manager",
            "company": "Freshworks",
            "location": "Chennai",
            "description": "We are looking for an Engineering Manager with Golang, AWS, Kubernetes experience to lead a team of 10 engineers building SaaS products. DevSecOps knowledge preferred.",
            "url": "https://freshworks.com/careers",
            "salary": "70-90 LPA",
        },
        {
            "title": "Engineering Manager",
            "company": "Some Unrelated Company",
            "location": "Bangalore",
            "description": "Looking for EM with Ruby on Rails and React experience for a fintech startup.",
            "url": "https://example.com",
            "salary": "40-50 LPA",
        },
    ]

    result = run_resume_matcher(sample_jobs, settings.resume_path)
    console.print(f"\n[green]Matched {result['recommended']} of {result['total_scored']} jobs[/green]")
    print_json(data=result)
