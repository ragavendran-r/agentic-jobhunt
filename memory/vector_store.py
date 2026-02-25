"""
Vector Store — ChromaDB wrapper
Stores and retrieves resume chunks and JD embeddings for RAG-based matching.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document

from config.settings import settings


# ── Embeddings ────────────────────────────────────────────────────────────────


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model instance."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.google_api_key,  # type: ignore
    )


# ── Resume Vector Store ───────────────────────────────────────────────────────


def build_resume_store(resume_text: str, collection_name: str = "resume") -> Chroma:
    """
    Chunk and embed resume text into ChromaDB.

    Args:
        resume_text: Full resume text
        collection_name: ChromaDB collection name

    Returns:
        Chroma vector store instance
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(resume_text)

    docs = [
        Document(page_content=chunk, metadata={"source": "resume", "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    os.makedirs(settings.chroma_path, exist_ok=True)

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=settings.chroma_path,
    )

    return store


def load_resume_store(collection_name: str = "resume") -> Chroma:
    """Load an existing resume vector store from disk."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )


def retrieve_relevant_chunks(
    query: str,
    store: Chroma,
    k: int = 4,
) -> list[str]:
    """
    Retrieve the most relevant resume chunks for a given query (JD text).

    Args:
        query: The job description or query text
        store: Chroma vector store to search
        k: Number of chunks to retrieve

    Returns:
        List of relevant text chunks
    """
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


# ── JD Vector Store ───────────────────────────────────────────────────────────


def build_jd_store(job_descriptions: list[dict], collection_name: str = "jd_store") -> Chroma:
    """
    Embed all job descriptions into ChromaDB for similarity search.

    Args:
        job_descriptions: List of job dicts with 'description', 'company', 'title'
        collection_name: ChromaDB collection name

    Returns:
        Chroma vector store instance
    """
    docs = []
    for i, jd in enumerate(job_descriptions):
        content = f"{jd.get('title', '')} at {jd.get('company', '')}\n{jd.get('description', '')}"
        docs.append(
            Document(
                page_content=content[:1000],
                metadata={
                    "company": jd.get("company", ""),
                    "title": jd.get("title", ""),
                    "url": jd.get("url", ""),
                    "index": i,
                },
            )
        )

    os.makedirs(settings.chroma_path, exist_ok=True)

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=settings.chroma_path,
    )


def find_similar_jobs(query: str, store: Chroma, k: int = 5) -> list[dict]:
    """
    Find jobs similar to a given query from the JD store.

    Returns:
        List of job metadata dicts
    """
    results = store.similarity_search_with_score(query, k=k)
    return [{**doc.metadata, "similarity_score": round(score, 3)} for doc, score in results]


def clear_collection(collection_name: str) -> None:
    """Delete a ChromaDB collection."""
    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )
    store.delete_collection()
    print(f"✅ Cleared collection: {collection_name}")


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.print("[bold blue]🧠 Testing Vector Store...[/bold blue]")

    sample_resume = """
    Engineering Manager with 21 years experience in Golang, AWS, Kubernetes.
    Led team at CloudBees delivering SaaS DevSecOps product.
    Built Golang microservices from scratch to production in 9 months.
    Certified CKA, AWS Solutions Architect Professional.
    Skilled in Kafka, MongoDB, ReactJS, Docker, CI/CD.
    """

    store = build_resume_store(sample_resume, collection_name="test_resume")
    console.print("[green]✅ Resume store built[/green]")

    chunks = retrieve_relevant_chunks(
        query="Engineering Manager Golang AWS Kubernetes DevSecOps",
        store=store,
        k=3,
    )

    console.print(f"\n[yellow]Top {len(chunks)} relevant chunks:[/yellow]")
    for i, chunk in enumerate(chunks):
        console.print(f"\n[dim]{i+1}. {chunk[:150]}[/dim]")

    clear_collection("test_resume")
    console.print("\n[green]✅ Test collection cleared[/green]")
