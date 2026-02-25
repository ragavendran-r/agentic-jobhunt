from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM APIs
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")  # type: ignore
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")  # pyright: ignore[reportCallIssue]

    # Search
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")  # pyright: ignore[reportCallIssue]

    # Observability
    langchain_api_key: str = Field("", env="LANGCHAIN_API_KEY")  # pyright: ignore[reportCallIssue]
    langchain_tracing_v2: str = Field(
        "true", env="LANGCHAIN_TRACING_V2"
    )  # pyright: ignore[reportCallIssue]
    langchain_project: str = Field("agentic-jobhunt", env="LANGCHAIN_PROJECT")  # type: ignore

    # App Config
    resume_path: str = Field("resume.pdf", env="RESUME_PATH")  # type: ignore
    db_path: str = Field("data/jobs.db", env="DB_PATH")  # type: ignore
    chroma_path: str = Field("data/chroma", env="CHROMA_PATH")  # type: ignore
    log_level: str = Field("INFO", env="LOG_LEVEL")  # type: ignore

    # Model names
    gemini_model: str = "gemini/gemini-2.5-flash"
    openai_model: str = "gpt-4o"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()  # type: ignore
