"""
PulseAgent — Central Configuration
All settings loaded from environment variables via pydantic-settings.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    # Agent → LLM routing
    classifier_llm: str = "groq"
    scorer_llm: str = "groq"
    rag_llm: str = "gemini"
    response_llm: str = "gemini"
    roadmap_llm: str = "gemini"
    trend_llm: str = "groq"

    # ── Vector DB ─────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "product_docs"

    # ── API ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── App ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    max_reviews_per_run: int = 100
    churn_urgency_multiplier: float = 2.5
    trend_window_days: int = 14


@lru_cache
def get_settings() -> Settings:
    return Settings()