"""
PulseAgent — LLM Factory
Returns the correct LangChain LLM instance based on config.
Groq → fast, free tier generous → used for classification & scoring.
Gemini → larger context, better reasoning → used for RAG, response, roadmap.
"""
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from config import get_settings


def get_llm(role: str) -> BaseChatModel:
    """
    Return LLM instance for a given agent role.
    Role → provider mapping is set in .env (e.g. CLASSIFIER_LLM=groq).
    """
    settings = get_settings()
    provider_map = {
        "classifier": settings.classifier_llm,
        "scorer": settings.scorer_llm,
        "rag": settings.rag_llm,
        "response": settings.response_llm,
        "roadmap": settings.roadmap_llm,
        "trend": settings.trend_llm,
    }

    provider = provider_map.get(role, "groq")
    logger.debug(f"[LLM Factory] role={role} → provider={provider}")

    if provider == "groq":
        return _get_groq()
    elif provider == "gemini":
        return _get_gemini()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache
def _get_groq() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.1,
        max_retries=3,
    )


@lru_cache
def _get_gemini() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=0.2,
        max_retries=3,
    )