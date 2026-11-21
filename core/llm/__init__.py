"""
LLM factory module for ResearchLens AI.
Provides a unified factory function to instantiate the active LLM client
based on application configuration.
"""

from typing import Optional

from core.config import settings
from core.llm.base import LLMClient
from core.llm.gemini_client import GeminiClient
from core.llm.ollama_client import OllamaClient
from core.llm.openai_compat_client import OpenAICompatClient
from core.logging_config import get_logger

logger = get_logger("llm.factory")

_client_instance: Optional[LLMClient] = None


def get_llm_client(force_reload: bool = False) -> LLMClient:
    """Returns the configured LLMClient singleton based on current settings."""
    global _client_instance
    if _client_instance is not None and not force_reload:
        return _client_instance

    provider = settings.llm_provider.lower()
    logger.info(f"Initializing LLM client with provider: {provider}")

    if provider == "gemini":
        _client_instance = GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )
    elif provider == "ollama":
        _client_instance = OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model,
        )
    elif provider == "openai":
        _client_instance = OpenAICompatClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.openai_model,
        )
    else:
        logger.warning(f"Unknown provider '{provider}'. Defaulting to Gemini.")
        _client_instance = GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
        )

    return _client_instance
