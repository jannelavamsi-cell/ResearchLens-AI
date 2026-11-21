"""
Base LLM client interface for ResearchLens AI.
Provides abstract methods for text and structured generation, response caching in SQLite,
token estimation, and retry handling.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from core.logging_config import get_logger

logger = get_logger("llm.base")

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract base class for all LLM backend implementations."""

    def __init__(self, model_name: str, enable_cache: bool = True) -> None:
        self.model_name = model_name
        self.enable_cache = enable_cache
        # Lazy imported to avoid circular dependency
        from core.config import settings
        from database.db_manager import DBManager

        self.db = DBManager(settings.database_path)

    def _compute_hash(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0
    ) -> str:
        """Calculates SHA-256 hash for caching."""
        content = f"{self.model_name}|{temperature:.2f}|{system_prompt or ''}|{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_cached(self, prompt_hash: str) -> Optional[str]:
        """Retrieves cached response if caching is enabled."""
        if not self.enable_cache:
            return None
        return self.db.get_cached_llm_response(prompt_hash)

    def set_cached(self, prompt_hash: str, response_text: str) -> None:
        """Saves generated response into the cache."""
        if self.enable_cache:
            self.db.cache_llm_response(prompt_hash, response_text, self.model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
    ) -> str:
        """Convenience alias for generate_text."""
        return self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
    ) -> str:
        """Generates raw text response for the given prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """Generates a structured Pydantic object conforming to the provided schema."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Verifies connectivity, authentication, and readiness of the backend."""
        pass
