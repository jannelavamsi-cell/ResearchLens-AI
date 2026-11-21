"""
OpenAI-compatible client for ResearchLens AI.
Supports OpenAI API as well as any compatible proxy or self-hosted endpoint
(Groq, vLLM, DeepSeek, LocalAI, Together, LM Studio).
"""

import os
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from core.llm.base import LLMClient
from core.logging_config import get_logger

logger = get_logger("llm.openai")
T = TypeVar("T", bound=BaseModel)


class OpenAICompatClient(LLMClient):
    """Client for OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        enable_cache: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, enable_cache=enable_cache)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = None

        if self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key or "local-key", base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
    ) -> str:
        prompt_hash = self._compute_hash(prompt, system_prompt, temperature)
        cached = self.get_cached(prompt_hash)
        if cached:
            return cached

        if not self.client:
            raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY in .env.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            self.set_cached(prompt_hash, text)
            return text
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        prompt_hash = self._compute_hash(
            f"STRUCT_{schema.__name__}|{prompt}", system_prompt, temperature
        )
        cached = self.get_cached(prompt_hash)
        if cached:
            try:
                return schema.model_validate_json(cached)
            except Exception:
                pass

        if not self.client:
            raise RuntimeError("OpenAI client is not configured. Set OPENAI_API_KEY in .env.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use beta parse if available, otherwise JSON format fallback
            if hasattr(self.client.beta.chat.completions, "parse"):
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                    temperature=temperature,
                )
                parsed = completion.choices[0].message.parsed
                if parsed:
                    self.set_cached(prompt_hash, parsed.model_dump_json())
                    return parsed

            # Fallback to response_format json_object
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_json = response.choices[0].message.content or "{}"
            parsed = schema.model_validate_json(raw_json)
            self.set_cached(prompt_hash, raw_json)
            return parsed
        except Exception as e:
            raise RuntimeError(f"OpenAI structured generation failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Checks OpenAI endpoint connectivity."""
        if not self.client:
            return {
                "status": "unconfigured",
                "message": "OpenAI client is not configured. Set OPENAI_API_KEY in .env.",
                "model": self.model_name,
            }
        try:
            self.client.models.list()
            return {
                "status": "healthy",
                "message": f"Connected to endpoint {self.base_url} ({self.model_name}).",
                "model": self.model_name,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"OpenAI health check error: {e}",
                "model": self.model_name,
            }
