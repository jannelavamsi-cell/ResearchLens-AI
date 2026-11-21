"""
Google Gemini LLM client for ResearchLens AI using the google-genai SDK.
Supports text generation, native Pydantic structured output, and health checks.
"""

import os
import time
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from core.llm.base import LLMClient
from core.logging_config import get_logger

logger = get_logger("llm.gemini")
T = TypeVar("T", bound=BaseModel)


class GeminiClient(LLMClient):
    """Client for Google Gemini models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        enable_cache: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, enable_cache=enable_cache)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = None

        if self.api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai Client: {e}")

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
            logger.debug(f"Cache hit for prompt hash {prompt_hash[:8]}")
            return cached

        if not self.client:
            raise RuntimeError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(3):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_prompt if system_prompt else None,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""
                self.set_cached(prompt_hash, text)
                return text
            except Exception as e:
                last_error = e
                # If network is completely unreachable (e.g. sandbox DNS block), fail fast without retrying
                if any(err_msg in str(e).lower() for err_msg in ["getaddrinfo failed", "connecterror", "connection refused"]):
                    logger.debug(f"Gemini client detected offline network state: {e}")
                    break
                wait_time = 2**attempt
                logger.warning(
                    f"Gemini API error (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(f"Gemini generation failed after 3 attempts: {last_error}")

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
            except Exception as e:
                logger.warning(f"Corrupted cache entry for {schema.__name__}: {e}")

        if not self.client:
            raise RuntimeError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        last_error = None
        for attempt in range(3):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    system_instruction=system_prompt if system_prompt else None,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                raw_json = response.text or "{}"
                # Clean possible markdown fence wrappers
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                raw_json = raw_json.strip()

                parsed = schema.model_validate_json(raw_json)
                self.set_cached(prompt_hash, raw_json)
                return parsed
            except Exception as e:
                last_error = e
                if any(err_msg in str(e).lower() for err_msg in ["getaddrinfo failed", "connecterror", "connection refused"]):
                    logger.debug(f"Gemini client detected offline network state: {e}")
                    break
                wait_time = 2**attempt
                logger.warning(
                    f"Gemini structured error (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(f"Gemini structured generation failed after 3 attempts: {last_error}")

    def health_check(self) -> Dict[str, Any]:
        """Checks Gemini API connectivity."""
        if not self.api_key:
            return {
                "status": "unconfigured",
                "message": "GEMINI_API_KEY is not set in environment or .env.",
                "model": self.model_name,
            }
        try:
            # Lightweight probe
            test_response = self.generate_text(
                prompt="Say 'ResearchLens Ready'",
                temperature=0.0,
                max_tokens=10,
            )
            return {
                "status": "healthy",
                "message": f"Connected to Gemini API ({self.model_name}). Probe: {test_response.strip()}",
                "model": self.model_name,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Gemini API check failed: {str(e)}",
                "model": self.model_name,
            }
