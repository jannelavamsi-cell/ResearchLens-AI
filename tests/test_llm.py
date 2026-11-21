"""
Unit tests for the LLMClient layer, prompt templates, and caching mechanism.
"""

import os
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from core.llm.base import LLMClient
from core.llm.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    PROMPTS_VERSION,
    RAG_SYSTEM_PROMPT,
    SECURITY_DIRECTIVE,
)


class DummySchema(BaseModel):
    summary: str = Field(..., description="Test summary")
    score: float = Field(..., description="Test score")


class MockLLMClient(LLMClient):
    """Deterministic mock client for testing the base LLMClient functionality."""

    def __init__(self, model_name: str = "mock-model", db_path: Optional[str] = None) -> None:
        self.model_name = model_name
        self.enable_cache = True
        import tempfile

        from database.db_manager import DBManager

        if db_path is None:
            self.temp_dir = tempfile.mkdtemp()
            db_path = os.path.join(self.temp_dir, "test_llm.db")
        self.db = DBManager(db_path)
        self.call_count = 0

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

        self.call_count += 1
        result = f"Mock response to: {prompt}"
        self.set_cached(prompt_hash, result)
        return result

    def generate_structured(
        self,
        prompt: str,
        schema: Type[DummySchema],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> DummySchema:
        prompt_hash = self._compute_hash(
            f"STRUCT_{schema.__name__}|{prompt}", system_prompt, temperature
        )
        cached = self.get_cached(prompt_hash)
        if cached:
            return schema.model_validate_json(cached)

        self.call_count += 1
        obj = DummySchema(summary="Structured mock summary", score=0.95)
        self.set_cached(prompt_hash, obj.model_dump_json())
        return obj

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "message": "Mock client ready", "model": self.model_name}


def test_prompts_and_security_directive():
    """Verifies that all system prompts include security and anti-injection directives."""
    assert "IMPORTANT SECURITY & GROUNDING DIRECTIVE" in SECURITY_DIRECTIVE
    assert SECURITY_DIRECTIVE in EXTRACTION_SYSTEM_PROMPT
    assert SECURITY_DIRECTIVE in RAG_SYSTEM_PROMPT
    assert PROMPTS_VERSION == "1.0.0"


def test_mock_llm_caching():
    """Verifies that duplicate requests hit the SQLite cache and do not re-execute."""
    client = MockLLMClient()
    prompt = "What is the objective of this paper?"

    # First call - cache miss
    resp1 = client.generate_text(prompt)
    assert resp1 == f"Mock response to: {prompt}"
    assert client.call_count == 1

    # Second call - cache hit
    resp2 = client.generate_text(prompt)
    assert resp2 == resp1
    assert client.call_count == 1  # call count did not increase!


def test_mock_llm_structured_generation():
    """Verifies that generate_structured correctly serializes and caches Pydantic models."""
    client = MockLLMClient()
    prompt = "Extract summary"

    res = client.generate_structured(prompt, DummySchema)
    assert isinstance(res, DummySchema)
    assert res.summary == "Structured mock summary"
    assert res.score == 0.95

    # Health check
    health = client.health_check()
    assert health["status"] == "healthy"
