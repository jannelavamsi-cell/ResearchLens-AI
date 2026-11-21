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


def test_gemini_client_unconfigured(monkeypatch):
    import pytest

    from core.llm.gemini_client import GeminiClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient(api_key="", enable_cache=False)
    assert client.health_check()["status"] == "unconfigured"
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        client.generate_text("test")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        client.generate_structured("test", DummySchema)


def test_gemini_client_mock_generation():
    from unittest.mock import MagicMock

    from core.llm.gemini_client import GeminiClient

    client = GeminiClient(api_key="mock_key", enable_cache=False)
    mock_genai = MagicMock()
    client.client = mock_genai

    # Text generation
    mock_genai.models.generate_content.return_value = MagicMock(text="Generated text response")
    res = client.generate_text("Prompt")
    assert res == "Generated text response"

    # Structured generation
    mock_genai.models.generate_content.return_value = MagicMock(text='{"summary": "Test Summary", "score": 0.88}')
    res_struct = client.generate_structured("Prompt", DummySchema)
    assert isinstance(res_struct, DummySchema)
    assert res_struct.summary == "Test Summary"
    assert res_struct.score == 0.88

    # Health check
    health = client.health_check()
    assert health["status"] == "healthy"


def test_ollama_client_mock():
    from unittest.mock import MagicMock, patch

    import pytest
    import requests

    from core.llm.ollama_client import OllamaClient

    client = OllamaClient(base_url="http://localhost:11434", model_name="llama3.1:8b", enable_cache=False)

    # Health check healthy
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": [{"name": "llama3.1:8b"}]})
        h = client.health_check()
        assert h["status"] == "healthy"

    # Health check offline
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Offline")):
        h = client.health_check()
        assert h["status"] == "offline"
        assert "not running" in h["message"]

    # Text generation
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"response": "Ollama answer"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        text = client.generate_text("Test prompt")
        assert text == "Ollama answer"

    # Structured generation
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"response": '{"summary": "Ollama struct", "score": 0.9}'})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        struct = client.generate_structured("Test prompt", DummySchema)
        assert struct.summary == "Ollama struct"

    # Connection error during generation
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Ollama offline")):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            client.generate_text("Test prompt")


def test_openai_compat_client_mock():
    from unittest.mock import MagicMock

    from core.llm.openai_compat_client import OpenAICompatClient

    # Unconfigured
    client_unconf = OpenAICompatClient(api_key="", enable_cache=False)
    assert client_unconf.health_check()["status"] == "unconfigured"

    # Configured with mock
    client = OpenAICompatClient(api_key="mock_key", enable_cache=False)
    mock_openai = MagicMock()
    client.client = mock_openai

    # Text generation
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI response text"
    mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    assert client.generate_text("Prompt") == "OpenAI response text"

    # Structured generation
    mock_choice_struct = MagicMock()
    mock_choice_struct.message.parsed = DummySchema(summary="OpenAI struct", score=0.99)
    mock_openai.beta.chat.completions.parse.return_value = MagicMock(choices=[mock_choice_struct])
    parsed = client.generate_structured("Prompt", DummySchema)
    assert parsed.summary == "OpenAI struct"

    # Health check healthy
    mock_openai.models.list.return_value = ["gpt-4o-mini"]
    assert client.health_check()["status"] == "healthy"

    # Health check error
    mock_openai.models.list.side_effect = Exception("Unauthorized")
    assert client.health_check()["status"] == "error"

