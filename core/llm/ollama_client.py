"""
Ollama client for ResearchLens AI supporting fully local, offline inference.
Supports Ollama local HTTP API with JSON-mode structured extraction and health checks.
"""

import json
from typing import Any, Dict, Optional, Type, TypeVar

import requests
from pydantic import BaseModel

from core.llm.base import LLMClient
from core.logging_config import get_logger

logger = get_logger("llm.ollama")
T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):
    """Client for locally hosted Ollama models."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.1:8b",
        enable_cache: bool = True,
    ) -> None:
        super().__init__(model_name=model_name, enable_cache=enable_cache)
        self.base_url = base_url.rstrip("/")

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

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 2048,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "")
            self.set_cached(prompt_hash, text)
            return text
        except requests.exceptions.ConnectionError as err:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from err
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

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

        schema_json = json.dumps(schema.model_json_schema())
        instruction = (
            f"You MUST respond ONLY with valid JSON conforming to this schema:\n{schema_json}\n\n"
        )
        full_system = f"{system_prompt or ''}\n\n{instruction}".strip()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": full_system,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "{}")
            parsed = schema.model_validate_json(raw_text)
            self.set_cached(prompt_hash, raw_text)
            return parsed
        except requests.exceptions.ConnectionError as err:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from err
        except Exception as e:
            raise RuntimeError(f"Ollama structured generation failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Checks if Ollama daemon is running and lists local models."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                is_model_present = any(self.model_name in m for m in models)
                return {
                    "status": "healthy" if is_model_present else "warning",
                    "message": (
                        f"Ollama running. Model '{self.model_name}' available."
                        if is_model_present
                        else f"Ollama running, but model '{self.model_name}' not downloaded. Run `ollama pull {self.model_name}`."
                    ),
                    "model": self.model_name,
                    "available_models": models,
                }
            return {
                "status": "error",
                "message": f"Ollama returned HTTP {res.status_code}",
                "model": self.model_name,
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "offline",
                "message": f"Ollama is not running at {self.base_url}. Run `ollama serve` to use local models.",
                "model": self.model_name,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Ollama health check error: {e}",
                "model": self.model_name,
            }
