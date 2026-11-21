"""
Configuration settings for ResearchLens AI using Pydantic Settings.
Loads values from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Literal

# Ensure HuggingFace Hub uses cached local models without network timeout
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for ResearchLens AI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: Literal["gemini", "ollama", "openai"] = Field(
        default="gemini", description="Selected LLM backend provider"
    )

    # Google Gemini
    gemini_api_key: str = Field(default="", description="Google AI Studio API Key")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", description="Gemini model name")

    # Ollama (Local Privacy Mode)
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.1:8b", description="Ollama model tag")

    # OpenAI / Compatible Endpoint
    openai_api_key: str = Field(default="", description="OpenAI or compatible API Key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI compatible base URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")

    # Embedding and Reranker
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Sentence Transformers HuggingFace ID"
    )
    embedding_device: str = Field(default="cpu", description="Compute device: cpu or cuda")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-Encoder model ID"
    )
    enable_reranker: bool = Field(default=True, description="Enable Cross-Encoder reranker")

    # Storage and Paths
    database_path: str = Field(default="data/researchlens.db", description="Path to SQLite DB")
    data_dir: str = Field(default="data", description="Root data directory")
    index_dir: str = Field(default="data/indexes", description="FAISS index directory")
    upload_dir: str = Field(default="data/uploaded_papers", description="Uploaded PDF storage")

    # Chunking & Retrieval Parameters
    chunk_size_tokens: int = Field(default=350, description="Target chunk size in tokens")
    chunk_overlap_tokens: int = Field(default=50, description="Token overlap between chunks")
    top_k_dense: int = Field(default=15, description="Top-k dense candidates from FAISS")
    top_k_bm25: int = Field(default=15, description="Top-k sparse candidates from BM25")
    top_k_rrf: int = Field(default=20, description="Top-k candidates passed to reranker")
    top_k_final: int = Field(default=5, description="Final top-k chunks in prompt context")

    # Verification Settings
    enable_citation_verifier: bool = Field(
        default=True, description="Enable post-generation citation verification"
    )
    verifier_threshold: float = Field(
        default=0.85, description="Fuzzy string match ratio threshold for quote grounding"
    )

    def ensure_directories(self) -> None:
        """Creates required application directories if they do not exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("tests/fixtures").mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
