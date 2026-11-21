"""
Dense vector embedding module for ResearchLens AI.
Wraps SentenceTransformers with L2-normalized BAAI/bge-small-en-v1.5 embeddings
for inner-product cosine similarity search.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging_config import get_logger

logger = get_logger("embeddings")

_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy loader for SentenceTransformer singleton."""
    global _model_instance
    if _model_instance is None:
        model_name = settings.embedding_model_name
        device = settings.embedding_device
        logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
        _model_instance = SentenceTransformer(model_name, device=device)
    return _model_instance


class EmbeddingEngine:
    """Generates L2-normalized dense embeddings for documents and queries."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.embedding_model_name

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of texts into L2-normalized float32 numpy vectors.
        Shape: (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = get_embedding_model()
        # normalize_embeddings=True ensures inner product equals cosine similarity
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into an L2-normalized float32 vector.
        Shape: (1, embedding_dim)
        """
        model = get_embedding_model()
        emb = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)
