"""
Cross-Encoder reranking module for ResearchLens AI.
Uses ms-marco-MiniLM-L-6-v2 to evaluate query-passage semantic relevance pairs
and re-order hybrid retrieval candidates.
"""

from typing import List, Optional

from sentence_transformers import CrossEncoder

from core.config import settings
from core.logging_config import get_logger
from core.schemas import SearchResult

logger = get_logger("reranking")

_reranker_instance: Optional[CrossEncoder] = None


def get_reranker_model() -> CrossEncoder:
    """Lazy loader for CrossEncoder singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        model_name = settings.reranker_model_name
        logger.info(f"Loading Cross-Encoder reranker '{model_name}'...")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance


class CrossEncoderReranker:
    """Reranks candidate search passages using a deep Cross-Encoder."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.reranker_model_name

    def rerank(
        self, query: str, candidates: List[SearchResult], top_k: int = 5
    ) -> List[SearchResult]:
        """
        Re-scores candidates using Cross-Encoder.
        Pairs: [(query, c.text) for c in candidates]
        Sorts descending by rerank_score and returns top_k.
        """
        if not candidates:
            return []

        model = get_reranker_model()
        pairs = [[query, c.text] for c in candidates]

        scores = model.predict(pairs)

        for c, score in zip(candidates, scores, strict=False):
            c.rerank_score = round(float(score), 4)

        reranked = sorted(candidates, key=lambda x: x.rerank_score or -999.0, reverse=True)
        return reranked[:top_k]
