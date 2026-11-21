"""
Hybrid Search Engine for ResearchLens AI using Reciprocal Rank Fusion (RRF).
Fuses dense vector rankings (FAISS) and sparse lexical rankings (BM25)
using the standard RRF formula with k=60.
"""

from typing import Any, Dict, List, Optional

from core.config import settings
from core.logging_config import get_logger
from core.schemas import SearchResult
from modules.bm25_search import BM25SearchEngine
from modules.reranking import CrossEncoderReranker
from modules.semantic_search import SemanticSearchEngine

logger = get_logger("hybrid_search")


def compute_rrf_scores(
    ranked_lists: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores across multiple ranked lists.
    Formula: RRF_score(d) = sum(1 / (k + rank_m(d)))
    """
    rrf_map: Dict[str, float] = {}
    item_map: Dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank_0, item in enumerate(ranked_list):
            rank = rank_0 + 1  # 1-indexed rank
            chunk_id = item.chunk_id
            reciprocal_rank = 1.0 / (k + rank)
            rrf_map[chunk_id] = rrf_map.get(chunk_id, 0.0) + reciprocal_rank

            # Store / update item representation with dense or bm25 scores
            if chunk_id not in item_map:
                item_map[chunk_id] = item
            else:
                existing = item_map[chunk_id]
                if item.dense_score is not None:
                    existing.dense_score = item.dense_score
                if item.bm25_score is not None:
                    existing.bm25_score = item.bm25_score

    # Assign final RRF score and sort descending
    fused_results: List[SearchResult] = []
    for chunk_id, score in rrf_map.items():
        res = item_map[chunk_id]
        res.rrf_score = round(score, 6)
        fused_results.append(res)

    fused_results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
    return fused_results


class HybridSearchEngine:
    """Orchestrates dense, sparse, RRF fusion, and Cross-Encoder reranking."""

    def __init__(
        self,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        bm25_engine: Optional[BM25SearchEngine] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.semantic_engine = semantic_engine or SemanticSearchEngine()
        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        collection_id: str,
        query: str,
        top_k: int = 5,
        top_k_dense: int = 15,
        top_k_bm25: int = 15,
        top_k_rrf: int = 20,
        filter_paper_id: Optional[str] = None,
        filter_section: Optional[str] = None,
        enable_reranker: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Executes end-to-end hybrid retrieval:
        1. Dense retrieval via FAISS (top_k_dense)
        2. Sparse retrieval via BM25 (top_k_bm25)
        3. Reciprocal Rank Fusion (RRF k=60)
        4. Cross-Encoder reranking (top_k_rrf -> top_k)
        """
        if enable_reranker is None:
            enable_reranker = settings.enable_reranker

        logger.info(f"Hybrid search query: '{query}' in collection {collection_id}")

        # 1. Dense retrieval
        dense_results = self.semantic_engine.search(
            collection_id=collection_id,
            query=query,
            top_k=top_k_dense,
            filter_paper_id=filter_paper_id,
            filter_section=filter_section,
        )

        # 2. Sparse retrieval
        bm25_results = self.bm25_engine.search(
            collection_id=collection_id,
            query=query,
            top_k=top_k_bm25,
            filter_paper_id=filter_paper_id,
            filter_section=filter_section,
        )

        # 3. Reciprocal Rank Fusion
        fused = compute_rrf_scores([dense_results, bm25_results], k=60)
        fusion_candidates = fused[:top_k_rrf]

        # 4. Cross-Encoder reranking
        if enable_reranker and fusion_candidates:
            reranked = self.reranker.rerank(query, fusion_candidates, top_k=top_k)
            return reranked

        return fusion_candidates[:top_k]


def hybrid_search(
    query: str,
    collection_id: str,
    paper_ids: Optional[List[str]] = None,
    top_k: int = 5,
    db: Optional[Any] = None,
    filter_section: Optional[str] = None,
    enable_reranker: Optional[bool] = None,
) -> List[SearchResult]:
    """Convenience function for end-to-end hybrid retrieval with optional paper filtering."""
    engine = HybridSearchEngine()
    filter_paper_id = paper_ids[0] if (paper_ids and len(paper_ids) == 1) else None
    results = engine.search(
        collection_id=collection_id,
        query=query,
        top_k=top_k * 2 if (paper_ids and len(paper_ids) > 1) else top_k,
        filter_paper_id=filter_paper_id,
        filter_section=filter_section,
        enable_reranker=enable_reranker,
    )
    if paper_ids and len(paper_ids) > 1:
        results = [r for r in results if r.paper_id in paper_ids]
    return results[:top_k]

