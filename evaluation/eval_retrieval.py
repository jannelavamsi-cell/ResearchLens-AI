"""
Evaluation harness: Retrieval Ablation Study.
Benchmarks Dense-only (FAISS), BM25-only, Hybrid (RRF), and Hybrid + Cross-Encoder Rerank.
Computes Precision@K, Recall@K, MRR, and nDCG@K.
"""

import logging
import math
from typing import Any, Dict, List

from database.db_manager import DatabaseManager
from modules.bm25_search import BM25SearchEngine
from modules.hybrid_search import compute_rrf_scores
from modules.reranking import CrossEncoderReranker
from modules.semantic_search import SemanticSearchEngine

logger = logging.getLogger("researchlens.eval.retrieval")


def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        rank = i + 1
        dcg += rel / math.log2(rank + 1)
    return dcg


def compute_ndcg(relevances: List[int], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def is_chunk_relevant(chunk_text: str, relevant_keywords: List[str]) -> bool:
    """Determines binary relevance based on presence of key target terms."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in relevant_keywords if kw.lower() in text_lower)
    return matches >= 1


def evaluate_retrieval_ablation(
    queries: List[Dict[str, Any]],
    collection_id: str,
    db: DatabaseManager,
    k_eval: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Runs multi-stage ablation comparing:
    1. Dense-only (FAISS)
    2. BM25-only (Okapi)
    3. Hybrid (RRF)
    4. Hybrid + Cross-Encoder Rerank
    """
    semantic_engine = SemanticSearchEngine()
    bm25_engine = BM25SearchEngine()
    reranker = CrossEncoderReranker()

    modes = ["Dense Only", "BM25 Only", "Hybrid (RRF)", "Hybrid + Reranker"]
    metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"p_at_k": [], "r_at_k": [], "mrr": [], "ndcg": []} for m in modes
    }

    for item in queries:
        query = item["query"]
        target_keywords = item["relevant_keywords"]

        # 1. Dense retrieval
        dense_hits = semantic_engine.search(collection_id=collection_id, query=query, top_k=10)

        # 2. BM25 retrieval
        bm25_hits = bm25_engine.search(collection_id=collection_id, query=query, top_k=10)

        # 3. Hybrid RRF
        rrf_hits = compute_rrf_scores([dense_hits, bm25_hits], k=60)

        # 4. Reranked
        reranked_hits = reranker.rerank(query, rrf_hits[:10], top_k=10) if rrf_hits else []

        eval_sets = {
            "Dense Only": dense_hits[:k_eval],
            "BM25 Only": bm25_hits[:k_eval],
            "Hybrid (RRF)": rrf_hits[:k_eval],
            "Hybrid + Reranker": reranked_hits[:k_eval],
        }

        for mode, hits in eval_sets.items():
            relevances = [1 if is_chunk_relevant(h.text, target_keywords) else 0 for h in hits]

            # Precision@K
            p_k = sum(relevances) / max(1, k_eval)
            metrics[mode]["p_at_k"].append(p_k)

            # Recall@K (assume total relevant is at least 1)
            r_k = 1.0 if sum(relevances) > 0 else 0.0
            metrics[mode]["r_at_k"].append(r_k)

            # MRR
            mrr = 0.0
            for rank_0, rel in enumerate(relevances):
                if rel == 1:
                    mrr = 1.0 / (rank_0 + 1)
                    break
            metrics[mode]["mrr"].append(mrr)

            # nDCG
            ndcg_val = compute_ndcg(relevances, k_eval)
            metrics[mode]["ndcg"].append(ndcg_val)

    # Average metrics
    summary: Dict[str, Dict[str, float]] = {}
    for mode, scores in metrics.items():
        summary[mode] = {
            f"Precision@{k_eval}": round(sum(scores["p_at_k"]) / max(1, len(scores["p_at_k"])), 3),
            f"Recall@{k_eval}": round(sum(scores["r_at_k"]) / max(1, len(scores["r_at_k"])), 3),
            "MRR": round(sum(scores["mrr"]) / max(1, len(scores["mrr"])), 3),
            f"nDCG@{k_eval}": round(sum(scores["ndcg"]) / max(1, len(scores["ndcg"])), 3),
        }

    return summary
