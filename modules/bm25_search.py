"""
Sparse lexical search engine for ResearchLens AI using BM25.
Tokenizes academic passages, filters stopwords, and performs Okapi BM25 ranking.
"""

import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from core.logging_config import get_logger
from core.schemas import SearchResult

logger = get_logger("bm25_search")

# Basic English stopwords for lexical search
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text into lowercased alphanumeric tokens, removing stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25SearchEngine:
    """Manages in-memory and persisted BM25 models per collection."""

    def __init__(self) -> None:
        self._models: Dict[str, BM25Okapi] = {}
        self._chunks: Dict[str, List[Dict[str, Any]]] = {}

    def index_chunks(self, collection_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Builds an Okapi BM25 model over the provided chunks for a collection."""
        if not chunks:
            self._models.pop(collection_id, None)
            self._chunks.pop(collection_id, None)
            return

        tokenized_corpus = [tokenize_for_bm25(c["text"]) for c in chunks]
        self._models[collection_id] = BM25Okapi(tokenized_corpus)
        self._chunks[collection_id] = chunks
        logger.info(f"Indexed {len(chunks)} chunks in BM25 for collection {collection_id}.")

    def search(
        self,
        collection_id: str,
        query: str,
        top_k: int = 15,
        filter_paper_id: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Executes BM25 lexical scoring for the query against collection chunks.
        Returns top_k SearchResult objects with bm25_score.
        """
        if collection_id not in self._models or collection_id not in self._chunks:
            from database.db_manager import DatabaseManager
            db = DatabaseManager()
            all_chunks = db.get_all_chunks(collection_id)
            if all_chunks:
                self.index_chunks(collection_id, all_chunks)
            else:
                return []

        bm25 = self._models[collection_id]
        chunks = self._chunks[collection_id]
        query_tokens = tokenize_for_bm25(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1]

        results: List[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            chunk = chunks[idx]

            # Apply filters
            if filter_paper_id and chunk.get("paper_id") != filter_paper_id:
                continue
            if filter_section and chunk.get("section_name", "").lower() != filter_section.lower():
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk["id"],
                    paper_id=chunk["paper_id"],
                    paper_title=chunk.get("paper_title", "Unknown Title"),
                    section_name=chunk["section_name"],
                    page_start=chunk["page_start"],
                    page_end=chunk["page_end"],
                    text=chunk["text"],
                    bm25_score=round(score, 4),
                )
            )

            if len(results) >= top_k:
                break

        return results
