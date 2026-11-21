"""
Dense vector semantic search engine for ResearchLens AI using FAISS.
Manages disk-persisted IndexFlatIP indices per collection with incremental indexing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss

from core.config import settings
from core.logging_config import get_logger
from core.schemas import SearchResult
from modules.embeddings import EmbeddingEngine

logger = get_logger("semantic_search")


class SemanticSearchEngine:
    """Manages FAISS inner product indices and vector retrieval per collection."""

    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = EmbeddingEngine()
        self._indices: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}

    def _index_path(self, collection_id: str) -> Path:
        return self.index_dir / f"{collection_id}.faiss"

    def _meta_path(self, collection_id: str) -> Path:
        return self.index_dir / f"{collection_id}.meta.json"

    def _load_or_create(self, collection_id: str, dim: int = 384) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Loads FAISS index and metadata from disk, or initializes an empty IndexFlatIP."""
        if collection_id in self._indices:
            return self._indices[collection_id], self._metadata[collection_id]

        idx_file = self._index_path(collection_id)
        meta_file = self._meta_path(collection_id)

        if idx_file.exists() and meta_file.exists():
            try:
                index = faiss.read_index(str(idx_file))
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._indices[collection_id] = index
                self._metadata[collection_id] = meta
                logger.debug(f"Loaded FAISS index for {collection_id} with {index.ntotal} vectors.")
                return index, meta
            except Exception as e:
                logger.warning(f"Failed to load existing index for {collection_id}: {e}. Creating new.")

        # Initialize fresh index
        index = faiss.IndexFlatIP(dim)
        meta = []
        self._indices[collection_id] = index
        self._metadata[collection_id] = meta
        return index, meta

    def _save(self, collection_id: str) -> None:
        """Saves active FAISS index and metadata to disk."""
        if collection_id in self._indices:
            idx_file = self._index_path(collection_id)
            meta_file = self._meta_path(collection_id)
            faiss.write_index(self._indices[collection_id], str(idx_file))
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self._metadata[collection_id], f, indent=2)
            logger.debug(f"Saved FAISS index for {collection_id} to disk.")

    def index_chunks(self, collection_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Embeds and appends a list of chunks to the collection's FAISS index."""
        if not chunks:
            return

        index, meta = self._load_or_create(collection_id)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        index.add(embeddings)
        meta.extend(chunks)
        self._save(collection_id)
        logger.info(f"Indexed {len(chunks)} chunks into FAISS for collection {collection_id}.")

    def rebuild_index(self, collection_id: str, all_chunks: List[Dict[str, Any]]) -> None:
        """Completely rebuilds the FAISS index for a collection from scratch."""
        dim = 384
        index = faiss.IndexFlatIP(dim)
        meta = []

        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            embeddings = self.embedder.embed_texts(texts)
            index.add(embeddings)
            meta.extend(all_chunks)

        self._indices[collection_id] = index
        self._metadata[collection_id] = meta
        self._save(collection_id)
        logger.info(f"Rebuilt FAISS index for {collection_id} with {len(all_chunks)} chunks.")

    def search(
        self,
        collection_id: str,
        query: str,
        top_k: int = 15,
        filter_paper_id: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Performs inner-product (cosine) search against the collection vector index.
        Returns top_k SearchResult objects with dense_score.
        """
        index, meta = self._load_or_create(collection_id)
        if index.ntotal == 0 or not meta:
            from database.db_manager import DatabaseManager
            db = DatabaseManager()
            all_chunks = db.get_all_chunks(collection_id)
            if all_chunks:
                self.index_chunks(collection_id, all_chunks)
                index, meta = self._load_or_create(collection_id)
            else:
                return []

        query_vec = self.embedder.embed_query(query)

        # Retrieve a broader pool to accommodate metadata filtering
        search_k = min(top_k * 3, index.ntotal)
        scores, indices = index.search(query_vec, search_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(meta):
                continue

            chunk = meta[idx]

            # Apply filters if specified
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
                    dense_score=float(score),
                )
            )

            if len(results) >= top_k:
                break

        return results
