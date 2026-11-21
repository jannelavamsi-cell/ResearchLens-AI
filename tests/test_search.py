"""
Unit tests for Keyword extraction, FAISS vector indexing, BM25, RRF fusion,
and Cross-Encoder reranking in ResearchLens AI.
"""

import tempfile

import pytest

from core.schemas import SearchResult
from modules.bm25_search import BM25SearchEngine
from modules.hybrid_search import compute_rrf_scores
from modules.keyword_extraction import KeywordExtractor
from modules.semantic_search import SemanticSearchEngine


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "chk-1",
            "paper_id": "paper-1",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_name": "Abstract",
            "text": "MedLens-Transformer introduces multi-head self-attention for clinical medical records and EHR document intelligence.",
            "paper_title": "Medical NLP Paper",
        },
        {
            "id": "chk-2",
            "paper_id": "paper-2",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_name": "Abstract",
            "text": "CiteGraph-GNN is a relational graph neural network predicting citation links in academic bibliographies.",
            "paper_title": "Citation Graph Paper",
        },
        {
            "id": "chk-3",
            "paper_id": "paper-3",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_name": "Methodology",
            "text": "HybridLens combines Okapi BM25 and dense bi-encoder embeddings using Reciprocal Rank Fusion.",
            "paper_title": "Hybrid Search Paper",
        },
    ]


def test_rrf_fusion_mathematics():
    """Validates that RRF correctly applies formula: 1 / (60 + rank)."""
    item1 = SearchResult(
        chunk_id="chk-1",
        paper_id="p1",
        paper_title="T1",
        section_name="Abstract",
        page_start=1,
        page_end=1,
        text="Sample text 1",
        dense_score=0.9,
    )
    item2 = SearchResult(
        chunk_id="chk-2",
        paper_id="p2",
        paper_title="T2",
        section_name="Methodology",
        page_start=2,
        page_end=2,
        text="Sample text 2",
        dense_score=0.8,
    )

    list1 = [item1, item2]  # item1 rank 1, item2 rank 2
    list2 = [item2, item1]  # item2 rank 1, item1 rank 2

    # RRF with k=60
    # item1 score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    # item2 score = 1/(60+2) + 1/(60+1) = 0.032522
    fused = compute_rrf_scores([list1, list2], k=60)
    assert len(fused) == 2
    assert abs(fused[0].rrf_score - 0.032522) < 0.0001
    assert abs(fused[1].rrf_score - 0.032522) < 0.0001


def test_bm25_search(sample_chunks):
    """Tests BM25 indexing and lexical query scoring."""
    bm25 = BM25SearchEngine()
    bm25.index_chunks("test-collection", sample_chunks)

    # Search for medical attention
    results = bm25.search("test-collection", "clinical medical attention", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "chk-1"
    assert results[0].bm25_score > 0.0

    # Search with section filter
    res_filtered = bm25.search(
        "test-collection", "embeddings", top_k=2, filter_section="Methodology"
    )
    assert len(res_filtered) == 1
    assert res_filtered[0].chunk_id == "chk-3"


def test_faiss_semantic_search(sample_chunks):
    """Tests FAISS IndexFlatIP indexing, disk persistence, and vector retrieval."""
    temp_dir = tempfile.mkdtemp()
    engine = SemanticSearchEngine(index_dir=temp_dir)

    engine.index_chunks("test-faiss", sample_chunks)

    # Search for graph neural network
    res = engine.search("test-faiss", "graph neural network citation links", top_k=2)
    assert len(res) > 0
    assert res[0].chunk_id == "chk-2"
    assert res[0].dense_score is not None

    # Test reloading index from disk
    engine2 = SemanticSearchEngine(index_dir=temp_dir)
    res_reloaded = engine2.search("test-faiss", "medical clinical records", top_k=2)
    assert len(res_reloaded) > 0
    assert res_reloaded[0].chunk_id == "chk-1"


def test_keyword_extraction():
    """Tests keyword extraction merging KeyBERT, TF-IDF, and spaCy terms."""
    text = (
        "MedLens-Transformer introduces a multi-head self-attention mechanism tailored for clinical "
        "electronic health records and medical document intelligence. The model outperforms ClinicalBERT."
    )
    extractor = KeywordExtractor(top_n=5)
    keywords = extractor.extract_paper_keywords(text)

    assert len(keywords) > 0
    kw_names = [k["keyword"] for k in keywords]
    assert any("attention" in k or "clinical" in k or "medlens" in k for k in kw_names)
