"""
Tests for Phase 5: RAG Question Answering, Inline Citations, and Claim Verification.
"""

import os
import tempfile

import pytest

from core.schemas import SearchResult
from database.db_manager import DatabaseManager
from modules.citation_verifier import (
    clean_claim_text_without_citations,
    compute_support_score,
    extract_citations_from_claim,
    split_claims,
    verify_claims,
)
from modules.rag_pipeline import (
    answer_research_question,
    format_grounding_documents,
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = DatabaseManager(db_path=db_path)
    db.init_db()
    yield db
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_split_claims():
    text = (
        "ClinicalBERT adapts Transformer architectures for clinical texts [Paper 1, p. 1]. "
        "It achieves 88.4% Macro F1 on the MIMIC-III benchmark [Paper 1, p. 2]. "
        "However, high computational training costs remain a limitation [Paper 1, p. 2]."
    )
    claims = split_claims(text)
    assert len(claims) == 3
    assert "ClinicalBERT adapts" in claims[0]
    assert "[Paper 1, p. 1]" in claims[0]
    assert "[Paper 1, p. 2]" in claims[1]


def test_extract_citations():
    # Multi-format citations
    claim1 = "The model attains 94.2% accuracy on benchmark tests [Paper 1, p. 5]."
    cits1 = extract_citations_from_claim(claim1)
    assert len(cits1) == 1
    assert cits1[0]["paper_num"] == 1
    assert cits1[0]["page_num"] == 5

    claim2 = "Graph neural networks model citation topology effectively [Paper 2, pp. 3-4]."
    cits2 = extract_citations_from_claim(claim2)
    assert len(cits2) == 1
    assert cits2[0]["paper_num"] == 2
    assert cits2[0]["page_num"] == 3

    claim3 = "Page-only reference [p. 8]."
    cits3 = extract_citations_from_claim(claim3)
    assert len(cits3) == 1
    assert cits3[0]["page_num"] == 8

    # Clean claim text
    cleaned = clean_claim_text_without_citations(claim1)
    assert cleaned == "The model attains 94.2% accuracy on benchmark tests."


def test_claim_verification_engine():
    dummy_chunks = [
        SearchResult(
            chunk_id="c1",
            paper_id="pid_1",
            paper_title="ClinicalBERT Research",
            section_name="Results",
            page_start=2,
            page_end=2,
            text="ClinicalBERT achieves 88.4% Macro F1 on the MIMIC-III benchmark dataset.",
        ),
        SearchResult(
            chunk_id="c2",
            paper_id="pid_2",
            paper_title="Graph Neural Networks",
            section_name="Methodology",
            page_start=3,
            page_end=3,
            text="We utilize a 2-layer Graph Convolutional Network for citation propagation.",
        ),
    ]

    paper_map = {1: "pid_1", 2: "pid_2"}

    # 1. Grounded claim with matching citation
    grounded_claim = "ClinicalBERT achieves 88.4% Macro F1 on the MIMIC-III benchmark [Paper 1, p. 2]."
    verifications = verify_claims([grounded_claim], dummy_chunks, paper_map)
    assert len(verifications) == 1
    assert verifications[0].is_supported is True
    assert verifications[0].support_level == "supported"

    # 2. Fabricated claim with mismatched citation
    hallucinated_claim = "The model processes audio waveforms using Whisper [Paper 1, p. 2]."
    verifications_hallucinated = verify_claims([hallucinated_claim], dummy_chunks, paper_map)
    assert len(verifications_hallucinated) == 1
    assert verifications_hallucinated[0].is_supported is False
    assert verifications_hallucinated[0].support_level == "unsupported"

    # 3. Claim with no citation
    uncited_claim = "Transformers have revolutionized modern artificial intelligence."
    verifications_uncited = verify_claims([uncited_claim], dummy_chunks, paper_map)
    assert len(verifications_uncited) == 1
    assert verifications_uncited[0].is_supported is False
    assert verifications_uncited[0].support_level == "unsupported"

    # Overall score calculation
    score = compute_support_score(verifications + verifications_uncited)
    assert score == 0.5


def test_format_grounding_documents():
    chunks = [
        SearchResult(
            chunk_id="c1",
            paper_id="pid_1",
            paper_title="Paper Alpha",
            section_name="Abstract",
            page_start=1,
            page_end=1,
            text="Alpha excerpt.",
        ),
        SearchResult(
            chunk_id="c2",
            paper_id="pid_2",
            paper_title="Paper Beta",
            section_name="Results",
            page_start=4,
            page_end=4,
            text="Beta excerpt.",
        ),
    ]
    xml, p_map, citations = format_grounding_documents(chunks)
    assert "<documents>" in xml
    assert "</documents>" in xml
    assert 'paper="Paper 1: Paper Alpha"' in xml
    assert 'paper="Paper 2: Paper Beta"' in xml
    assert p_map[1] == "pid_1"
    assert p_map[2] == "pid_2"
    assert len(citations) == 2


def test_rag_insufficient_evidence(temp_db):
    col_id = temp_db.create_collection("Empty RAG Test")
    response = answer_research_question(
        query="What is the quantum computing advantage?",
        collection_id=col_id,
        db=temp_db,
    )
    assert response.insufficient_evidence is True
    assert "do not contain sufficient evidence" in response.answer.lower()
