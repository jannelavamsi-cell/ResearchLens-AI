"""
Tests for Phase 7: Literature Review Synthesis, Thematic Clustering, and Research Gaps.
"""

import os
import tempfile

import pytest

from core.schemas import ExtractedField, PageSchema
from database.db_manager import DatabaseManager
from modules.literature_review import (
    cluster_papers_thematically,
    review_to_markdown,
    synthesize_literature_review,
)
from modules.references import (
    format_apa_reference,
    format_bibtex_entry,
    format_ieee_reference,
    generate_references_list,
    parse_authors_list,
)
from modules.research_gap import discover_research_gaps


@pytest.fixture
def test_env():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = DatabaseManager(db_path=db_path)
    db.init_db()

    col_id = db.create_collection("Review Test Collection")

    # Paper 1
    p1_id = db.insert_paper(
        collection_id=col_id,
        file_hash="h1",
        filename="p1.pdf",
        file_path="p1.pdf",
        title="ClinicalBERT: Medical Language Modeling",
        n_pages=2,
    )
    p1_text = "ClinicalBERT applies transformers to electronic medical records."
    db.insert_pages(p1_id, [PageSchema(paper_id=p1_id, page_number=1, text=p1_text, is_ocr=False)])
    db.insert_chunks([{"id": "c1_1", "paper_id": p1_id, "chunk_index": 0, "page_start": 1, "page_end": 1, "section_name": "Abstract", "text": p1_text, "token_count": 15}])
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="objective", value="Clinical NLP", evidence_quote="clinical texts", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="methodology", value="Transformer Encoder", evidence_quote="transformers", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="results", value="88.4% Macro F1", evidence_quote="88.4%", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="limitations", value="High computational GPU cost", evidence_quote="computational cost", page=2, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="future_work", value="Quantization for edge", evidence_quote="quantization", page=2, confidence=0.9, verified=True))

    # Paper 2
    p2_id = db.insert_paper(
        collection_id=col_id,
        file_hash="h2",
        filename="p2.pdf",
        file_path="p2.pdf",
        title="CitationGNN: Graph Representation Learning",
        n_pages=2,
    )
    p2_text = "CitationGNN models bibliographic networks using graph convolutions."
    db.insert_pages(p2_id, [PageSchema(paper_id=p2_id, page_number=1, text=p2_text, is_ocr=False)])
    db.insert_chunks([{"id": "c2_1", "paper_id": p2_id, "chunk_index": 0, "page_start": 1, "page_end": 1, "section_name": "Abstract", "text": p2_text, "token_count": 15}])
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="objective", value="Graph Learning", evidence_quote="graph convolutions", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="methodology", value="2-layer GCN", evidence_quote="GCN", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="results", value="84.1% Accuracy", evidence_quote="84.1%", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="limitations", value="High memory overhead on large graphs", evidence_quote="memory overhead", page=2, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="future_work", value="Distributed neighbor sampling", evidence_quote="distributed", page=2, confidence=0.9, verified=True))

    yield db, col_id, [p1_id, p2_id]

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_reference_formatters():
    paper = {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani and Noam Shazeer and Niki Parmar",
        "year": 2017,
        "venue": "NeurIPS",
        "doi": "10.5555/3295222.3295349",
    }

    # Author parser
    authors = parse_authors_list(paper["authors"])
    assert len(authors) == 3
    assert authors[0] == "Ashish Vaswani"

    # IEEE
    ieee = format_ieee_reference(paper, index=1)
    assert "[1]" in ieee
    assert "A. Vaswani" in ieee
    assert '"Attention Is All You Need,"' in ieee
    assert "NeurIPS" in ieee

    # APA
    apa = format_apa_reference(paper)
    assert "Vaswani, A." in apa
    assert "(2017)" in apa
    assert "https://doi.org/" in apa

    # BibTeX
    bib = format_bibtex_entry(paper)
    assert "@article{" in bib
    assert "title     = {Attention Is All You Need}" in bib
    assert "year      = {2017}" in bib

    # Multi-style list
    refs = generate_references_list([paper], style="ieee")
    assert len(refs) == 1


def test_thematic_clustering():
    papers = [
        {"title": "ClinicalBERT for Clinical Records", "abstract": "Transformers on patient EHR notes."},
        {"title": "BioBERT for Biomedical Extraction", "abstract": "Pre-trained biomedical language model."},
        {"title": "GCN for Social Networks", "abstract": "Graph convolutional networks for node classification."},
    ]
    clusters = cluster_papers_thematically(papers, n_themes=2)
    assert len(clusters) >= 1


def test_synthesize_literature_review(test_env):
    db, col_id, pids = test_env

    draft = synthesize_literature_review(collection_id=col_id, paper_ids=pids, db=db)
    assert draft.title == "Comparative Academic Literature Review"
    assert "AI-generated draft" in draft.disclaimer
    assert len(draft.introduction) > 10
    assert len(draft.methodological_comparison) > 10
    assert len(draft.references) == 2

    # Render markdown
    md = review_to_markdown(draft)
    assert "# Comparative Academic Literature Review" in md
    assert "Mandatory Academic Notice" in md
    assert "## References" in md


def test_discover_research_gaps(test_env):
    db, col_id, pids = test_env

    gaps = discover_research_gaps(collection_id=col_id, paper_ids=pids, db=db)
    assert len(gaps) >= 1
    # Both papers declared scalability/computational cost limitations!
    scalability_gap = next((g for g in gaps if "Scalability" in g.gap_statement), None)
    assert scalability_gap is not None
    assert scalability_gap.frequency_count == 2
    assert len(scalability_gap.supporting_papers) == 2
    assert scalability_gap.suggested_direction is not None
