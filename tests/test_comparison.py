"""
Tests for Phase 6: Faithful Paper Comparison and Benchmark Comparability Guards.
"""

import os
import tempfile

import pytest

from core.schemas import ExtractedField, PageSchema
from database.db_manager import DatabaseManager
from modules.paper_comparison import (
    build_comparison_matrix,
    check_comparability,
    generate_comparison_narrative,
    matrix_to_dataframe,
    matrix_to_markdown,
)


@pytest.fixture
def comp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = DatabaseManager(db_path=db_path)
    db.init_db()

    # Create collection
    col_id = db.create_collection("Comparison Test Collection")

    # Paper 1: Medical NLP
    p1_id = db.insert_paper(
        collection_id=col_id,
        file_hash="hash_p1",
        filename="p1.pdf",
        file_path="p1.pdf",
        title="ClinicalBERT",
        n_pages=3,
    )
    p1_text = "ClinicalBERT on MIMIC-III measuring Macro-F1 achieves 88.4%."
    db.insert_pages(p1_id, [PageSchema(paper_id=p1_id, page_number=1, text=p1_text, is_ocr=False)])
    db.insert_chunks([{"id": "c1_p1", "paper_id": p1_id, "chunk_index": 0, "page_start": 1, "page_end": 1, "section_name": "Experiments", "text": p1_text, "token_count": 20}])
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="datasets", value="MIMIC-III clinical records", evidence_quote="MIMIC-III", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="metrics", value="Macro-F1 score", evidence_quote="Macro-F1", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p1_id, ExtractedField(field_name="results", value="88.4% Macro F1", evidence_quote="88.4%", page=1, confidence=0.9, verified=True))

    # Paper 2: Graph GNN
    p2_id = db.insert_paper(
        collection_id=col_id,
        file_hash="hash_p2",
        filename="p2.pdf",
        file_path="p2.pdf",
        title="CitationGNN",
        n_pages=3,
    )
    p2_text = "CitationGNN on CORA measuring Classification Accuracy achieves 84.1%."
    db.insert_pages(p2_id, [PageSchema(paper_id=p2_id, page_number=1, text=p2_text, is_ocr=False)])
    db.insert_chunks([{"id": "c1_p2", "paper_id": p2_id, "chunk_index": 0, "page_start": 1, "page_end": 1, "section_name": "Experiments", "text": p2_text, "token_count": 20}])
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="datasets", value="CORA citation graph", evidence_quote="CORA", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="metrics", value="Classification Accuracy", evidence_quote="Accuracy", page=1, confidence=0.9, verified=True))
    db.insert_or_update_extraction(p2_id, ExtractedField(field_name="results", value="84.1% Accuracy", evidence_quote="84.1%", page=1, confidence=0.9, verified=True))

    yield db, [p1_id, p2_id]

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_comparison_matrix_builder(comp_db):
    db, pids = comp_db

    # Test error with < 2 papers
    with pytest.raises(ValueError, match="at least 2"):
        build_comparison_matrix([pids[0]], db)

    # Test valid matrix
    matrix = build_comparison_matrix(pids, db)
    assert len(matrix["papers"]) == 2
    assert "objective" in matrix["rows"]
    assert "datasets" in matrix["rows"]
    assert len(matrix["rows"]["datasets"]) == 2


def test_comparability_warning_detection(comp_db):
    db, pids = comp_db
    matrix = build_comparison_matrix(pids, db)

    warning = check_comparability(matrix)
    # Datasets differ (MIMIC vs CORA) and metrics differ (F1 vs Accuracy)
    assert warning.has_warning is True
    assert len(warning.reasons) >= 1
    assert "Comparability Warning" in warning.message


def test_dataframe_and_markdown_exports(comp_db):
    db, pids = comp_db
    matrix = build_comparison_matrix(pids, db)
    warning = check_comparability(matrix)

    df = matrix_to_dataframe(matrix)
    assert df.shape[0] == 9  # 9 core fields
    assert df.shape[1] == 2  # 2 papers

    narrative = generate_comparison_narrative(matrix, warning)
    assert len(narrative) > 50
    assert "Comparability Notice" in narrative or "comparability" in narrative.lower()

    md_report = matrix_to_markdown(matrix, warning, narrative)
    assert "# ResearchLens AI: Comparative Paper Analysis" in md_report
    assert "Comparability Warning" in md_report


def test_comparability_warning_real_benchmarks():
    from core.schemas import ComparisonCell

    matrix = {
        "papers": [{"title": "Paper 1"}, {"title": "Paper 2"}],
        "rows": {
            "datasets": [
                ComparisonCell(
                    paper_id="1",
                    field_name="datasets",
                    value="RetrievalQA benchmark with 1271 questions",
                    evidence_quote="RetrievalQA benchmark",
                    page=1,
                ),
                ComparisonCell(
                    paper_id="2",
                    field_name="datasets",
                    value="HotpotQA multi-hop benchmark",
                    evidence_quote="HotpotQA multi-hop",
                    page=1,
                ),
            ],
            "metrics": [
                ComparisonCell(
                    paper_id="1",
                    field_name="metrics",
                    value="Accuracy and F1",
                    evidence_quote="Accuracy and F1",
                    page=1,
                ),
                ComparisonCell(
                    paper_id="2",
                    field_name="metrics",
                    value="Exact Match (EM)",
                    evidence_quote="Exact Match",
                    page=1,
                ),
            ],
        },
    }
    warning = check_comparability(matrix)
    assert warning.has_warning is True
    assert len(warning.reasons) >= 1
    assert "Comparability Warning" in warning.message
