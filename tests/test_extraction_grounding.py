"""
Tests for Phase 4: Structured Research Extraction and Evidence Grounding Verification.
"""

import os
import tempfile

import pytest

from core.schemas import ExtractedField, PageSchema
from database.db_manager import DatabaseManager
from modules.research_extraction import (
    CORE_FIELDS,
    extract_paper_components,
    verify_quote_against_page,
)
from modules.results_tables import is_likely_results_table, normalize_table_dict


@pytest.fixture
def test_db():
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


def test_quote_verification_logic():
    page_text = (
        "We present ClinicalBERT, an adaptation of Transformer architecture "
        "trained on de-identified MIMIC-III clinical records. Our model achieves "
        "a Macro F1-score of 88.4% on diagnostic code classification, outperforming "
        "standard BERT by 4.2%."
    )

    # 1. Exact match
    exact_quote = "trained on de-identified MIMIC-III clinical records"
    assert verify_quote_against_page(exact_quote, page_text) is True

    # 2. Minor whitespace / casing difference
    fuzzy_quote = "Trained On De-Identified   MIMIC-III clinical records"
    assert verify_quote_against_page(fuzzy_quote, page_text) is True

    # 3. Hallucinated / absent quote
    hallucinated_quote = "We evaluate our framework on 500,000 chest X-ray radiographs with ResNet-50."
    assert verify_quote_against_page(hallucinated_quote, page_text) is False

    # 4. Empty quote or empty text
    assert verify_quote_against_page("", page_text) is False
    assert verify_quote_against_page(exact_quote, "") is False


def test_is_likely_results_table():
    headers = ["Model", "Accuracy (%)", "Macro F1", "Parameters (M)"]
    rows = [["Baseline BERT", "84.2", "81.0", "110"], ["ClinicalBERT (Ours)", "89.5", "88.4", "110"]]
    assert is_likely_results_table(headers, rows) is True

    # Non-results table (e.g. author affiliations)
    layout_headers = ["Name", "Affiliation", "Email"]
    layout_rows = [["Alice Smith", "Stanford NLP", "alice@cs.stanford.edu"]]
    assert is_likely_results_table(layout_headers, layout_rows) is False


def test_rule_based_extraction_and_grounding(test_db):
    col_id = test_db.create_collection("Extraction Test Collection")
    paper_id = test_db.insert_paper(
        collection_id=col_id,
        file_hash="test_hash_grounding_1",
        filename="medical_nlp.pdf",
        file_path="dummy.pdf",
        title="ClinicalBERT: Medical NLP",
        n_pages=2,
    )

    # Insert mock pages
    p1_text = (
        "Abstract: We propose ClinicalBERT for clinical entity extraction. "
        "Methodology: The architecture utilizes a 12-layer Transformer encoder. "
        "Experiments: Evaluated on the MIMIC-III dataset measuring Macro F1 score."
    )
    p2_text = (
        "Results: ClinicalBERT achieves 88.4% Macro F1 outperforming all baselines. "
        "Limitations: High computational cost of fine-tuning remains a constraint. "
        "Future Work: We plan to explore quantization and edge deployment."
    )
    test_db.insert_pages(paper_id, [
        PageSchema(paper_id=paper_id, page_number=1, text=p1_text, is_ocr=False),
        PageSchema(paper_id=paper_id, page_number=2, text=p2_text, is_ocr=False),
    ])

    # Insert mock chunks
    test_db.insert_chunks([
        {
            "id": "c1",
            "paper_id": paper_id,
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_name": "Abstract",
            "text": p1_text,
            "token_count": 40,
        },
        {
            "id": "c2",
            "paper_id": paper_id,
            "chunk_index": 1,
            "page_start": 2,
            "page_end": 2,
            "section_name": "Results",
            "text": p2_text,
            "token_count": 40,
        },
    ])

    extractions = extract_paper_components(paper_id, test_db)

    # Check all 9 fields exist
    assert len(extractions) == 9
    for field_name in CORE_FIELDS:
        assert field_name in extractions
        field = extractions[field_name]
        assert isinstance(field, ExtractedField)
        assert field.value is not None

    # Check verification status
    assert extractions["results"].verified is True
    assert extractions["limitations"].verified is True
    assert extractions["results"].page == 2


def test_user_override_persistence(test_db):
    col_id = test_db.create_collection("Override Test Collection")
    paper_id = test_db.insert_paper(
        collection_id=col_id,
        file_hash="test_hash_override_1",
        filename="test_paper.pdf",
        file_path="dummy.pdf",
        title="Test Paper",
        n_pages=1,
    )

    initial_field = ExtractedField(
        field_name="metrics",
        value="Accuracy only",
        evidence_quote="measured accuracy",
        page=1,
        confidence=0.7,
        source_section="Experiments",
        verified=False,
    )
    test_db.insert_or_update_extraction(paper_id, initial_field)

    # Researcher manual override
    test_db.update_user_override(
        paper_id=paper_id,
        field_name="metrics",
        new_value="Accuracy, Macro-F1, and AUROC",
        page_number=1,
    )

    records = test_db.get_extractions(paper_id)
    assert records["metrics"]["value"] == "Accuracy, Macro-F1, and AUROC"
    assert records["metrics"]["user_override"] is True
    assert records["metrics"]["verified"] is True


def test_normalize_table_dict_all_keys():
    """Test table normalizer when all standard keys are explicitly provided."""
    input_table = {
        "page_number": 4,
        "page": 4,
        "table_index": 2,
        "headers": ["Model", "BLEU", "ROUGE"],
        "rows": [["Model A", "25.4", "45.1"], ["Model B", "28.1", "49.0"]],
        "row_count": 2,
        "col_count": 3,
        "caption": "Evaluation on benchmark test set",
        "is_results_table": True,
        "paper_id": "test_pid_123",
    }
    normalized = normalize_table_dict(input_table)
    assert normalized["page_number"] == 4
    assert normalized["page"] == 4
    assert normalized["table_index"] == 2
    assert normalized["headers"] == ["Model", "BLEU", "ROUGE"]
    assert normalized["header"] == ["Model", "BLEU", "ROUGE"]
    assert normalized["row_count"] == 2
    assert normalized["col_count"] == 3
    assert normalized["caption"] == "Evaluation on benchmark test set"
    assert normalized["is_results_table"] is True
    assert normalized["paper_id"] == "test_pid_123"


def test_normalize_table_dict_missing_counts():
    """Test table normalizer computing row_count, col_count, and headers from legacy format."""
    input_table = {
        "page_number": 7,
        "table_index": 1,
        "header": ["Method", "Precision", "Recall", "F1"],
        "rows": [
            ["Ours", "0.92", "0.88", "0.90"],
            ["Baseline", "0.85", "0.82", "0.83"],
            ["Ablation", "0.87", "0.84", "0.85"],
        ],
    }
    normalized = normalize_table_dict(input_table)
    assert normalized["page_number"] == 7
    assert normalized["table_index"] == 1
    assert normalized["headers"] == ["Method", "Precision", "Recall", "F1"]
    assert normalized["row_count"] == 3
    assert normalized["col_count"] == 4
    assert normalized["caption"] == ""
    assert normalized["is_results_table"] is True


def test_normalize_table_dict_empty():
    """Test table normalizer with empty tables and empty rows."""
    input_table = {
        "page_number": 1,
        "table_index": 1,
        "headers": [],
        "rows": [],
    }
    normalized = normalize_table_dict(input_table)
    assert normalized["row_count"] == 0
    assert normalized["col_count"] == 0
    assert normalized["rows"] == []
    assert normalized["headers"] == []
    assert normalized["is_results_table"] is False


def test_normalize_table_dict_malformed():
    """Test table normalizer with malformed data (non-dict, invalid types, None)."""
    # 1. Non-dict inputs
    assert normalize_table_dict(None)["row_count"] == 0
    assert normalize_table_dict("random_string")["col_count"] == 0
    assert normalize_table_dict([1, 2, 3])["headers"] == []

    # 2. Corrupted inner fields
    corrupt = {
        "page_number": "not_an_int",
        "table_index": None,
        "headers": None,
        "rows": "not_a_list",
        "row_count": "invalid",
        "col_count": None,
    }
    normalized = normalize_table_dict(corrupt)
    assert normalized["page_number"] == 1
    assert normalized["table_index"] == 1
    assert normalized["row_count"] == 0
    assert normalized["col_count"] == 0
    assert normalized["rows"] == []
    assert normalized["headers"] == []
    assert normalized["caption"] == ""

