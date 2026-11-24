"""
Tests for modules/results_tables.py.
Covers defensive filtering, candidate evaluation, classification taxonomy,
caption extraction, results normalization, database integration, and edge cases.
"""

import json

import pymupdf

from database.db_manager import DatabaseManager
from modules.results_tables import (
    _matches_term,
    classify_table,
    evaluate_table_candidate,
    extract_table_caption,
    extract_tables_from_pdf,
    get_paper_results_tables,
    is_likely_results_table,
    is_valid_table,
    normalize_results_records,
    normalize_table_dict,
    parse_table_semantics,
)

# ---------------------------------------------------------------------------
# 1. Candidate Filtering & Threshold Tests
# ---------------------------------------------------------------------------

def test_evaluate_candidate_real_results_table():
    """Test candidate evaluation accepts a well-formed results table."""
    headers = ["Model", "Accuracy (%)", "F1 Score", "Latency (ms)"]
    rows = [
        ["BERT-Base", "84.2", "83.9", "12.4"],
        ["RoBERTa-Large", "88.7", "88.5", "24.1"],
        ["Ours (ResearchLens)", "91.3", "91.0", "15.2"],
    ]
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is True
    assert reason is None


def test_evaluate_candidate_empty_table():
    """Test candidate evaluation rejects empty tables or 0 data rows."""
    # Completely empty
    accepted, reason = evaluate_table_candidate([], [])
    assert accepted is False
    assert "Fewer than" in reason

    # Empty rows
    accepted, reason = evaluate_table_candidate(["Col1", "Col2"], [])
    assert accepted is False
    assert "Fewer than" in reason

    # Grid with only empty strings
    accepted, reason = evaluate_table_candidate(["", ""], [["", ""], ["  ", ""]])
    assert accepted is False
    assert "non-empty rows" in reason


def test_evaluate_candidate_one_column_list():
    """Test candidate evaluation rejects 1-column bullet lists or enumerations."""
    headers = ["Features"]
    rows = [
        ["1. Cross-encoder reranking"],
        ["2. Dense retrieval"],
        ["3. Query expansion"],
    ]
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "Fewer than 2 columns" in reason


def test_evaluate_candidate_ragged_merged_cells():
    """Test candidate evaluation rejects wildly ragged/corrupted grids."""
    headers = ["Col1", "Col2"]
    rows = [
        ["A", "B"],
        ["A", "B", "C", "D", "E", "F", "G"],  # 7 columns vs 2 columns (difference > 3)
    ]
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "Inconsistent column count" in reason


def test_evaluate_candidate_excessive_empty_cells():
    """Test candidate evaluation rejects sparse grids exceeding MAX_EMPTY_CELL_RATIO."""
    headers = ["H1", "H2", "H3", "H4"]
    rows = [
        ["", "", "", ""],
        ["", "Val", "", ""],
        ["", "", "", ""],
    ]
    # Total cells = 16. Empty cells = 15 -> 93.7% empty > 60%
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "Excessive empty cell ratio" in reason


def test_evaluate_candidate_paragraph_misdetected_as_table():
    """Test candidate evaluation rejects narrative paragraphs misdetected by layout engines."""
    headers = ["Introduction", "Background"]
    rows = [
        [
            "In recent years, large language models have demonstrated remarkable capabilities across diverse tasks.",
            "However, multi-hop question answering remains a significant challenge requiring evidence aggregation.",
        ],
        [
            "To address this issue, researchers have proposed iterative retrieval and chain-of-thought prompting.",
            "Our work introduces a novel grounding framework that verifies citations against primary source pages.",
        ],
    ]
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "narrative" in reason.lower() or "paragraph" in reason.lower()


def test_evaluate_candidate_single_long_prose_sentences():
    """Test rejection when cells contain long narrative sentences (> 220 chars)."""
    headers = ["Section", "Summary"]
    long_sentence = (
        "We systematically examine the trade-offs between precision and recall across four distinct academic domains, "
        "demonstrating that our proposed hybrid reranker consistently outperforms competitive dense baselines while "
        "maintaining computational efficiency under strict latency constraints."
    )
    long_sentence_2 = (
        "Furthermore, our extensive ablation studies verify that each individual architectural component provides "
        "statistically significant improvements over naive lexical search across all evaluated benchmarks."
    )
    rows = [
        ["Methodology", long_sentence],
        ["Discussion", long_sentence_2],
    ]
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "narrative prose sentence" in reason.lower() or "paragraph" in reason.lower()


def test_evaluate_candidate_inside_references_section():
    """Test candidate evaluation rejects tables extracted within references or works cited."""
    headers = ["Index", "Citation"]
    rows = [
        ["[1]", "Vaswani et al. Attention is all you need. NeurIPS 2017."],
        ["[2]", "Devlin et al. BERT: Pre-training of deep bidirectional transformers. NAACL 2019."],
        ["[3]", "Brown et al. Language models are few-shot learners. NeurIPS 2020."],
    ]

    # Via section name
    accepted, reason = evaluate_table_candidate(headers, rows, section_name="8. References")
    assert accepted is False
    assert "references" in reason.lower()

    # Via surrounding text
    accepted, reason = evaluate_table_candidate(headers, rows, surrounding_text="References\n[1] ...")
    assert accepted is False
    assert "references" in reason.lower()

    # Via regex detection in cells
    accepted, reason = evaluate_table_candidate(headers, rows)
    assert accepted is False
    assert "references" in reason.lower()


def test_evaluate_candidate_vertical_coverage_threshold():
    """Test candidate evaluation rejects tables covering >85% of vertical page height."""
    headers = ["Header A", "Header B"]
    rows = [["Row 1A", "Row 1B"], ["Row 2A", "Row 2B"]]
    # Page height = 800. Table bbox y0=50, y1=750 -> height=700 (87.5% coverage)
    accepted, reason = evaluate_table_candidate(
        headers, rows, bbox=(50, 50, 550, 750), page_height=800.0
    )
    assert accepted is False
    assert "vertical coverage exceeds" in reason


# ---------------------------------------------------------------------------
# 2. Table Classification Tests
# ---------------------------------------------------------------------------

def test_classify_table_results():
    """Test classification of empirical results / benchmark tables."""
    headers = ["Method", "BLEU-4", "ROUGE-L", "Exact Match"]
    rows = [
        ["Baseline GNN", "24.5", "38.2", "45.1"],
        ["Ours + Rerank", "29.8", "43.1", "52.4"],
    ]
    caption = "Table 2: Performance comparison on HotpotQA benchmark"
    cl = classify_table(headers, rows, caption)
    assert cl == "results"


def test_classify_table_hyperparameters():
    """Test classification of hyperparameter and training settings tables."""
    headers = ["Parameter", "Search Space", "Optimal Value"]
    rows = [
        ["Learning Rate", "[1e-5, 5e-5]", "3e-5"],
        ["Batch Size", "[16, 32, 64]", "32"],
        ["Warmup Ratio", "0.05 - 0.1", "0.1"],
        ["Optimizer", "AdamW", "AdamW"],
    ]
    caption = "Table 4: Training hyperparameters and configuration"
    cl = classify_table(headers, rows, caption)
    assert cl == "hyperparameters"


def test_classify_table_dataset_statistics():
    """Test classification of dataset and corpus statistics tables."""
    headers = ["Dataset", "# Train", "# Dev", "# Test", "Avg Tokens"]
    rows = [
        ["HotpotQA", "90,447", "7,405", "7,405", "420"],
        ["2WikiMQA", "167,454", "12,576", "12,576", "512"],
    ]
    caption = "Table 1: Statistics of experimental data and benchmark splits"
    cl = classify_table(headers, rows, caption)
    assert cl == "dataset_statistics"


def test_classify_table_other():
    """Test classification of qualitative / model template / architecture tables."""
    headers = ["Module", "Component Description"]
    rows = [
        ["Encoder", "Pre-trained Transformer backbone with cross-attention"],
        ["Fusion Layer", "Bilinear pooling with residual connection"],
    ]
    caption = "Table 3: Structural overview of network modules"
    cl = classify_table(headers, rows, caption)
    assert cl == "other"


# ---------------------------------------------------------------------------
# 3. Results Normalization Tests
# ---------------------------------------------------------------------------

def test_normalize_results_records_standard():
    """Test normalizing a standard table with model rows and metric columns."""
    headers = ["Model", "Precision", "Recall", "F1"]
    rows = [
        ["BERT", "0.85", "0.82", "0.83"],
        ["RoBERTa", "0.89", "0.86", "0.87"],
    ]
    res = normalize_results_records(headers, rows, page_number=5, caption="Table 1: RetrievalQA test")
    assert res["reliable"] is True
    records = res["records"]
    assert len(records) == 6  # 2 models * 3 metrics

    bert_f1 = next(r for r in records if r["model"] == "BERT" and r["metric"] == "F1")
    assert bert_f1["value"] == "0.83"
    assert bert_f1["dataset"] == "RetrievalQA"
    assert bert_f1["page"] == 5


def test_normalize_results_records_multidimensional_benchmarks():
    """Test normalizing a table where columns represent benchmark datasets."""
    headers = ["System", "HotpotQA", "2WikiMQA", "MusiQue"]
    rows = [
        ["Standard RAG", "42.1", "38.5", "29.4"],
        ["LongRAG", "55.3", "51.8", "40.2"],
    ]
    res = normalize_results_records(headers, rows, page_number=6, caption="Table 2: Main evaluation results (%)")
    assert res["reliable"] is True
    records = res["records"]
    assert len(records) == 6

    longrag_hotpot = next(r for r in records if r["model"] == "LongRAG" and r["dataset"] == "HotpotQA")
    assert longrag_hotpot["value"] == "55.3"
    assert longrag_hotpot["metric"] == "Score (%)"


def test_normalize_results_records_multiline_squashed():
    """Test normalizing tables with newline-squashed multi-row entries."""
    headers = ["Model", "Acc", "F1"]
    rows = [
        ["Model A\nModel B", "75.0\n82.0", "74.2\n81.5"],
    ]
    res = normalize_results_records(headers, rows, page_number=3, caption="Results on SQuAD")
    assert res["reliable"] is True
    records = res["records"]
    assert len(records) == 4

    ma_acc = next(r for r in records if r["model"] == "Model A" and r["metric"] == "Acc")
    assert ma_acc["value"] == "75.0"
    mb_f1 = next(r for r in records if r["model"] == "Model B" and r["metric"] == "F1")
    assert mb_f1["value"] == "81.5"


def test_normalize_results_records_unreliable_fallback():
    """Test fallback when rows have unalignable squashed entries."""
    headers = ["Model", "F1"]
    rows = [
        ["Model A\nModel B\nModel C", "75.0\n82.0"],  # 3 models vs 2 values -> mismatch
    ]
    res = normalize_results_records(headers, rows, page_number=3)
    assert res["reliable"] is False
    assert "Irregular row/column alignment" in res["note"]


def test_normalize_results_records_no_metric_or_dataset_cols():
    """Test fallback when columns cannot be recognized as metrics or datasets."""
    headers = ["Author", "Institution"]
    rows = [["Alice", "MIT"], ["Bob", "Stanford"]]
    res = normalize_results_records(headers, rows, page_number=1)
    assert res is not None

    empty_res = normalize_results_records([], [])
    assert empty_res["reliable"] is False


# ---------------------------------------------------------------------------
# 4. Helper Functions & Table Semantics Tests
# ---------------------------------------------------------------------------

def test_is_valid_and_is_likely_results():
    """Test legacy compatibility helpers is_valid_table and is_likely_results_table."""
    valid_headers = ["Method", "Accuracy"]
    valid_rows = [["M1", "90.5"], ["M2", "92.1"]]
    assert is_valid_table(valid_headers, valid_rows) is True
    assert is_likely_results_table(valid_headers, valid_rows) is True

    # Single cell layout artifact
    assert is_valid_table(["Single"], [["Item"]]) is False
    assert is_likely_results_table(["Single"], [["Item"]]) is False


def test_parse_table_semantics():
    """Test parsing semantics (models, metrics, benchmarks) from table headers."""
    headers = ["Architecture", "BLEU-4", "ROUGE-2", "Loss"]
    rows = [["Transformer", "32.1", "15.4", "1.23"]]
    caption = "Performance on GLUE and SQuAD benchmarks"
    sem = parse_table_semantics(headers, rows, caption)
    assert sem["model_column"]["name"] == "Architecture"
    assert len(sem["metric_columns"]) == 3
    assert "GLUE" in sem["detected_benchmarks"]
    assert "SQUAD" in sem["detected_benchmarks"]


def test_term_boundary_matching():
    """Test word-boundary regex helper to avoid partial substring false positives."""
    assert _matches_term("em", "Exact match (EM)") is True
    assert _matches_term("em", "email address") is False
    assert _matches_term("acc", "Accuracy (acc)") is True
    assert _matches_term("acc", "accepted without changes") is False


# ---------------------------------------------------------------------------
# 5. Caption Extraction & Document Processing Tests
# ---------------------------------------------------------------------------

def test_extract_table_caption_synthetic_pdf():
    """Test caption extraction above and below tables using a synthetic PyMuPDF document."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    # Insert text above
    page.insert_text((50, 100), "Table 1: Main experimental results on benchmark datasets.")
    page.insert_text((50, 115), "Evaluated under standard 5-fold cross validation.")
    # Table bbox: (50, 130, 500, 250)
    bbox = (50, 130, 500, 250)

    caption = extract_table_caption(page, bbox)
    assert "Table 1: Main experimental results on benchmark datasets." in caption
    assert "cross validation" in caption
    doc.close()


def test_normalize_table_dict_full_lifecycle():
    """Test normalize_table_dict with all extended metadata fields."""
    raw = {
        "page_number": "2",
        "table_index": "1",
        "headers": ["Model", "F1 Score"],
        "rows": [["M1", "88.2"], ["M2", "91.4"]],
        "caption": "Table 1: Benchmark Results",
        "paper_id": "pid_test",
        "id": 42,
    }
    normalized = normalize_table_dict(raw)
    assert normalized["page_number"] == 2
    assert normalized["table_index"] == 1
    assert normalized["row_count"] == 2
    assert normalized["col_count"] == 2
    assert normalized["is_accepted"] is True
    assert normalized["classification"] == "results"
    assert normalized["is_results_table"] is True
    assert len(normalized["normalized_records"]) == 2
    assert normalized["id"] == 42
    assert normalized["paper_id"] == "pid_test"


# ---------------------------------------------------------------------------
# 6. Database Integration & Pipeline Tests
# ---------------------------------------------------------------------------

def test_extract_tables_from_pdf_and_db_persistence(tmp_path):
    """Test full PDF extraction with synthetic document and DB persistence."""
    # Create synthetic PDF
    pdf_path = str(tmp_path / "synthetic_paper.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    # Add heading and table caption
    page.insert_text((50, 80), "4. Experimental Results", fontsize=14)
    page.insert_text((50, 110), "Table 1: Evaluation Metrics on Target Task", fontsize=11)

    # Draw table lines for find_tables
    # Headers
    page.draw_rect(pymupdf.Rect(50, 130, 200, 160))
    page.insert_text((60, 150), "Model")
    page.draw_rect(pymupdf.Rect(200, 130, 350, 160))
    page.insert_text((210, 150), "Accuracy (%)")
    page.draw_rect(pymupdf.Rect(350, 130, 500, 160))
    page.insert_text((360, 150), "F1 Score")

    # Row 1
    page.draw_rect(pymupdf.Rect(50, 160, 200, 190))
    page.insert_text((60, 180), "Baseline-1")
    page.draw_rect(pymupdf.Rect(200, 160, 350, 190))
    page.insert_text((210, 180), "82.4")
    page.draw_rect(pymupdf.Rect(350, 160, 500, 190))
    page.insert_text((360, 180), "81.9")

    # Row 2
    page.draw_rect(pymupdf.Rect(50, 190, 200, 220))
    page.insert_text((60, 210), "ResearchLens")
    page.draw_rect(pymupdf.Rect(200, 190, 350, 220))
    page.insert_text((210, 210), "94.6")
    page.draw_rect(pymupdf.Rect(350, 190, 500, 220))
    page.insert_text((360, 210), "94.1")

    doc.save(pdf_path)
    doc.close()

    # Initialize in-memory / temporary DB
    db_path = str(tmp_path / "test_tables.db")
    db = DatabaseManager(db_path=db_path)

    # Insert paper to fulfill foreign key requirement
    paper_id = db.insert_paper(
        collection_id="default-collection",
        file_hash="hash_synth_123",
        filename="synthetic_paper.pdf",
        file_path=pdf_path,
        title="Synthetic Benchmark Paper",
        n_pages=1,
    )

    # Non-existent PDF test
    assert extract_tables_from_pdf("non_existent_file.pdf", paper_id) == []

    # Extract tables
    extracted = extract_tables_from_pdf(pdf_path, paper_id, db=db)
    assert len(extracted) >= 1
    t = extracted[0]
    assert t["is_accepted"] is True
    assert t["classification"] == "results"
    assert t["row_count"] == 2
    assert t["col_count"] >= 3

    # Verify retrieval from database
    stored = get_paper_results_tables(paper_id, db)
    assert len(stored) == len(extracted)
    s = stored[0]
    assert len(s["headers"]) >= 3
    assert s["row_count"] == 2
    assert s["col_count"] >= 3
    assert s["is_results_table"] is True


def test_get_paper_results_tables_corrupt_data(tmp_path):
    """Test get_paper_results_tables gracefully recovers from corrupt JSON entries."""
    db_path = str(tmp_path / "corrupt_tables.db")
    db = DatabaseManager(db_path=db_path)

    # Insert paper to fulfill foreign key constraint
    paper_id = db.insert_paper(
        collection_id="default-collection",
        file_hash="hash_corrupt_123",
        filename="corrupt.pdf",
        file_path="/tmp/corrupt.pdf",
        title="Corrupt Test Paper",
        n_pages=2,
    )

    # Insert valid table and corrupt record
    db.insert_results_table(paper_id, page_number=1, table_index=1, table_json="NOT_VALID_JSON")
    db.insert_results_table(
        paper_id,
        page_number=2,
        table_index=2,
        table_json=json.dumps({"headers": ["A", "B"], "rows": [["1", "2"]]}),
    )

    tables = get_paper_results_tables(paper_id, db)
    # The malformed JSON string is caught and skipped; the valid one is normalized
    assert len(tables) == 1
    assert tables[0]["page_number"] == 2
    assert tables[0]["col_count"] == 2
