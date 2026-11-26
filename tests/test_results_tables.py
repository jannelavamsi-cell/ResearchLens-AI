"""
Tests for modules/results_tables.py.
Covers defensive filtering, candidate evaluation, classification taxonomy,
caption extraction, results normalization, database integration, and edge cases.
"""

import json

import pymupdf

from database.db_manager import DatabaseManager
from modules.results_tables import (
    _is_data_token,
    _matches_term,
    classify_table,
    evaluate_table_candidate,
    extract_table_caption,
    extract_tables_from_pdf,
    get_paper_results_tables,
    is_likely_results_table,
    is_merged_table_candidate,
    is_valid_table,
    normalize_results_records,
    normalize_table_dict,
    parse_table_semantics,
    reconstruct_table_from_words,
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


def test_real_paper6_page4_table_values():
    """
    Regression test for real Paper 6 (2024.findings-acl.415) Page 4 Table 1.
    Verifies that real PDF values are preserved verbatim:
      - Llama-2 (7B): ['80.3', '26.1', '36.0']
      - Average gain: ['+14.9', '+6.7', '–']
    And verifies that complex multi-level / squashed tables strictly flag reliable=False.
    """
    headers = ["Baselines (1,271)", "Adaptive", "Retrieval", "Always Retrieval"]
    rows = [
        ["Self-RAG (7B) t = 0.25 t = 0.5 t = 0.75", "Retrieval Match Calibration-based 100.0 31.9 23.0 10.6 0.0 6.0", "", "Match 31.9"],
        ["t = None Vanilla §3", "0.4", "6.0 Model-based", ""],
        ["TinyLlama (1.1B) Phi-2 (2.7B)", "39.1 94.1", "14.7 35.0", "28.2 36.4"],
        ["Llama-2 (7B)", "80.3", "26.1", "36.0"],
        ["GPT-3.5", "49.3", "20.8", "38.2"],
        ["GPT-4* Ours TA-ARE §4", "67.6", "37.6", "46.0"],
        ["TinyLlama (1.1B) Phi-2 (2.7B) Llama-2 (7B) GPT-3.5 GPT-4*", "54.1(+15.0) 95.5(+1.4) 86.0(+5.7) 86.3(+37.0) 83.2(+15.6)", "19.0(+4.3) 36.0(+1.0) 30.7(+4.6) 35.8(+15.0) 46.4(+8.8)", "28.2 36.4 36.0 38.2 46.0"],
        ["Average gain", "+14.9", "+6.7", "–"],
    ]
    caption = "Table 1: Retrieval and match accuracy on RetrievalQA."

    # Assert ground truth cell values
    assert rows[3] == ["Llama-2 (7B)", "80.3", "26.1", "36.0"]
    assert rows[7] == ["Average gain", "+14.9", "+6.7", "–"]

    # Verify normalizer flags reliable=False due to squashed sub-rows
    norm = normalize_results_records(headers, rows, page_number=4, caption=caption)
    assert norm["reliable"] is False
    assert "irregular" in norm["note"].lower() or "collapsed" in norm["note"].lower()
    # Emitted clean records must not contain squashed multi-model strings
    for r in norm["records"]:
        assert "TinyLlama" not in r["model"] or "Phi-2" not in r["model"]
        assert "t =" not in r["model"]
        assert "§" not in r["model"]


def test_real_paper7_page6_table_values():
    """
    Regression test for real Paper 7 (2024.emnlp-main.1259) Page 6 Table 1.
    Verifies that complex benchmark tables with squashed baseline rows strictly flag
    reliable=False, while clean single-model rows preserve exact numbers and gains.
    """
    headers = ["Model", "HotpotQA", "2WikiMQA", "MusiQue", "Average"]
    rows = [
        ["", "", "", "", ""],
        ["#", "Long-Context LLM Methods #", "", "", ""],
        ["LongAlign-7B-64k (Llama2) (Bai et al., 2024) LongLoRA-13B-32k (Llama2) (Chen et al., 2023b)", "48.85 47.45", "28.56 42.92", "25.14 29.46", "34.18 39.94"],
        ["", "# Advanced RAG Methods #", "", "", ""],
        ["CFIC-7B (Llama2) (Qian et al., 2024) CRAG (GPT-3.5-Turbo) (Yan et al., 2024) Self-RAG (GPT-3.5-Turbo) (Asai et al., 2023)", "34.00 52.04 50.51", "- 41.13 46.75", "14.70 25.34 24.62", "24.35 39.50 40.63"],
        ["LongRAG-ChatGLM3-6B-32k", "55.93 (3.36↑)", "54.85 (12.29↑)", "33.00 (7.49↑)", "47.93 (7.71↑)"],
        ["LongRAG-GLM-4", "62.11 (4.70↑)", "57.16 (4.25↑)", "38.40 (10.85↑)", "52.56 (6.60↑)"],
    ]
    caption = "Table 1: Results (%) of overall performance on three multi-hop datasets."

    norm = normalize_results_records(headers, rows, page_number=6, caption=caption)
    assert norm["reliable"] is False

    models_extracted = {r["model"] for r in norm["records"]}
    assert "LongRAG-ChatGLM3-6B-32k" in models_extracted
    assert "LongRAG-GLM-4" in models_extracted
    # Squashed multi-model row must be excluded from records
    assert not any("LongAlign" in m for m in models_extracted)


# ---------------------------------------------------------------------------
# 7. Word-Position Row Reconstruction & Spanning Headers Tests
# ---------------------------------------------------------------------------

def test_word_position_synthetic_spanning_headers_and_groups(tmp_path):
    """
    Test word-position row reconstruction on a synthetic unruled PDF table
    featuring spanning headers, group rows, wrapped labels, and numbers with gains.
    """
    pdf_path = str(tmp_path / "synthetic_unruled.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    # Caption above
    page.insert_text((50, 60), "Table 1: Retrieval and match accuracy on Benchmark Dataset.")

    # Top header line (spanning "Adaptive Retrieval" over col 1 & 2, "Always Retrieval" over col 3)
    page.insert_text((180, 75), "Adaptive Retrieval")
    page.insert_text((380, 70), "Always")
    page.insert_text((380, 80), "Retrieval")

    # Bottom header line
    page.insert_text((50, 95), "Model")
    page.insert_text((180, 95), "Retrieval")
    page.insert_text((280, 95), "Match")
    page.insert_text((380, 95), "Match")

    # Section 1 Header
    page.insert_text((50, 115), "Vanilla §3")

    # Row 1
    page.insert_text((50, 135), "TinyLlama (1.1B)")
    page.insert_text((180, 135), "39.1")
    page.insert_text((280, 135), "14.7")
    page.insert_text((380, 135), "28.2")

    # Row 2
    page.insert_text((50, 155), "Phi-2 (2.7B)")
    page.insert_text((180, 155), "94.1")
    page.insert_text((280, 155), "35.0")
    page.insert_text((380, 155), "36.4")

    # Section 2 Header
    page.insert_text((50, 175), "Ours TA-ARE §4")

    # Row 3 (with relative gains and dashes)
    page.insert_text((50, 195), "TinyLlama (1.1B)")
    page.insert_text((180, 195), "54.1(+15.0)")
    page.insert_text((280, 195), "19.0(+4.3)")
    page.insert_text((380, 195), "28.2")

    # Row 4 (Average gain)
    page.insert_text((50, 215), "Average gain")
    page.insert_text((180, 215), "+14.9")
    page.insert_text((280, 215), "+6.7")
    page.insert_text((380, 215), "-")

    doc.save(pdf_path)
    doc.close()

    doc = pymupdf.open(pdf_path)
    p = doc[0]
    bbox = (45, 65, 450, 225)
    raw_headers = ["Model", "Adaptive", "Retrieval", "Always Retrieval"]
    raw_rows = [
        ["TinyLlama (1.1B) Phi-2 (2.7B)", "39.1 94.1", "14.7 35.0", "28.2 36.4"],
        ["Average gain", "+14.9", "+6.7", "-"],
    ]

    recon = reconstruct_table_from_words(
        page=p,
        bbox=bbox,
        raw_headers=raw_headers,
        raw_rows=raw_rows,
        caption="Table 1: Retrieval and match accuracy on Benchmark Dataset.",
        page_number=1,
    )
    doc.close()

    assert recon is not None
    assert recon["reconstruction_passed_gate"] is True
    assert recon["reliable"] is True
    assert len(recon["rows"]) == 4

    headers = recon["headers"]
    assert headers[0] == "Model"
    assert "Retrieval" in headers[1]
    assert "Match" in headers[2]
    assert "Always Retrieval" in headers[3] and "Match" in headers[3]

    # Verify groups
    assert recon["row_groups"][0] == "Vanilla §3"
    assert recon["row_groups"][1] == "Vanilla §3"
    assert recon["row_groups"][2] == "Ours TA-ARE §4"
    assert recon["row_groups"][3] == "Ours TA-ARE §4"

    # Verify row values
    assert recon["rows"][0] == ["TinyLlama (1.1B)", "39.1", "14.7", "28.2"]
    assert recon["rows"][2] == ["TinyLlama (1.1B)", "54.1(+15.0)", "19.0(+4.3)", "28.2"]
    assert recon["rows"][3] == ["Average gain", "+14.9", "+6.7", "-"]

    # Verify normalized records have group field
    records = recon["normalized_records"]
    assert len(records) == 9  # 3 model rows * 3 cols (Average gain is excluded from records)
    for r in records:
        assert "group" in r
        assert r["group"] in ["Vanilla §3", "Ours TA-ARE §4"]
        assert r["page"] == 1


def test_word_position_refusal_gate(tmp_path):
    """
    Test confidence gate refusal when row reconstruction produces non-numeric tokens
    in numeric columns or ragged irregular shapes.
    """
    pdf_path = str(tmp_path / "synthetic_bad.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    # Header
    page.insert_text((50, 95), "Model")
    page.insert_text((160, 95), "Score A")
    page.insert_text((260, 95), "Score B")

    # Row with non-numeric text in numeric column
    page.insert_text((50, 120), "Model-A")
    page.insert_text((160, 120), "INVALID_TEXT_ENTRY")
    page.insert_text((260, 120), "45.2")

    page.insert_text((50, 140), "Model-B")
    page.insert_text((160, 140), "88.1")
    page.insert_text((260, 140), "79.3")

    doc.save(pdf_path)
    doc.close()

    doc = pymupdf.open(pdf_path)
    p = doc[0]
    bbox = (45, 90, 320, 150)
    raw_headers = ["Model", "Score A", "Score B"]
    raw_rows = [["Model-A\nModel-B", "INVALID_TEXT_ENTRY\n88.1", "45.2\n79.3"]]

    # Reconstruct should refuse due to invalid non-numeric token
    recon = reconstruct_table_from_words(
        page=p,
        bbox=bbox,
        raw_headers=raw_headers,
        raw_rows=raw_rows,
        caption="Table 1: Bad table",
        page_number=1,
    )
    doc.close()

    assert recon is None


def test_clean_ruled_table_unaltered():
    """
    Test clean ruled tables are detected as non-merged and not touched
    (reconstruction_method == 'none').
    """
    # Clean table with distinct rows and columns
    raw_headers = ["Model", "Accuracy", "F1 Score"]
    raw_rows = [
        ["BERT", "85.2", "84.1"],
        ["RoBERTa", "89.1", "88.7"],
        ["DeBERTa", "91.4", "91.0"],
    ]

    is_merged = is_merged_table_candidate(raw_headers, raw_rows, line_count=4)
    assert is_merged is False


def test_real_paper6_page4_word_position_reconstruction():
    """
    Regression test for real Paper 6 (2024.findings-acl.415) Page 4 Table 1.
    Verifies that word-position reconstruction separates all 15 model rows,
    preserves exact PDF values, and verifies the GPT-4* row from the text layer.
    """
    import os
    pdf_paths = [
        "data/uploaded_papers/default-collection/87cf21366266_2024.findings-acl.415-paper-6.pdf",
        "data/uploaded_papers/2024.findings-acl.415-paper-6.pdf",
    ]
    pdf_path = next((p for p in pdf_paths if os.path.exists(p)), None)
    if not pdf_path:
        return

    doc = pymupdf.open(pdf_path)
    page = doc[3]  # Page 4 (0-indexed 3)
    tables = page.find_tables().tables
    assert len(tables) >= 1
    table = tables[0]

    raw_headers = table.extract()[0]
    raw_rows = table.extract()[1:]
    caption = "Table 1: Retrieval and match accuracy on RetrievalQA."

    recon = reconstruct_table_from_words(
        page=page,
        bbox=tuple(table.bbox),
        raw_headers=raw_headers,
        raw_rows=raw_rows,
        caption=caption,
        page_number=4,
    )
    doc.close()

    assert recon is not None
    assert recon["reconstruction_passed_gate"] is True
    assert recon["reliable"] is True
    assert len(recon["rows"]) == 15

    # Check composite headers
    assert recon["headers"][0] == "Baselines (1,271)"
    assert "Adaptive Retrieval" in recon["headers"][1] and "Retrieval" in recon["headers"][1]
    assert "Adaptive Retrieval" in recon["headers"][2] and "Match" in recon["headers"][2]
    assert "Always Retrieval" in recon["headers"][3] and "Match" in recon["headers"][3]

    # Model rows
    rows = recon["rows"]
    # Row 4: TinyLlama (1.1B) under Vanilla
    assert rows[4] == ["TinyLlama (1.1B)", "39.1", "14.7", "28.2"]
    # Row 5: Phi-2 (2.7B) under Vanilla
    assert rows[5] == ["Phi-2 (2.7B)", "94.1", "35.0", "36.4"]
    # Row 6: Llama-2 (7B) under Vanilla
    assert rows[6] == ["Llama-2 (7B)", "80.3", "26.1", "36.0"]
    # Row 7: GPT-3.5 under Vanilla
    assert rows[7] == ["GPT-3.5", "49.3", "20.8", "38.2"]
    # Row 8: GPT-4* under Vanilla
    assert rows[8] == ["GPT-4*", "67.6", "37.6", "46.0"]
    # Verify GPT-4* middle cell in text layer is strictly 37.6
    print("Real Paper 6 Page 4 GPT-4* row from text layer:", rows[8])
    assert rows[8][2] == "37.6"

    # Ours TA-ARE §4 section rows
    assert rows[9] == ["TinyLlama (1.1B)", "54.1(+15.0)", "19.0(+4.3)", "28.2"]
    assert rows[10] == ["Phi-2 (2.7B)", "95.5(+1.4)", "36.0(+1.0)", "36.4"]
    assert rows[11] == ["Llama-2 (7B)", "86.0(+5.7)", "30.7(+4.6)", "36.0"]
    assert rows[12] == ["GPT-3.5", "86.3(+37.0)", "35.8(+15.0)", "38.2"]
    assert rows[13] == ["GPT-4*", "83.2(+15.6)", "46.4(+8.8)", "46.0"]

    # Row 14: Average gain
    assert rows[14] == ["Average gain", "+14.9", "+6.7", "–"]


def test_real_paper7_page6_word_position_reconstruction():
    """
    Regression test for real Paper 7 (2024.emnlp-main.1259) Page 6 Table 1.
    Verifies that 22 rows are separated and the 4 dataset columns are correctly mapped.
    """
    import os
    pdf_paths = [
        "data/uploaded_papers/default-collection/0a5f3581f839_2024.emnlp-main.1259-paper-7.pdf",
        "data/uploaded_papers/2024.emnlp-main.1259-paper-7.pdf",
    ]
    pdf_path = next((p for p in pdf_paths if os.path.exists(p)), None)
    if not pdf_path:
        return

    doc = pymupdf.open(pdf_path)
    page = doc[5]  # Page 6 (0-indexed 5)
    tables = page.find_tables().tables
    assert len(tables) >= 1
    table = tables[0]

    raw_headers = table.extract()[0]
    raw_rows = table.extract()[1:]
    caption = "Table 1: Results (%) of overall performance on three multi-hop datasets."

    recon = reconstruct_table_from_words(
        page=page,
        bbox=tuple(table.bbox),
        raw_headers=raw_headers,
        raw_rows=raw_rows,
        caption=caption,
        page_number=6,
    )
    doc.close()

    assert recon is not None
    assert recon["reconstruction_passed_gate"] is True
    assert recon["reliable"] is True
    assert len(recon["rows"]) == 22
    assert recon["headers"] == ["Model", "HotpotQA", "2WikiMQA", "MusiQue", "Average"]

    # Verify rows
    rows = recon["rows"]
    assert rows[0] == ["LongAlign-7B-64k (Llama2) (Bai et al., 2024)", "48.85", "28.56", "25.14", "34.18"]
    assert rows[2] == ["CFIC-7B (Llama2) (Qian et al., 2024)", "34.00", "-", "14.70", "24.35"]
    assert rows[21] == ["LongRAG-GLM-4", "62.11 (4.70↑)", "57.16 (4.25↑)", "38.40 (10.85↑)", "52.56 (6.60↑)"]

    # Verify normalized records map HotpotQA, 2WikiMQA, MusiQue
    rec_datasets = {r["dataset"] for r in recon["normalized_records"]}
    assert "HotpotQA" in rec_datasets
    assert "2WikiMQA" in rec_datasets
    assert "MusiQue" in rec_datasets


def test_is_merged_table_candidate_variants():
    """Test various triggers for candidate detection."""
    assert is_merged_table_candidate([], []) is False
    assert is_merged_table_candidate(["Model", "Score"], [["Single\nLine\nWrapped", "10.5"]]) is True
    assert is_merged_table_candidate(["Model", "Score"], [["Model A (7B) Model B (13B)", "10.5"]]) is True
    assert is_merged_table_candidate(["Model", "Score"], [["Model A", "10.5 12.4"]]) is True
    assert is_merged_table_candidate(["Model", "Score"], [["M1", "10.5"]], line_count=10) is True


def test_is_data_token_coverage():
    """Test helper _is_data_token against academic table tokens."""
    # Data tokens
    assert _is_data_token((0, 0, 10, 10, "54.1(+15.0)"), [(0, 0, 10, 10, "54.1(+15.0)")]) is True
    assert _is_data_token((0, 0, 10, 10, "–"), [(0, 0, 10, 10, "–")]) is True
    assert _is_data_token((0, 0, 10, 10, "-"), [(0, 0, 10, 10, "-")]) is True
    # Non-data tokens
    assert _is_data_token((0, 0, 10, 10, "(7B)"), [(0, 0, 10, 10, "(7B)")]) is False
    assert _is_data_token((0, 0, 10, 10, "(1,271)"), [(0, 0, 10, 10, "(1,271)")]) is False
    assert _is_data_token((0, 0, 10, 10, "none"), [(0, 0, 10, 10, "none")]) is False
    # Threshold t = 0.5
    line_t = [(0, 0, 0, 0, "t"), (0, 0, 0, 0, "="), (0, 0, 0, 0, "0.5")]
    assert _is_data_token(line_t[2], line_t) is False


def test_extract_tables_from_pdf_with_word_reconstruction(tmp_path):
    """Test extract_tables_from_pdf on real Paper 6 with DB persistence and reconstruction."""
    import os
    pdf_paths = [
        "data/uploaded_papers/default-collection/87cf21366266_2024.findings-acl.415-paper-6.pdf",
        "data/uploaded_papers/2024.findings-acl.415-paper-6.pdf",
    ]
    pdf_path = next((p for p in pdf_paths if os.path.exists(p)), None)
    if not pdf_path:
        return

    db_path = str(tmp_path / "test_p6.db")
    db = DatabaseManager(db_path=db_path)
    paper_id = db.insert_paper(
        collection_id="default-collection",
        file_hash="p6_hash_test",
        filename="p6.pdf",
        file_path=pdf_path,
        title="Paper 6",
        n_pages=5,
    )

    tables = extract_tables_from_pdf(pdf_path, paper_id, db=db)
    assert len(tables) >= 1
    # Check that Table 1 on page 4 was reconstructed
    t1 = next((t for t in tables if t["page_number"] == 4 and t["table_index"] == 1), None)
    assert t1 is not None
    assert t1["reconstruction_method"] == "words"
    assert t1["reconstruction_passed_gate"] is True
    assert t1["normalization_reliable"] is True
    assert len(t1["rows"]) == 15
    assert len(t1["normalized_records"]) > 0


def test_normalize_table_dict_edge_cases():
    """Test normalize_table_dict with edge case formats and invalid types."""
    # Non-dictionary input
    res = normalize_table_dict("not_a_dict")
    assert res["is_accepted"] is False
    assert res["rejection_reason"] == "Non-dictionary input"

    # Single string header
    res2 = normalize_table_dict({"headers": "SingleHeader", "rows": [["val"]]})
    assert res2["headers"] == ["SingleHeader"]

    # String row_count and col_count
    res3 = normalize_table_dict({
        "headers": ["A", "B"],
        "rows": [["1", "2"]],
        "row_count": "invalid",
        "col_count": "invalid",
    })
    assert res3["row_count"] == 1
    assert res3["col_count"] == 2

    # Missing headers but valid rows
    res4 = normalize_table_dict({"rows": [["a", "b", "c"]]})
    assert res4["col_count"] == 3
    assert res4["headers"] == ["Col_1", "Col_2", "Col_3"]



