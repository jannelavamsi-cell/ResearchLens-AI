"""
Results table extraction, filtering, classification, and normalization module for ResearchLens AI.
Extracts tabular data from research papers using PyMuPDF find_tables(),
applies strict defensive filtering to reject false positives (paragraphs, reference lists, layout artifacts),
detects and links nearby 'Table N' captions, classifies tables into a structured academic taxonomy,
and normalizes empirical results tables into canonical {model, dataset, metric, value, page} tuples.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from database.db_manager import DatabaseManager

logger = logging.getLogger("researchlens.tables")

# ---------------------------------------------------------------------------
# Explicit, Documented Filtering Thresholds & Constraints
# ---------------------------------------------------------------------------
MIN_TABLE_ROWS: int = 2                 # Minimum rows (header + at least 1 data row)
MIN_TABLE_COLS: int = 2                 # Minimum columns (single-column lists rejected)
MAX_EMPTY_CELL_RATIO: float = 0.60      # Max ratio of empty cells (skeletons rejected)
MAX_AVG_CELL_LENGTH: float = 120.0      # Max average cell length in chars (paragraphs rejected)
MAX_CELL_SENTENCE_LENGTH: int = 220     # Max length for individual prose sentence cells
MIN_NUMERIC_RATIO_RESULTS: float = 0.15 # Min ratio of numeric cells for results classification
MAX_PAGE_HEIGHT_COVERAGE: float = 0.85  # Table bbox covering >85% of page vertical height is body text


def _matches_term(term: str, text: str) -> bool:
    """Checks if a term appears with word boundaries or exact token match, permitting optional plural 's'."""
    pattern = r"(?:^|\W)" + re.escape(term) + r"s?(?:$|\W)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def evaluate_table_candidate(
    headers: List[str],
    rows: List[List[Any]],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    page_height: Optional[float] = None,
    section_name: Optional[str] = None,
    surrounding_text: Optional[str] = None,
    caption: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether raw tabular extraction constitutes a genuine, well-formed table
    versus false positives (layout artifacts, paragraphs, reference lists, or empty grids).

    Explicit criteria applied:
    1. Minimum 2 rows and 2 columns.
    2. Consistent column count across rows (reject ragged grids).
    3. Ratio of empty cells <= MAX_EMPTY_CELL_RATIO.
    4. Maximum average cell length <= MAX_AVG_CELL_LENGTH (reject paragraphs).
    5. Reject tables containing long narrative sentences.
    6. Reject tables located inside references / bibliography sections.
    7. Reject tables covering most of a text column (>85% page vertical height).

    Returns:
        (is_accepted: bool, rejection_reason: Optional[str])
    """
    # 1. Row dimension check
    total_rows = len(rows) + (1 if headers else 0)
    if total_rows < MIN_TABLE_ROWS:
        return False, f"Fewer than {MIN_TABLE_ROWS} rows (total={total_rows})"

    # 2. Column dimension & consistency check
    all_rows = ([headers] if headers else []) + [r for r in rows if isinstance(r, list)]
    if not all_rows:
        return False, "Empty table structure"

    col_counts = [len(r) for r in all_rows]
    max_cols = max(col_counts) if col_counts else 0
    min_cols = min(col_counts) if col_counts else 0

    if max_cols < MIN_TABLE_COLS:
        return False, f"Fewer than {MIN_TABLE_COLS} columns (columns={max_cols})"

    # Reject wildly ragged layouts (difference between max and min cols > 3)
    if max_cols - min_cols > 3:
        return False, f"Inconsistent column count across rows (ragged range: {min_cols} to {max_cols} columns)"

    # Check for empty grid or single non-empty row
    non_empty_rows = [r for r in all_rows if any(str(c).strip() for c in r if c is not None)]
    if len(non_empty_rows) < MIN_TABLE_ROWS:
        return False, f"Fewer than {MIN_TABLE_ROWS} non-empty rows"

    # 3. Cell content analysis
    total_cells = 0
    empty_cells = 0
    cell_lengths: List[int] = []
    numeric_cells = 0
    narrative_sentences = 0
    bib_citation_matches = 0

    bib_patterns = [
        r"^\[\d+\]",
        r"\bet\s+al\b",
        r"\b(arxiv|doi|proceedings|ieee|acm|vol\.|pp\.)\b",
        r"\b(19\d\d|20\d\d)\b",
    ]
    bib_regex = re.compile("|".join(bib_patterns), re.IGNORECASE)

    for r in all_rows:
        for c in r:
            total_cells += 1
            s = str(c).strip() if c is not None else ""
            if not s:
                empty_cells += 1
            else:
                cell_lengths.append(len(s))
                if any(ch.isdigit() for ch in s):
                    numeric_cells += 1

                # Check for prose paragraph indicators:
                # A prose narrative cell has substantial text without tabular numbers
                digits_count = sum(1 for ch in s if ch.isdigit())
                digits_ratio = digits_count / len(s) if len(s) > 0 else 0.0
                is_prose_block = digits_ratio < 0.08

                if is_prose_block:
                    if len(s) > MAX_CELL_SENTENCE_LENGTH:
                        narrative_sentences += 1
                    elif len(s) > 75 and (s.count(". ") >= 1 or s.endswith(".")):
                        narrative_sentences += 1

                # Check for bibliography markers
                if bib_regex.search(s):
                    bib_citation_matches += 1

    if total_cells == 0:
        return False, "Zero total cells in table grid"

    # 4. Empty cell ratio threshold
    empty_ratio = empty_cells / total_cells
    if empty_ratio > MAX_EMPTY_CELL_RATIO:
        return False, f"Excessive empty cell ratio ({empty_ratio:.1%} > {MAX_EMPTY_CELL_RATIO:.0%})"

    # 5. Average cell length threshold (reject paragraphs)
    avg_len = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0.0
    num_ratio = numeric_cells / total_cells
    if avg_len > MAX_AVG_CELL_LENGTH and num_ratio < MIN_NUMERIC_RATIO_RESULTS:
        return False, f"Average cell length exceeds threshold ({avg_len:.1f} > {MAX_AVG_CELL_LENGTH:.0f} chars); narrative text block"

    # 6. Narrative sentences check
    if narrative_sentences >= 1 and num_ratio < MIN_NUMERIC_RATIO_RESULTS:
        return False, f"Contains {narrative_sentences} narrative prose sentence cell(s); misdetected paragraph"
    if narrative_sentences >= 2:
        return False, f"Contains {narrative_sentences} narrative prose sentence cell(s); misdetected paragraph"

    # 7. References / Bibliography check
    sec_lower = str(section_name or "").lower()
    surr_lower = str(surrounding_text or "").lower()
    is_ref_sec = any(
        k in sec_lower for k in ("reference", "bibliography", "works cited", "literature cited")
    ) or any(
        k in surr_lower for k in ("references\n", "bibliography\n", "literature cited\n")
    )
    has_table_caption = bool(
        caption and re.match(r"^(Table\s+\d+|TABLE\s+[IVXLCDM\d]+)", caption.strip(), re.IGNORECASE)
    )

    if is_ref_sec and not has_table_caption:
        return False, "Located inside references or bibliography section"

    if bib_citation_matches >= 3 and not has_table_caption:
        return False, "Located inside references or bibliography section"

    # 8. Excessive vertical page coverage (>85% page height)
    if bbox is not None and page_height is not None and page_height > 0:
        table_height = bbox[3] - bbox[1]
        coverage_ratio = table_height / page_height
        if coverage_ratio > MAX_PAGE_HEIGHT_COVERAGE:
            return False, f"Table vertical coverage exceeds {MAX_PAGE_HEIGHT_COVERAGE:.0%} of page height ({coverage_ratio:.1%})"

    return True, None


def extract_table_caption(
    page: pymupdf.Page,
    bbox: Tuple[float, float, float, float],
    max_search_dist: float = 75.0,
) -> str:
    """
    Searches immediately above and below a table bounding box for standard
    academic captions matching patterns like 'Table N: ...', 'TABLE I: ...'.
    """
    rect = pymupdf.Rect(bbox)
    above_rect = pymupdf.Rect(0, max(0, rect.y0 - max_search_dist), page.rect.width, rect.y0 + 4)
    below_rect = pymupdf.Rect(0, rect.y1 - 4, page.rect.width, min(page.rect.height, rect.y1 + max_search_dist))

    lines_above = [ln.strip() for ln in page.get_text("text", clip=above_rect).split("\n") if ln.strip()]
    lines_below = [ln.strip() for ln in page.get_text("text", clip=below_rect).split("\n") if ln.strip()]

    caption_pattern = re.compile(
        r"^(Table\s+\d+|TABLE\s+[IVXLCDM\d]+)[:.\s-]\s*(.*)", re.IGNORECASE
    )

    # Check lines below first (standard in ACL / EMNLP), then above
    for candidate_lines in [lines_below, lines_above]:
        for i, line in enumerate(candidate_lines):
            m = caption_pattern.match(line)
            if m:
                caption = line
                # Join subsequent 1-2 continuation lines if they are part of the caption
                for j in range(i + 1, min(i + 3, len(candidate_lines))):
                    next_line = candidate_lines[j]
                    if not re.match(
                        r"^(Table|Figure|Algorithm|Section|\d+\.|\d+\s+[A-Z])",
                        next_line,
                        re.IGNORECASE,
                    ):
                        caption += " " + next_line
                    else:
                        break
                return caption.strip()

    return ""


def classify_table(
    headers: List[str],
    rows: List[List[Any]],
    caption: str = "",
) -> str:
    """
    Classifies a genuine table into one of four categories:
    1. 'results': Empirical performance, benchmark comparisons, ablation studies.
    2. 'hyperparameters': Training settings, parameters, hardware configurations.
    3. 'dataset_statistics': Corpus metrics, split sizes, token counts, sample counts.
    4. 'other': Model architectures list, prompt templates, qualitative examples.
    """
    text_corpus = (caption + " " + " ".join(str(h) for h in headers)).lower()

    # Count numeric cells
    numeric_cells = 0
    total_cells = 0
    for r in rows:
        if isinstance(r, list):
            for c in r:
                total_cells += 1
                s = str(c or "").strip()
                if any(ch.isdigit() for ch in s):
                    numeric_cells += 1

    # 1. Hyperparameters
    hp_terms = [
        "hyperparameter", "learning rate", "batch size", "optimizer",
        "temperature", "top_p", "weight decay", "dropout", "warmup",
        "hidden size", "implementation hyperparameters"
    ]
    if any(_matches_term(t, text_corpus) for t in hp_terms):
        return "hyperparameters"

    # 2. Dataset statistics
    ds_terms = [
        "dataset statistics", "statistics of", "statistics of experimental data",
        "data statistics", "#original", "#after filtering", "num of long-context",
        "num of cot", "avg. length of p", "token length fed into", "dataset size"
    ]
    if any(_matches_term(t, text_corpus) for t in ds_terms):
        return "dataset_statistics"

    # 3. Results tables: must feature numbers AND metric indicators
    metric_terms = [
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge", "auc",
        "map", "ndcg", "mrr", "loss", "error", "latency", "throughput", "p@1",
        "em", "exact match", "perplexity", "score", "match", "results (%)",
        "overall performance", "performance", "ablation", "averagegain",
        "relative gain", "results", "transferability", "benchmark"
    ]
    has_metric = any(_matches_term(t, text_corpus) for t in metric_terms)

    # Reject qualitative tables (e.g. data examples, prompts, QA pairs)
    qual_header_terms = ["question", "answer", "prompt", "completion", "input", "output"]
    headers_corpus = " ".join(str(h) for h in headers).lower()
    has_header_metric = any(
        _matches_term(m, headers_corpus)
        for m in ["accuracy", "match", "f1", "score", "precision", "recall", "bleu", "rouge", "loss", "retrieval"]
    )
    is_qualitative = any(_matches_term(q, headers_corpus) for q in qual_header_terms) and not has_header_metric

    if not is_qualitative and has_metric and numeric_cells >= 1:
        return "results"

    return "other"


def _is_multi_value_cell(val_str: str) -> bool:
    """Check if a table cell contains multiple collapsed numeric values or mixed phrases."""
    v = val_str.strip()
    if not v:
        return False
    # Contaminated with table subheaders
    if any(term in v.lower() for term in ["match", "retrieval", "calibration", "model-based"]):
        return True
    tokens = [t for t in v.split() if re.match(r"^[-+]?\d+(?:\.\d+)?(?:\([+-]?\d+(?:\.\d+)?[↑↓%]?\))?%?$", t)]
    if len(tokens) > 1:
        return True
    if len(v.split()) > 3 and len(tokens) >= 1:
        return True
    return False


def _is_squashed_model(m_str: str) -> bool:
    """Check if a model cell contains multiple squashed model names or section markers."""
    m = m_str.strip()
    if re.search(r"\bt\s*=", m, re.I):
        return True
    if len(re.findall(r"\(\d+(?:\.\d+)?[Bb]\)", m)) > 1:
        return True
    if len(re.findall(r"\(\s*[A-Za-z]+\s+et\s+al\.", m)) > 1:
        return True
    families = ["llama", "phi", "gpt", "vicuna", "glm", "qwen", "mistral", "claude", "bert", "t5"]
    found = [fam for fam in families if re.search(r"\b" + fam, m, re.I)]
    if len(found) > 1:
        return True
    if "§" in m or "ourswith" in m.lower().replace(" ", ""):
        return True
    return False


def normalize_results_records(
    headers: List[str],
    rows: List[List[Any]],
    page_number: int = 1,
    caption: str = "",
    row_groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Converts empirical results tables into structured records:
    [{"model": str, "group": str, "dataset": str, "metric": str, "value": str, "page": int}, ...]

    If the table structure does not permit unambiguous decomposition
    (e.g. squashed multiline text without row borders, irregular headers, missing identifiers),
    it flags reliable=False, preserves raw table, and provides an explanatory note without inventing values.
    """
    if not headers or not rows:
        return {
            "reliable": False,
            "records": [],
            "note": "Empty headers or rows; cannot decompose.",
        }

    # Detect dataset name from caption or headers
    detected_dataset = "Benchmark Dataset"
    known_benchmarks = [
        "RetrievalQA", "HotpotQA", "2WikiMultiHopQA", "2WikiMQA", "MusiQue",
        "QASPER", "SQuAD", "GLUE", "SuperGLUE", "ImageNet", "FreshQA", "RealTimeQA"
    ]
    caption_clean = caption.lower()
    for bm in known_benchmarks:
        if bm.lower() in caption_clean:
            detected_dataset = bm
            break

    # Identify candidate model column
    model_terms = ["model", "method", "approach", "baseline", "system", "architecture", "variant", "generator"]
    model_col_idx = 0
    for idx, h in enumerate(headers):
        if any(_matches_term(m, str(h).lower()) for m in model_terms):
            model_col_idx = idx
            break

    # Identify whether headers are metrics or datasets
    metric_terms = [
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge",
        "auc", "map", "ndcg", "mrr", "loss", "error", "latency", "match",
        "score", "em", "p@1", "average"
    ]

    header_types: List[Dict[str, Any]] = []
    for idx, h in enumerate(headers):
        if idx == model_col_idx:
            continue
        h_str = str(h).strip()
        h_lower = h_str.lower()
        if not h_str:
            continue

        # Check if header matches a benchmark name
        is_bm = any(bm.lower() in h_lower for bm in known_benchmarks)
        is_metric = any(_matches_term(m, h_lower) for m in metric_terms)

        if is_bm:
            header_types.append({"col_idx": idx, "type": "dataset", "name": h_str})
        elif is_metric:
            header_types.append({"col_idx": idx, "type": "metric", "name": h_str})
        else:
            header_types.append({"col_idx": idx, "type": "metric", "name": h_str})

    if not header_types:
        return {
            "reliable": False,
            "records": [],
            "note": "No distinct metric or dataset columns identified in headers; raw table preserved.",
        }

    records: List[Dict[str, Any]] = []
    unreliable_detected = False
    invalid_models = {
        "#", ".,", ".", "–", "-", "average", "average gain", "averagegain",
        "total", "total/average", "vanilla §3", "ours ta-are §4",
    }
    invalid_values = {".", ".,", "–", "-", ""}

    for row_idx, row in enumerate(rows):
        if not isinstance(row, list) or len(row) <= model_col_idx:
            continue

        group_val = row_groups[row_idx] if row_groups and row_idx < len(row_groups) else ""

        raw_model = str(row[model_col_idx]).strip()
        if not raw_model or raw_model.lower() in invalid_models or raw_model.startswith("#"):
            continue

        if _is_squashed_model(raw_model):
            unreliable_detected = True
            continue

        # Check for multiline squashed rows
        model_lines = [ln.strip() for ln in raw_model.split("\n") if ln.strip()]
        model_lines = [m for m in model_lines if m.lower() not in invalid_models and not m.startswith("#")]

        if not model_lines:
            continue

        if len(model_lines) > 1:
            col_lines: Dict[int, List[str]] = {}
            line_match = True
            for ht in header_types:
                c_idx = ht["col_idx"]
                if c_idx < len(row):
                    val_lines = [v.strip() for v in str(row[c_idx] or "").split("\n") if v.strip()]
                    col_lines[c_idx] = val_lines
                    if len(val_lines) != len(model_lines):
                        line_match = False
                else:
                    line_match = False

            if not line_match:
                unreliable_detected = True
                continue

            for item_idx, m_name in enumerate(model_lines):
                if m_name.lower() in invalid_models or m_name.startswith("#") or _is_squashed_model(m_name):
                    unreliable_detected = True
                    continue
                for ht in header_types:
                    c_idx = ht["col_idx"]
                    val = col_lines[c_idx][item_idx]
                    if val in invalid_values:
                        continue
                    if _is_multi_value_cell(val):
                        unreliable_detected = True
                        continue
                    if ht["type"] == "dataset":
                        ds_name = ht["name"]
                        met_name = "Score (%)" if "%" in caption else "Performance"
                    else:
                        ds_name = detected_dataset
                        met_name = ht["name"]

                    records.append({
                        "model": m_name,
                        "group": group_val,
                        "dataset": ds_name,
                        "metric": met_name,
                        "value": val,
                        "page": page_number,
                    })
        else:
            m_name = model_lines[0]
            if m_name.lower() in invalid_models or m_name.startswith("#") or _is_squashed_model(m_name):
                unreliable_detected = True
                continue
            for ht in header_types:
                c_idx = ht["col_idx"]
                val = str(row[c_idx]).strip() if c_idx < len(row) and row[c_idx] is not None else ""
                if not val or val in invalid_values:
                    continue
                if _is_multi_value_cell(val):
                    unreliable_detected = True
                    continue

                if ht["type"] == "dataset":
                    ds_name = ht["name"]
                    met_name = "Score (%)" if "%" in caption else "Performance"
                else:
                    ds_name = detected_dataset
                    met_name = ht["name"]

                records.append({
                    "model": m_name,
                    "group": group_val,
                    "dataset": ds_name,
                    "metric": met_name,
                    "value": val,
                    "page": page_number,
                })

    if not records or unreliable_detected:
        return {
            "reliable": False,
            "records": records,
            "note": "Irregular row/column alignment or collapsed sub-rows prevent unambiguous decomposition; raw table preserved.",
        }

    return {
        "reliable": True,
        "records": records,
        "note": f"Successfully decomposed into {len(records)} structured results tuples.",
    }


def is_valid_table(headers: List[str], rows: List[List[Any]]) -> bool:
    """
    Checks if tabular data has a meaningful structural shape.
    Rejects single-cell artifacts, 1-row metadata blocks, and empty grids.
    (Maintained for backwards-compatibility).
    """
    accepted, _ = evaluate_table_candidate(headers, rows)
    return accepted


def is_likely_results_table(headers: List[str], rows: List[List[Any]]) -> bool:
    """
    Checks if an extracted table appears to be an empirical results table.
    (Maintained for backwards-compatibility).
    """
    if not is_valid_table(headers, rows):
        return False
    cl = classify_table(headers, rows)
    return cl == "results"


def parse_table_semantics(
    headers: List[str],
    rows: List[List[Any]],
    caption: str = "",
) -> Dict[str, Any]:
    """
    Heuristically extracts experimental semantics from a results table.
    """
    metric_terms = {
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge",
        "auc", "map", "ndcg", "mrr", "loss", "error", "latency", "em", "score", "match"
    }
    model_terms = {"model", "method", "approach", "baseline", "system", "architecture", "generator"}

    metric_cols = []
    model_col = None

    for idx, h in enumerate(headers):
        h_clean = str(h).lower().strip()
        if any(_matches_term(m, h_clean) for m in metric_terms):
            metric_cols.append({"index": idx, "name": h})
        if model_col is None and any(_matches_term(m, h_clean) for m in model_terms):
            model_col = {"index": idx, "name": h}

    if model_col is None and headers:
        model_col = {"index": 0, "name": headers[0]}

    known_benchmarks = [
        "squad", "glue", "superglue", "imagenet", "retrievalqa", "mmlongbench",
        "hotpotqa", "triviaqa", "ms-marco", "msmarco", "pubmed", "nq", "natural questions",
        "2wikimqa", "2wikimultihopqa", "musique", "qasper"
    ]
    detected_benchmarks = []
    text_corpus = (caption + " " + " ".join(headers)).lower()
    for bm in known_benchmarks:
        if bm in text_corpus:
            detected_benchmarks.append(bm.upper())

    return {
        "metric_columns": metric_cols,
        "model_column": model_col,
        "detected_benchmarks": detected_benchmarks,
    }


def _is_data_token(w: Tuple[Any, ...], line: List[Tuple[Any, ...]]) -> bool:
    """Helper to detect empirical numbers, dashes, and gains in visual text lines."""
    t = str(w[4]).strip()
    if t in ["–", "-", "—"]:
        return True
    if re.match(r"^\(\d+(?:\.\d+)?[Bb]\)$", t):
        return False
    if re.match(r"^\(\d+(?:,\d+)?\)$", t):
        return False
    if t.lower() == "none":
        return False
    w_idx = line.index(w)
    if w_idx >= 2 and str(line[w_idx - 2][4]).lower() == "t" and str(line[w_idx - 1][4]) == "=":
        return False
    if w_idx >= 1 and str(line[w_idx - 1][4]).lower().startswith("t="):
        return False
    return bool(re.match(r"^[-+]?\d+(?:\.\d+)?(?:\s*\([+-]?\d+(?:\.\d+)?[↑↓%]?\))?%?$", t))


def is_merged_table_candidate(
    raw_headers: List[str],
    raw_rows: List[List[Any]],
    line_count: int = 0,
) -> bool:
    """
    Determines if a table extracted by PyMuPDF find_tables() exhibits merged/collapsed
    rows or multiline text cells requiring word-position reconstruction.
    """
    if not raw_rows:
        return False
    for r in raw_rows:
        if not isinstance(r, list):
            continue
        for idx, c in enumerate(r):
            c_str = str(c or "").strip()
            if not c_str:
                continue
            words = c_str.split()
            num_tokens = [
                w for w in words
                if re.match(r"^[-+]?\d+(?:\.\d+)?(?:\s*\([+-]?\d+(?:\.\d+)?[↑↓%]?\))?%?$", w)
            ]
            if idx > 0 and (len(num_tokens) > 1 or (len(words) > 3 and len(num_tokens) >= 1)):
                return True
            if c_str.count("\n") >= 2:
                return True
            if len(re.findall(r"\(\d+(?:\.\d+)?[Bb]\)", c_str)) > 1:
                return True
            if _is_squashed_model(c_str):
                return True
    if line_count >= 5 and len(raw_rows) < line_count * 0.6:
        return True
    return False


def reconstruct_table_from_words(
    page: pymupdf.Page,
    bbox: Tuple[float, float, float, float],
    raw_headers: List[str],
    raw_rows: List[List[Any]],
    caption: str = "",
    page_number: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Reconstructs tables with merged multiline cells and spanning headers using
    PyMuPDF word coordinates (page.get_text('words')).

    Applies a strict confidence gate:
    (a) Every reconstructed row has exactly the expected column count.
    (b) Numeric columns contain valid numbers or standard missing value indicators.
    (c) Reconstructed row count >= original row count.

    Returns a dict with reconstructed headers, rows, row_groups, and normalized_records
    if the gate passes; returns None if reconstruction is ambiguous or fails the gate.
    """
    expanded_bbox = (bbox[0] - 2, bbox[1] - 4, bbox[2] + 2, bbox[3] + 4)
    words = page.get_text("words", clip=pymupdf.Rect(expanded_bbox))
    if not words or len(words) < 6:
        return None

    # Group words into visual text lines by y-coordinate
    sorted_words = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[Tuple[Any, ...]]] = []
    for w in sorted_words:
        placed = False
        for line in lines:
            line_y_avg = sum(x[1] for x in line) / len(line)
            line_h_avg = sum(x[3] - x[1] for x in line) / len(line)
            if abs(w[1] - line_y_avg) <= max(3.0, line_h_avg * 0.45):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    for line in lines:
        line.sort(key=lambda w: w[0])
    lines.sort(key=lambda line: min(w[1] for w in line))

    if not is_merged_table_candidate(raw_headers, raw_rows, line_count=len(lines)):
        return None

    # Identify data lines and data words
    data_lines = [ln for ln in lines if any(_is_data_token(w, ln) for w in ln)]
    if not data_lines or len(data_lines) < 2:
        return None

    data_words = []
    for ln in data_lines:
        for w in ln:
            if _is_data_token(w, ln):
                data_words.append(w)

    x_centers = sorted([(w[0] + w[2]) / 2 for w in data_words])
    clusters: List[List[float]] = []
    for xc in x_centers:
        if not clusters or xc - clusters[-1][-1] > 18.0:
            clusters.append([xc])
        else:
            clusters[-1].append(xc)

    clusters = [cl for cl in clusters if len(cl) >= 2]
    if len(clusters) < 1:
        return None

    num_cols = len(clusters)
    col_centers = [sum(cl) / len(cl) for cl in clusters]

    # Calculate column boundaries
    dividers = []
    for i in range(num_cols - 1):
        dividers.append((col_centers[i] + col_centers[i + 1]) / 2.0)

    # Label divider calculated strictly from data lines
    leftmost_data_x0 = min(min(w[0] for w in ln if _is_data_token(w, ln)) for ln in data_lines)
    label_words_on_data_lines = []
    for ln in data_lines:
        first_data_x0_in_line = min(w[0] for w in ln if _is_data_token(w, ln))
        for w in ln:
            if w[2] <= first_data_x0_in_line:
                label_words_on_data_lines.append(w)

    max_label_x1 = max(w[2] for w in label_words_on_data_lines) if label_words_on_data_lines else bbox[0]
    label_divider = (max_label_x1 + leftmost_data_x0) / 2.0
    col_divs = [label_divider] + dividers

    def get_col_idx(w: Tuple[Any, ...]) -> int:
        xc = (w[0] + w[2]) / 2
        for idx, d in enumerate(col_divs):
            if xc < d:
                return idx
        return len(col_divs)

    # Header block: ends before first section header or data line
    first_data_line_idx = lines.index(data_lines[0])
    section_kw = [
        "calibration", "model-based", "vanilla", "ours", "sft", "self-rag",
        "ablation", "without sft", "long-context", "methods", "rag-base", "#"
    ]
    header_end_idx = first_data_line_idx
    for idx in range(first_data_line_idx):
        line_txt = " ".join(str(w[4]) for w in lines[idx]).lower()
        if any(kw in line_txt for kw in section_kw):
            header_end_idx = idx
            break

    header_lines = lines[:header_end_idx]

    # Reconstruct headers
    composite_headers = ["" for _ in range(num_cols + 1)]
    if header_lines:
        col0_words = []
        for hl in header_lines:
            for w in hl:
                if get_col_idx(w) == 0:
                    col0_words.append(str(w[4]))
        composite_headers[0] = " ".join(col0_words).strip() or (raw_headers[0] if raw_headers else "Model")

        bottom_hl = header_lines[-1]
        sub_headers: List[List[str]] = [[] for _ in range(num_cols + 1)]
        for w in bottom_hl:
            c_i = get_col_idx(w)
            sub_headers[c_i].append(str(w[4]))

        top_hls = header_lines[:-1]

        # Spanning phrases across multiple numeric columns
        multi_col_phrases: List[Tuple[float, float, str]] = []
        for thl in top_hls:
            thl_sorted = sorted([w for w in thl if get_col_idx(w) > 0], key=lambda x: x[0])
            if not thl_sorted:
                continue
            cur_p = [thl_sorted[0]]
            for w in thl_sorted[1:]:
                if w[0] - cur_p[-1][2] <= 10.0:
                    cur_p.append(w)
                else:
                    multi_col_phrases.append((cur_p[0][0], cur_p[-1][2], " ".join(str(x[4]) for x in cur_p)))
                    cur_p = [w]
            multi_col_phrases.append((cur_p[0][0], cur_p[-1][2], " ".join(str(x[4]) for x in cur_p)))

        spanning_by_col: Dict[int, str] = {}
        for p_x0, p_x1, p_txt in multi_col_phrases:
            covered_cols = [c_idx + 1 for c_idx, cc in enumerate(col_centers) if p_x0 - 5 <= cc <= p_x1 + 5]
            if len(covered_cols) > 1:
                for c_i in covered_cols:
                    spanning_by_col[c_i] = p_txt

        # Also collect vertically stacked words in top lines per column
        col_top_stacked: Dict[int, List[str]] = {}
        for thl in top_hls:
            for w in thl:
                c_i = get_col_idx(w)
                if c_i > 0:
                    col_top_stacked.setdefault(c_i, []).append(str(w[4]))

        for col_i in range(1, num_cols + 1):
            sub_str = " ".join(sub_headers[col_i]).strip()
            top_str = ""
            if col_i in spanning_by_col:
                top_str = spanning_by_col[col_i]
            elif col_i in col_top_stacked:
                top_str = " ".join(col_top_stacked[col_i])

            if top_str and sub_str and top_str != sub_str:
                composite_headers[col_i] = f"{top_str} / {sub_str}"
            elif sub_str:
                composite_headers[col_i] = sub_str
            elif top_str:
                composite_headers[col_i] = top_str
            else:
                composite_headers[col_i] = f"Col_{col_i+1}"
    else:
        composite_headers = list(raw_headers)

    # Process body lines
    reconstructed_rows: List[List[str]] = []
    row_groups: List[str] = []
    current_group = ""
    current_model_prefix = ""

    for ln in lines[header_end_idx:]:
        line_data = [w for w in ln if _is_data_token(w, ln)]
        line_text = " ".join(str(w[4]) for w in ln)

        # Section header or non-data line
        if not line_data:
            lower_txt = line_text.lower()
            if any(kw in lower_txt for kw in ["based", "vanilla", "ours", "ta-are", "ablation", "baseline", "§", "sft", "methods", "rag-base", "#"]):
                current_group = line_text.strip().replace("#", "").strip()
                current_model_prefix = ""
            elif len(ln) <= 2 and get_col_idx(ln[0]) == 0:
                current_model_prefix = line_text.strip()
            continue

        # Data line
        col_buckets: List[List[str]] = [[] for _ in range(num_cols + 1)]
        for w in ln:
            c_i = get_col_idx(w)
            col_buckets[c_i].append(str(w[4]))

        col_strings = [" ".join(b).strip() for b in col_buckets]

        # Orphan single-column continuation: merge into preceding row with missing value in that column
        if not col_strings[0] and sum(1 for c in col_strings[1:] if c) == 1 and reconstructed_rows:
            for c_i in range(1, num_cols + 1):
                if col_strings[c_i] and not reconstructed_rows[-1][c_i]:
                    reconstructed_rows[-1][c_i] = col_strings[c_i]
            continue

        if current_model_prefix and (col_strings[0].startswith("t =") or col_strings[0].startswith("t=")):
            col_strings[0] = f"{current_model_prefix} {col_strings[0]}"

        reconstructed_rows.append(col_strings)
        row_groups.append(current_group)

    # Confidence Gate
    if len(reconstructed_rows) < 2 or len(composite_headers) != num_cols + 1:
        return None

    valid_num_pattern = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:\s*\([+-]?\d+(?:\.\d+)?[↑↓%]?\))?%?$")
    for r in reconstructed_rows:
        if len(r) != num_cols + 1:
            return None
        for val in r[1:]:
            val_clean = val.strip()
            if not val_clean or val_clean in ["–", "-", "—", "None", "N/A"]:
                continue
            if not valid_num_pattern.match(val_clean):
                return None

    if len(reconstructed_rows) < len(raw_rows):
        return None

    # Gate passed! Generate normalized empirical records if results table
    known_benchmarks = [
        "RetrievalQA", "HotpotQA", "2WikiMultiHopQA", "2WikiMQA", "MusiQue",
        "QASPER", "SQuAD", "GLUE", "SuperGLUE", "ImageNet", "FreshQA", "RealTimeQA"
    ]
    detected_dataset = "Benchmark Dataset"
    for bm in known_benchmarks:
        if bm.lower() in caption.lower():
            detected_dataset = bm
            break

    records = []
    for r_idx, r in enumerate(reconstructed_rows):
        model = r[0]
        group = row_groups[r_idx] if r_idx < len(row_groups) else ""
        if not model or model.lower() in ["average", "average gain", "averagegain", "total"]:
            continue
        for c_i in range(1, num_cols + 1):
            val = r[c_i]
            if not val or val in ["–", "-", "—"]:
                continue
            h_name = composite_headers[c_i]
            is_bm = any(bm.lower() == h_name.lower() or bm.lower() in h_name.lower() for bm in known_benchmarks)
            if is_bm and not any(m in h_name.lower() for m in ["retrieval", "match", "acc", "f1"]):
                ds = h_name
                met = "Score (%)" if "%" in caption else "Performance"
            else:
                ds = detected_dataset
                met = h_name

            records.append({
                "model": model,
                "group": group,
                "dataset": ds,
                "metric": met,
                "value": val,
                "page": page_number,
            })

    return {
        "headers": composite_headers,
        "rows": reconstructed_rows,
        "row_groups": row_groups,
        "normalized_records": records,
        "reliable": True,
        "note": f"Successfully reconstructed {len(reconstructed_rows)} rows from word positions with spanning headers and groups into {len(records)} structured results tuples.",
        "reconstruction_method": "words",
        "reconstruction_passed_gate": True,
    }


def normalize_table_dict(table_data: Any) -> Dict[str, Any]:
    """
    Normalizes a table dictionary ensuring consistent, canonical keys:
    - page_number (int)
    - table_index (int)
    - headers (List[str])
    - rows (List[List[Any]])
    - row_count (int)
    - col_count (int)
    - caption (str)
    - is_results_table (bool)
    - is_accepted (bool)
    - rejection_reason (Optional[str])
    - classification (str: 'results', 'hyperparameters', 'dataset_statistics', 'other')
    - normalized_records (List[Dict[str, Any]])
    - normalization_reliable (bool)
    - normalization_note (str)
    - raw_table (Dict[str, Any])
    - paper_id (Optional[str])
    - id (Optional[int or str])

    Safely computes row_count and col_count from rows/headers if missing.
    Gracefully handles malformed inputs without crashing.
    Canonical keys only (legacy aliases 'page' and 'header' are sanitized).
    """
    if not isinstance(table_data, dict):
        return {
            "page_number": 1,
            "table_index": 1,
            "headers": [],
            "rows": [],
            "row_count": 0,
            "col_count": 0,
            "caption": "",
            "is_results_table": False,
            "is_accepted": False,
            "rejection_reason": "Non-dictionary input",
            "classification": "other",
            "normalized_records": [],
            "normalization_reliable": False,
            "normalization_note": "Malformed table input",
            "raw_table": {},
        }

    # Extract page number
    page = table_data.get("page_number")
    if page is None:
        page = table_data.get("page", 1)
    try:
        page_num = int(page)
    except (ValueError, TypeError):
        page_num = 1

    # Extract table index
    try:
        t_idx = int(table_data.get("table_index", 1))
    except (ValueError, TypeError):
        t_idx = 1

    # Extract rows
    raw_rows = table_data.get("rows")
    if not isinstance(raw_rows, list):
        rows = []
    else:
        rows = []
        for r in raw_rows:
            if isinstance(r, list):
                rows.append([str(c) if c is not None else "" for c in r])
            elif r is not None:
                rows.append([str(r)])

    # Extract headers
    raw_headers = table_data.get("headers")
    if raw_headers is None:
        raw_headers = table_data.get("header")

    if isinstance(raw_headers, list):
        headers = [str(c).strip() if c is not None else f"Col_{i+1}" for i, c in enumerate(raw_headers)]
    elif isinstance(raw_headers, str) and raw_headers:
        headers = [raw_headers]
    else:
        headers = []

    # Compute row_count
    row_count = table_data.get("row_count")
    if row_count is None:
        row_count = len(rows)
    else:
        try:
            row_count = int(row_count)
        except (ValueError, TypeError):
            row_count = len(rows)

    # Compute col_count
    col_count = table_data.get("col_count")
    if col_count is None:
        if headers:
            col_count = len(headers)
        elif rows and isinstance(rows[0], list):
            col_count = max((len(r) for r in rows), default=0)
        else:
            col_count = 0
    else:
        try:
            col_count = int(col_count)
        except (ValueError, TypeError):
            col_count = len(headers) if headers else 0

    if not headers and col_count > 0:
        headers = [f"Col_{i+1}" for i in range(col_count)]

    caption = str(table_data.get("caption") or "")

    # Evaluation / acceptance
    is_accepted = table_data.get("is_accepted")
    rejection_reason = table_data.get("rejection_reason")
    if is_accepted is None:
        accepted_eval, reason_eval = evaluate_table_candidate(headers, rows, caption=caption)
        is_accepted = accepted_eval
        rejection_reason = reason_eval
    else:
        is_accepted = bool(is_accepted)

    # Classification
    classification = table_data.get("classification")
    if not classification or classification == "unclassified":
        classification = classify_table(headers, rows, caption)

    # is_results_table flag
    is_results = table_data.get("is_results_table")
    if is_results is None:
        is_results = (classification == "results")
    else:
        is_results = bool(is_results)

    # Normalized records & reconstruction metadata
    normalized_records = table_data.get("normalized_records")
    norm_reliable = table_data.get("normalization_reliable")
    norm_note = str(table_data.get("normalization_note") or "")
    reconstruction_method = str(table_data.get("reconstruction_method") or "none")
    reconstruction_passed_gate = bool(table_data.get("reconstruction_passed_gate", False))
    row_groups = table_data.get("row_groups") or []
    if not isinstance(row_groups, list):
        row_groups = []

    if normalized_records is None:
        if is_results and is_accepted:
            norm_res = normalize_results_records(
                headers, rows, page_number=page_num, caption=caption, row_groups=row_groups
            )
            normalized_records = norm_res["records"]
            norm_reliable = norm_res["reliable"]
            norm_note = norm_res["note"]
        else:
            normalized_records = []
            norm_reliable = False
            norm_note = "Non-results or rejected table; normalization skipped."

    raw_table = table_data.get("raw_table") or {
        "headers": headers,
        "rows": rows,
        "raw_headers": headers,
        "raw_rows": rows,
        "caption": caption,
    }

    normalized: Dict[str, Any] = {
        "page_number": page_num,
        "table_index": t_idx,
        "headers": headers,
        "rows": rows,
        "row_count": row_count,
        "col_count": col_count,
        "caption": caption,
        "is_results_table": is_results,
        "is_accepted": is_accepted,
        "rejection_reason": rejection_reason,
        "classification": classification,
        "normalized_records": normalized_records,
        "normalization_reliable": norm_reliable,
        "normalization_note": norm_note,
        "reconstruction_method": reconstruction_method,
        "reconstruction_passed_gate": reconstruction_passed_gate,
        "row_groups": row_groups,
        "raw_table": raw_table,
    }

    if "paper_id" in table_data:
        normalized["paper_id"] = str(table_data["paper_id"])
    if "id" in table_data:
        normalized["id"] = table_data["id"]

    return normalized


def extract_tables_from_pdf(
    pdf_path: str,
    paper_id: str,
    db: Optional[DatabaseManager] = None,
) -> List[Dict[str, Any]]:
    """
    Finds, filters, classifies, and normalizes tables across all pages of a research PDF.
    Persists accepted, well-formed tables to the database if db is provided.
    Preserves raw table extraction while recording rejection reasons for filtered candidates.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found for table extraction: {pdf_path}")
        return []

    accepted_tables: List[Dict[str, Any]] = []

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as err:
        logger.error(f"Failed to open PDF for table extraction ({pdf_path}): {err}")
        return []

    try:
        for page_index in range(len(doc)):
            page_num = page_index + 1
            page = doc[page_index]

            try:
                table_finder = page.find_tables()
            except Exception as e:
                logger.debug(f"Table detection not supported or failed on page {page_num}: {e}")
                continue

            if not table_finder.tables:
                continue

            for t_idx, table in enumerate(table_finder.tables):
                # Clean cell extraction via row cell bounding boxes to preserve intact decimals
                grid_rows = []
                if hasattr(table, "rows") and table.rows:
                    for r_obj in table.rows:
                        row_vals = []
                        for c in r_obj.cells:
                            if c:
                                w_list = page.get_text("words", clip=pymupdf.Rect(c))
                                txt = " ".join(w[4] for w in w_list)
                                row_vals.append(txt.strip())
                            else:
                                row_vals.append("")
                        grid_rows.append(row_vals)

                if not grid_rows:
                    raw_rows = table.extract() or []
                    for r in raw_rows:
                        grid_rows.append([str(c).strip() if c is not None else "" for c in r])

                if not grid_rows:
                    continue

                raw_headers = [str(c).strip() if c else f"Col_{i+1}" for i, c in enumerate(grid_rows[0])]
                raw_rows = grid_rows[1:] if len(grid_rows) > 1 else []

                # Extract bounding box and caption
                bbox = tuple(table.bbox)
                caption = extract_table_caption(page, bbox)

                # Attempt word-position reconstruction for merged tables
                recon = None
                reconstruction_method = "none"
                reconstruction_passed_gate = False
                row_groups: List[str] = []
                headers = list(raw_headers)
                cleaned_rows = list(raw_rows)

                if is_merged_table_candidate(raw_headers, raw_rows):
                    recon = reconstruct_table_from_words(
                        page=page,
                        bbox=bbox,
                        raw_headers=raw_headers,
                        raw_rows=raw_rows,
                        caption=caption,
                        page_number=page_num,
                    )
                    if recon is not None:
                        reconstruction_method = "words"
                        reconstruction_passed_gate = True
                        headers = recon["headers"]
                        cleaned_rows = recon["rows"]
                        row_groups = recon.get("row_groups", [])
                    else:
                        reconstruction_method = "words_refused"
                        reconstruction_passed_gate = False

                # Filter and evaluate
                is_accepted, rejection_reason = evaluate_table_candidate(
                    headers=headers,
                    rows=cleaned_rows,
                    bbox=bbox,
                    page_height=page.rect.height,
                    caption=caption,
                )

                classification = classify_table(headers, cleaned_rows, caption=caption)
                is_results = (classification == "results")

                norm_data: Dict[str, Any] = {
                    "records": [],
                    "reliable": False,
                    "note": "Rejected or non-results table",
                }
                if is_accepted and is_results:
                    if reconstruction_passed_gate and recon is not None:
                        norm_data = {
                            "records": recon["normalized_records"],
                            "reliable": recon["reliable"],
                            "note": recon["note"],
                        }
                    elif reconstruction_method == "words_refused":
                        norm_data = {
                            "records": [],
                            "reliable": False,
                            "note": "Table reconstruction refused by confidence gate; raw table preserved without guessing.",
                        }
                    else:
                        norm_data = normalize_results_records(
                            headers=headers,
                            rows=cleaned_rows,
                            page_number=page_num,
                            caption=caption,
                            row_groups=row_groups,
                        )

                table_data = normalize_table_dict({
                    "paper_id": paper_id,
                    "page_number": page_num,
                    "table_index": t_idx + 1,
                    "headers": headers,
                    "rows": cleaned_rows,
                    "row_count": len(cleaned_rows),
                    "col_count": len(headers),
                    "caption": caption,
                    "is_results_table": is_results,
                    "is_accepted": is_accepted,
                    "rejection_reason": rejection_reason,
                    "classification": classification,
                    "normalized_records": norm_data["records"],
                    "normalization_reliable": norm_data["reliable"],
                    "normalization_note": norm_data["note"],
                    "reconstruction_method": reconstruction_method,
                    "reconstruction_passed_gate": reconstruction_passed_gate,
                    "row_groups": row_groups,
                    "raw_table": {
                        "headers": raw_headers,
                        "rows": raw_rows,
                        "raw_headers": raw_headers,
                        "raw_rows": raw_rows,
                        "bbox": bbox,
                        "caption": caption,
                    },
                })

                if is_accepted:
                    accepted_tables.append(table_data)
                    if db is not None:
                        db.insert_results_table(
                            paper_id=paper_id,
                            page_number=page_num,
                            table_index=t_idx + 1,
                            table_json=json.dumps(table_data),
                        )
                else:
                    logger.debug(
                        f"Page {page_num} table {t_idx+1} rejected: {rejection_reason}"
                    )

        logger.info(f"Extracted {len(accepted_tables)} accepted tables for paper {paper_id}")
    finally:
        doc.close()

    return accepted_tables


def get_paper_results_tables(
    paper_id: str,
    db: DatabaseManager,
) -> List[Dict[str, Any]]:
    """
    Retrieves stored tables for a paper, decoding and normalizing their JSON payloads.
    Guarantees consistent keys: page_number, table_index, headers, rows, row_count, col_count,
    is_accepted, classification, normalized_records, etc.
    """
    rows = db.get_results_tables(paper_id)
    tables = []
    for r in rows:
        try:
            data = json.loads(r["table_json"])
            if not isinstance(data, dict):
                continue
            data["id"] = r.get("id")
            if not data.get("page_number"):
                data["page_number"] = r.get("page_number", 1)
            tables.append(normalize_table_dict(data))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error decoding table JSON for record {r.get('id')}: {e}")
            continue
    return tables
