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
                lines = s.split("\n")
                has_multiline_list = len(lines) >= 3 and all(len(ln.strip()) < 50 for ln in lines if ln.strip())
                if not has_multiline_list:
                    if len(s) > MAX_CELL_SENTENCE_LENGTH:
                        narrative_sentences += 1
                    elif len(s) > 75 and (s.count(". ") >= 1 or s.endswith(".")) and not any(ch.isdigit() for ch in s):
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
    if is_ref_sec or bib_citation_matches >= 3:
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

    num_ratio = numeric_cells / total_cells if total_cells else 0.0

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

    # 3. Results tables: must feature numbers AND metric indicators or high numeric density
    metric_terms = [
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge", "auc",
        "map", "ndcg", "mrr", "loss", "error", "latency", "throughput", "p@1",
        "em", "exact match", "perplexity", "score", "match", "results (%)",
        "overall performance", "ablation", "averagegain", "relative gain"
    ]
    has_metric = any(_matches_term(t, text_corpus) for t in metric_terms)
    model_terms = ["model", "method", "approach", "baseline", "system", "architecture"]
    has_model = any(_matches_term(m, text_corpus) for m in model_terms)

    # An empirical results table requires numbers
    if (has_metric or has_model) and numeric_cells >= 2:
        return "results"
    if has_metric and numeric_cells >= 1:
        return "results"
    if num_ratio >= 0.35 and numeric_cells >= 3:
        return "results"

    return "other"


def normalize_results_records(
    headers: List[str],
    rows: List[List[Any]],
    page_number: int = 1,
    caption: str = "",
) -> Dict[str, Any]:
    """
    Converts empirical results tables into structured records:
    [{"model": str, "dataset": str, "metric": str, "value": str, "page": int}, ...]

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

    for row in rows:
        if not isinstance(row, list) or len(row) <= model_col_idx:
            continue

        raw_model = str(row[model_col_idx]).strip()
        if not raw_model:
            continue

        # Check for multiline squashed rows
        model_lines = [ln.strip() for ln in raw_model.split("\n") if ln.strip()]

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
                for ht in header_types:
                    c_idx = ht["col_idx"]
                    val = col_lines[c_idx][item_idx]
                    if ht["type"] == "dataset":
                        ds_name = ht["name"]
                        met_name = "Score (%)" if "%" in caption else "Performance"
                    else:
                        ds_name = detected_dataset
                        met_name = ht["name"]

                    records.append({
                        "model": m_name,
                        "dataset": ds_name,
                        "metric": met_name,
                        "value": val,
                        "page": page_number,
                    })
        else:
            m_name = model_lines[0]
            for ht in header_types:
                c_idx = ht["col_idx"]
                val = str(row[c_idx]).strip() if c_idx < len(row) and row[c_idx] is not None else ""
                if not val:
                    continue

                if ht["type"] == "dataset":
                    ds_name = ht["name"]
                    met_name = "Score (%)" if "%" in caption else "Performance"
                else:
                    ds_name = detected_dataset
                    met_name = ht["name"]

                records.append({
                    "model": m_name,
                    "dataset": ds_name,
                    "metric": met_name,
                    "value": val,
                    "page": page_number,
                })

    if not records or (unreliable_detected and len(records) < 2):
        return {
            "reliable": False,
            "records": records,
            "note": "Irregular row/column alignment prevented unambiguous decomposition; raw table preserved.",
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
        accepted_eval, reason_eval = evaluate_table_candidate(headers, rows)
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

    # Normalized records
    normalized_records = table_data.get("normalized_records")
    norm_reliable = table_data.get("normalization_reliable")
    norm_note = str(table_data.get("normalization_note") or "")

    if normalized_records is None:
        if is_results and is_accepted:
            norm_res = normalize_results_records(headers, rows, page_number=page_num, caption=caption)
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
                raw_rows = table.extract()
                if not raw_rows:
                    continue

                headers = [str(c).strip() if c else f"Col_{i+1}" for i, c in enumerate(raw_rows[0])]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

                cleaned_rows = []
                for row in data_rows:
                    cleaned_rows.append([str(c).strip() if c is not None else "" for c in row])

                # Extract bounding box and caption
                bbox = tuple(table.bbox)
                caption = extract_table_caption(page, bbox)

                # Filter and evaluate
                is_accepted, rejection_reason = evaluate_table_candidate(
                    headers=headers,
                    rows=cleaned_rows,
                    bbox=bbox,
                    page_height=page.rect.height,
                )

                classification = classify_table(headers, cleaned_rows, caption=caption)
                is_results = (classification == "results")

                norm_data: Dict[str, Any] = {
                    "records": [],
                    "reliable": False,
                    "note": "Rejected or non-results table",
                }
                if is_accepted and is_results:
                    norm_data = normalize_results_records(
                        headers=headers,
                        rows=cleaned_rows,
                        page_number=page_num,
                        caption=caption,
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
                    "raw_table": {
                        "raw_headers": headers,
                        "raw_rows": cleaned_rows,
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
