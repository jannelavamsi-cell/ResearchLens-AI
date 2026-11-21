"""
Results table extraction module for ResearchLens AI.
Extracts tabular data from research papers using PyMuPDF find_tables(),
identifies experimental results tables, and normalizes them into structured records.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pymupdf

from database.db_manager import DatabaseManager

logger = logging.getLogger("researchlens.tables")


def is_valid_table(headers: List[str], rows: List[List[Any]]) -> bool:
    """
    Checks if tabular data has a meaningful structural shape.
    Rejects single-cell artifacts, 1-row metadata blocks, and empty grids.
    """
    if not rows or not isinstance(rows, list):
        return False
    # At least one non-empty row
    valid_rows = [r for r in rows if isinstance(r, list) and any(str(c).strip() for c in r)]
    if not valid_rows:
        return False
    # Total cells across headers and valid rows must be >= 4 to constitute a genuine table
    total_cells = len(headers) + sum(len(r) for r in valid_rows)
    return total_cells >= 4


def is_likely_results_table(headers: List[str], rows: List[List[str]]) -> bool:
    """
    Checks if an extracted table appears to be an empirical results / evaluation table
    based on metric and model keywords in headers and numerical entries.
    Filters out layout blocks, text-only glossaries, and single-cell artifacts.
    """
    if not is_valid_table(headers, rows):
        return False

    metric_keywords = [
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge", "auc",
        "map", "ndcg", "mrr", "loss", "error", "latency", "throughput", "p@1",
        "em", "exact match", "perplexity", "top-1", "top-5", "runtime", "%", "score"
    ]
    model_keywords = [
        "model", "method", "approach", "baseline", "ours", "proposed", "system",
        "architecture", "variant", "framework"
    ]
    header_str = " ".join([str(h).lower() for h in headers if h])

    has_metric_header = any(k in header_str for k in metric_keywords)
    has_model_header = any(k in header_str for k in model_keywords)

    # Check for numerical density in table body
    numeric_cell_count = 0
    total_cell_count = 0
    for row in rows:
        if isinstance(row, list):
            for cell in row:
                cell_str = str(cell).strip()
                if cell_str:
                    total_cell_count += 1
                    # Match floating point, integer, or percentage
                    if any(ch.isdigit() for ch in cell_str):
                        numeric_cell_count += 1

    if total_cell_count == 0:
        return False

    numeric_ratio = numeric_cell_count / total_cell_count

    # Empirical results tables generally contain both metric/model indicators and significant numbers
    if (has_metric_header or has_model_header) and numeric_cell_count >= 2:
        return True
    if numeric_ratio >= 0.35 and numeric_cell_count >= 3:
        return True

    return False


def parse_table_semantics(
    headers: List[str],
    rows: List[List[Any]],
    caption: str = "",
) -> Dict[str, Any]:
    """
    Heuristically extracts experimental semantics from a results table:
    - metric_columns: columns measuring performance (e.g. Accuracy, F1)
    - model_column: column identifying architectures / approaches
    - detected_benchmarks: datasets or benchmarks identified in headers or caption
    """
    metric_terms = {
        "accuracy", "acc", "f1", "precision", "recall", "bleu", "rouge",
        "auc", "map", "ndcg", "mrr", "loss", "error", "latency", "em", "score"
    }
    model_terms = {"model", "method", "approach", "baseline", "system", "architecture"}

    metric_cols = []
    model_col = None

    for idx, h in enumerate(headers):
        h_clean = str(h).lower().strip()
        if any(m in h_clean for m in metric_terms):
            metric_cols.append({"index": idx, "name": h})
        if model_col is None and any(m in h_clean for m in model_terms):
            model_col = {"index": idx, "name": h}

    # Default model column to index 0 if not explicitly named and rows have cells
    if model_col is None and headers:
        model_col = {"index": 0, "name": headers[0]}

    # Detect benchmark mentions in caption or headers
    known_benchmarks = [
        "squad", "glue", "superglue", "imagenet", "retrievalqa", "mmlongbench",
        "hotpotqa", "triviaqa", "ms-marco", "msmarco", "pubmed", "nq", "natural questions"
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
    - page (int alias for page_number)
    - table_index (int)
    - headers (List[str])
    - header (List[str] alias for headers)
    - rows (List[List[Any]])
    - row_count (int)
    - col_count (int)
    - caption (str)
    - is_results_table (bool)
    - paper_id (Optional[str])
    - id (Optional[int or str])

    Safely computes row_count and col_count from the rows/headers if missing.
    Gracefully handles malformed inputs (non-dict, None, invalid types) without crashing.
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

    # Auto-generate headers if missing but rows have columns
    if not headers and col_count > 0:
        headers = [f"Col_{i+1}" for i in range(col_count)]

    caption = str(table_data.get("caption") or "")

    # Results table classification
    is_results = table_data.get("is_results_table")
    if is_results is None:
        is_results = is_likely_results_table(headers, rows)
    else:
        is_results = bool(is_results)

    normalized: Dict[str, Any] = {
        "page_number": page_num,
        "table_index": t_idx,
        "headers": headers,
        "rows": rows,
        "row_count": row_count,
        "col_count": col_count,
        "caption": caption,
        "is_results_table": is_results,
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
    Finds and extracts tables across all pages of a research paper PDF.
    Persists valid tables to the results_tables table if db is provided.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found for table extraction: {pdf_path}")
        return []

    extracted_tables: List[Dict[str, Any]] = []

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
                if not raw_rows or len(raw_rows) < 2:
                    continue

                headers = [str(c).strip() if c else f"Col_{i}" for i, c in enumerate(raw_rows[0])]
                data_rows = raw_rows[1:]

                # Clean cell text
                cleaned_rows = []
                for row in data_rows:
                    cleaned_rows.append([str(c).strip() if c is not None else "" for c in row])

                is_results = is_likely_results_table(headers, cleaned_rows)

                table_data = normalize_table_dict(
                    {
                        "paper_id": paper_id,
                        "page_number": page_num,
                        "table_index": t_idx + 1,
                        "headers": headers,
                        "header": headers,
                        "rows": cleaned_rows,
                        "is_results_table": is_results,
                        "row_count": len(cleaned_rows),
                        "col_count": len(headers),
                        "caption": "",
                    }
                )

                extracted_tables.append(table_data)

                if db is not None:
                    db.insert_results_table(
                        paper_id=paper_id,
                        page_number=page_num,
                        table_index=t_idx + 1,
                        table_json=json.dumps(table_data),
                    )

        logger.info(f"Extracted {len(extracted_tables)} tables for paper {paper_id}")
    finally:
        doc.close()

    return extracted_tables


def get_paper_results_tables(
    paper_id: str,
    db: DatabaseManager,
) -> List[Dict[str, Any]]:
    """
    Retrieves stored tables for a paper, decoding and normalizing their JSON payloads.
    Guarantees consistent keys: page_number, table_index, headers, rows, row_count, col_count, etc.
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
            if not data.get("table_index"):
                data["table_index"] = r.get("table_index", 1)
            tables.append(normalize_table_dict(data))
        except Exception as err:
            logger.warning(f"Failed to load/normalize table for paper {paper_id}: {err}")
            continue
    return tables
