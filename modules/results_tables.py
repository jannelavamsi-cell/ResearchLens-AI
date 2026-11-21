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


def is_likely_results_table(headers: List[str], rows: List[List[str]]) -> bool:
    """
    Checks if an extracted table appears to be an empirical results / evaluation table
    based on metric and model keywords in headers and numerical entries.
    """
    keywords = [
        "model", "method", "dataset", "accuracy", "acc", "f1", "precision", "recall",
        "bleu", "rouge", "auc", "map", "ndcg", "mrr", "loss", "score", "error", "latency",
        "parameters", "params", "time", "%", "std", "baseline", "ours", "proposed"
    ]
    header_str = " ".join([str(h).lower() for h in headers if h])

    # Check if header contains any relevant keywords
    has_keyword = any(k in header_str for k in keywords)

    # Check if table has numerical content
    has_numbers = False
    for row in rows:
        for cell in row:
            if cell and any(ch.isdigit() for ch in str(cell)):
                has_numbers = True
                break
        if has_numbers:
            break

    return has_keyword or has_numbers


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
            "page": 1,
            "table_index": 1,
            "headers": [],
            "header": [],
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
        "page": page_num,
        "table_index": t_idx,
        "headers": headers,
        "header": headers,
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
