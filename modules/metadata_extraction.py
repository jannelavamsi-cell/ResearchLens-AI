"""
Bibliographic metadata extraction engine for ResearchLens AI.
Extracts title, authors, publication year, venue, and DOI from PDF metadata
and first-page layout heuristics, with optional Crossref API enrichment.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pymupdf
import requests

from core.logging_config import get_logger
from core.schemas import PaperMetadata

logger = get_logger("metadata_extraction")

DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


class MetadataExtractor:
    """Extracts bibliographic information from academic PDFs."""

    def __init__(self, enable_online_enrichment: bool = True) -> None:
        self.enable_online_enrichment = enable_online_enrichment

    def _extract_doi_and_year(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        """Searches text for DOI strings and publication years."""
        doi = None
        year = None

        doi_match = DOI_PATTERN.search(text)
        if doi_match:
            # Strip trailing punctuation if present
            raw_doi = doi_match.group(1).rstrip(".,;)")
            doi = raw_doi

        year_matches = YEAR_PATTERN.findall(text[:2000])  # Search first 2000 chars
        if year_matches:
            # Pick the most plausible recent year
            valid_years = [int(y) for y in year_matches if 1950 <= int(y) <= 2030]
            if valid_years:
                year = max(valid_years)

        return doi, year

    def _extract_first_page_heuristics(
        self, doc: pymupdf.Document
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts title and authors from first page by inspecting font sizes.
        Title is usually the block with largest font size.
        """
        if len(doc) == 0:
            return None, None

        page = doc[0]
        blocks = page.get_text("dict")["blocks"]

        title_candidate = None
        author_candidate = None

        text_spans = []
        for b in blocks:
            if b.get("type") == 0:  # text block
                for line in b.get("lines", []):
                    line_spans = line.get("spans", [])
                    if line_spans:
                        span_text = " ".join(s.get("text", "").strip() for s in line_spans)
                        avg_size = max(s.get("size", 0.0) for s in line_spans)
                        clean_span = span_text.strip()
                        # Ignore margin stamps, arxiv identifiers, and publication headers
                        if clean_span and not re.search(
                            r"^(arxiv:\d|proceedings of|conference on|volume \d|issn|isbn|doi:)",
                            clean_span,
                            re.I,
                        ):
                            text_spans.append((avg_size, clean_span))

        if not text_spans:
            return None, None

        # Sort spans by font size descending to find title
        text_spans_by_size = sorted(text_spans, key=lambda x: x[0], reverse=True)
        title_parts = []
        if text_spans_by_size:
            max_size = text_spans_by_size[0][0]
            # Concatenate spans with the largest font size
            title_parts = [t for size, t in text_spans if abs(size - max_size) < 1.0]
            title_candidate = " ".join(title_parts)

        # Author heuristic: look at spans appearing after title with medium font size
        title_found = False
        author_lines = []
        for _size, t in text_spans:
            if not title_found:
                if (
                    t == title_candidate
                    or (title_parts and any(p in t or t in p for p in title_parts))
                ):
                    title_found = True
                continue
            # Do not collect parts of the title as authors
            if title_parts and any(p == t or t in p for p in title_parts):
                continue
            if title_found and len(author_lines) < 3:
                # Exclude obvious non-author lines, emails, and section headers
                if re.search(r"(abstract|introduction|doi:|http|@|university|institute|lab)", t, re.I):
                    break
                # Clean author line of footnote markers and numbers
                clean_author = re.sub(r"[\d\*†‡§\^]", "", t).strip()
                clean_author = re.sub(r"\s+", " ", clean_author)
                if clean_author and len(clean_author) > 2:
                    author_lines.append(clean_author)
            elif len(author_lines) >= 3:
                break

        if author_lines:
            author_candidate = ", ".join(author_lines)

        return title_candidate, author_candidate

    def enrich_via_crossref(self, doi: str) -> Dict[str, Any]:
        """Queries Crossref API to enrich paper metadata (offline-safe)."""
        if not self.enable_online_enrichment or not doi:
            return {}

        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "ResearchLensAI/1.0 (mailto:researchlens@academic.edu)"}

        try:
            res = requests.get(url, headers=headers, timeout=3.0)
            if res.status_code == 200:
                item = res.json().get("message", {})
                title = item.get("title", [None])[0]
                authors_list = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item.get("author", [])
                ]
                authors = ", ".join(authors_list) if authors_list else None
                year = None
                pub_parts = item.get("published", {}).get("date-parts", [[]])[0]
                if pub_parts:
                    year = pub_parts[0]
                venue = item.get("container-title", [None])[0]

                return {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": doi,
                }
        except Exception as e:
            logger.debug(f"Crossref lookup failed for DOI {doi}: {e}")

        return {}

    def extract_metadata(self, file_path: str) -> PaperMetadata:
        """Extracts complete bibliographic metadata from a PDF file."""
        path = Path(file_path)
        doc = pymupdf.open(str(path))

        meta = doc.metadata or {}
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""

        # Step 1: DOI & Year from text
        doi, year = self._extract_doi_and_year(first_page_text)

        # Step 2: Title and Authors from layout heuristics
        layout_title, layout_authors = self._extract_first_page_heuristics(doc)

        # Step 3: Check embedded PDF metadata fallback
        pdf_title = meta.get("title", "").strip()
        pdf_author = meta.get("author", "").strip()

        # Prefer layout heuristics if PDF title is generic (e.g. filename or untrusted)
        final_title = layout_title or pdf_title or path.stem.replace("_", " ").title()
        final_authors = layout_authors or pdf_author or "Unknown Authors"
        final_venue = None

        # Step 4: Optional Crossref enrichment
        if doi and self.enable_online_enrichment:
            crossref_data = self.enrich_via_crossref(doi)
            if crossref_data:
                final_title = crossref_data.get("title") or final_title
                final_authors = crossref_data.get("authors") or final_authors
                year = crossref_data.get("year") or year
                final_venue = crossref_data.get("venue") or final_venue

        doc.close()

        return PaperMetadata(
            title=final_title,
            authors=final_authors,
            year=year,
            venue=final_venue,
            doi=doi,
        )
