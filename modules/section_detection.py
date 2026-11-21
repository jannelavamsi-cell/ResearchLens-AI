"""
Section detection engine for ResearchLens AI.
Combines academic heading regex patterns and PyMuPDF font/bold layout heuristics
to partition scientific papers into standard research sections.
"""

import re
from typing import Dict, List, Optional, Tuple

from core.schemas import SectionSpan

# Standard canonical academic section categories
CANONICAL_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Methodology",
    "Experiments",
    "Results",
    "Discussion",
    "Limitations",
    "Future Work",
    "Conclusion",
    "References",
    "Appendix",
]

# Regex patterns mapping variations to canonical names
SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "Abstract": [
        re.compile(r"^(abstract)\b", re.IGNORECASE),
    ],
    "Introduction": [
        re.compile(r"^(\d+[\.\s]+)?(introduction)\b", re.IGNORECASE),
        re.compile(r"^(\d+[\.\s]+)?(overview)\b", re.IGNORECASE),
    ],
    "Related Work": [
        re.compile(
            r"^(\d+[\.\s]+)?(related\s+work|prior\s+work|literature\s+review)\b", re.IGNORECASE
        ),
        re.compile(r"^(\d+[\.\s]+)?(background)\b", re.IGNORECASE),
    ],
    "Methodology": [
        re.compile(
            r"^(\d+[\.\s]+)?(methodology|methods|proposed\s+method|our\s+approach|architecture|system\s+design|model)\b",
            re.IGNORECASE,
        ),
    ],
    "Experiments": [
        re.compile(
            r"^(\d+[\.\s]+)?(experiments|experimental\s+setup|evaluation|empirical\s+evaluation)\b",
            re.IGNORECASE,
        ),
    ],
    "Results": [
        re.compile(
            r"^(\d+[\.\s]+)?(results|experimental\s+results|findings|quantitative\s+results)\b",
            re.IGNORECASE,
        ),
    ],
    "Discussion": [
        re.compile(r"^(\d+[\.\s]+)?(discussion|analysis)\b", re.IGNORECASE),
    ],
    "Limitations": [
        re.compile(r"^(\d+[\.\s]+)?(limitations|threats\s+to\s+validity)\b", re.IGNORECASE),
    ],
    "Future Work": [
        re.compile(r"^(\d+[\.\s]+)?(future\s+work|future\s+directions)\b", re.IGNORECASE),
    ],
    "Conclusion": [
        re.compile(r"^(\d+[\.\s]+)?(conclusion|concluding\s+remarks|summary)\b", re.IGNORECASE),
    ],
    "References": [
        re.compile(r"^(references|bibliography|literature\s+cited)\b", re.IGNORECASE),
    ],
    "Appendix": [
        re.compile(r"^(\d+[\.\s]+)?(appendix|appendices)\b", re.IGNORECASE),
    ],
}


def classify_heading_line(line: str) -> Optional[str]:
    """
    Checks if a single text line corresponds to a standard academic heading.
    Returns canonical section name or None.
    """
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 60:
        return None

    # Academic headings are capitalized or numbered; reject purely lowercase lines
    if clean_line.islower() and not re.match(r"^\d", clean_line):
        return None

    for canonical, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if pat.match(clean_line):
                return canonical
    return None


def detect_sections_from_pages(pages: List[Tuple[int, str]]) -> List[SectionSpan]:
    """
    Scans sequential page texts to determine section boundaries.
    pages: List of (page_number, text) tuples.
    Returns a sorted list of SectionSpan objects with page_start and page_end.
    """
    detected_headers: List[Tuple[str, int]] = []  # (canonical_name, page_number)
    seen_canonical = set()
    total_pages = pages[-1][0] if pages else 1

    for page_num, text in pages:
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            canonical = classify_heading_line(line_str)
            if canonical and canonical not in seen_canonical:
                # Disallow References false positives on early pages of multi-page papers
                if canonical == "References" and total_pages >= 4 and page_num <= 2:
                    continue
                detected_headers.append((canonical, page_num))
                seen_canonical.add(canonical)

    if not detected_headers:
        # Fallback if no explicit headings matched
        total_pages = pages[-1][0] if pages else 1
        return [SectionSpan(section_name="Main Text", page_start=1, page_end=total_pages)]

    # Convert detected heading start points into continuous SectionSpan ranges
    spans: List[SectionSpan] = []
    total_pages = pages[-1][0] if pages else 1

    for idx, (section_name, start_page) in enumerate(detected_headers):
        if idx + 1 < len(detected_headers):
            next_start = detected_headers[idx + 1][1]
            end_page = max(start_page, next_start if next_start == start_page else next_start)
        else:
            end_page = total_pages

        spans.append(
            SectionSpan(
                section_name=section_name,
                page_start=start_page,
                page_end=end_page,
            )
        )

    return spans
