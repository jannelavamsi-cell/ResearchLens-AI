"""
Text preprocessing and cleaning utilities for ResearchLens AI.
Handles dehyphenation across line breaks, unicode ligature normalization,
whitespace consolidation, and cross-page header/footer removal.
"""

import re
from typing import List, Set


def normalize_ligatures(text: str) -> str:
    """Replaces standard Unicode ligatures with standard ASCII characters."""
    ligature_map = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
        "–": "-",  # en-dash
        "—": "-",  # em-dash
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for lig, rep in ligature_map.items():
        text = text.replace(lig, rep)
    return text


def fix_hyphenation(text: str) -> str:
    """
    Reconstructs words hyphenated across line breaks.
    Example: 'trans-\\nformer' -> 'transformer'
    Example: 'algo-\\n rithm' -> 'algorithm'
    Preserves intentional hyphens like 'state-of-the-art'.
    """
    # Regex: lowercase letter + hyphen + newline (and optional whitespace) + lowercase letter
    pattern = r"([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)"
    return re.sub(pattern, r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Normalizes excessive horizontal spaces while preserving logical paragraph breaks."""
    # Normalize multiple horizontal spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse three or more newlines to double newlines (paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_repeated_headers_footers(page_texts: List[str], threshold: float = 0.6) -> Set[str]:
    """
    Identifies recurring header and footer lines that appear on more than
    threshold percentage of pages (e.g. journal names, running headers, page numbers).
    """
    if len(page_texts) < 3:
        return set()

    line_counts = {}
    total_pages = len(page_texts)

    for text in page_texts:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Only examine first 3 lines (headers) and last 3 lines (footers)
        candidates = set(lines[:3] + lines[-3:])
        for line in candidates:
            # Skip very short lines or pure numbers (handled separately)
            if len(line) >= 4:
                line_counts[line] = line_counts.get(line, 0) + 1

    repeated = {line for line, count in line_counts.items() if (count / total_pages) >= threshold}
    return repeated


def clean_page_text(text: str, repeated_headers: Set[str] = None) -> str:
    """Applies the full cleaning pipeline to raw page text."""
    if not text:
        return ""

    text = normalize_ligatures(text)
    text = fix_hyphenation(text)

    if repeated_headers:
        lines = text.split("\n")
        cleaned_lines = [line for line in lines if line.strip() not in repeated_headers]
        text = "\n".join(cleaned_lines)

    text = normalize_whitespace(text)
    return text
