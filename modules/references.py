"""
Bibliographic reference formatting module for ResearchLens AI.
Formats paper metadata into IEEE, APA, and BibTeX citation strings.
"""

import re
from typing import Any, Dict, List, Optional


def clean_string(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_authors_list(authors_str: Optional[str]) -> List[str]:
    """Splits an author string into individual author names."""
    if not authors_str:
        return ["Unknown Author"]
    # Handle separators: and, comma, semicolon
    authors_str = authors_str.replace(";", ",")
    raw_list = re.split(r",\s*|\s+and\s+", authors_str, flags=re.IGNORECASE)
    cleaned = [clean_string(a) for a in raw_list if clean_string(a)]
    return cleaned if cleaned else ["Unknown Author"]


def format_ieee_authors(authors: List[str]) -> str:
    """Formats authors into IEEE style: 'A. Smith and B. Jones' or 'A. Smith et al.'"""
    if not authors or authors == ["Unknown Author"]:
        return "Unknown Author"

    formatted = []
    for a in authors:
        parts = a.split()
        if len(parts) >= 2:
            initials = "".join([f"{p[0]}." for p in parts[:-1]])
            formatted.append(f"{initials} {parts[-1]}")
        else:
            formatted.append(a)

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    elif len(formatted) <= 4:
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"
    else:
        return f"{formatted[0]} et al."


def format_apa_authors(authors: List[str]) -> str:
    """Formats authors into APA style: 'Smith, A., & Jones, B.'"""
    if not authors or authors == ["Unknown Author"]:
        return "Unknown Author"

    formatted = []
    for a in authors:
        parts = a.split()
        if len(parts) >= 2:
            initials = "".join([f" {p[0]}." for p in parts[:-1]])
            formatted.append(f"{parts[-1]},{initials}")
        else:
            formatted.append(a)

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    elif len(formatted) <= 6:
        return f"{', '.join(formatted[:-1])}, & {formatted[-1]}"
    else:
        return f"{formatted[0]}, et al."


def format_ieee_reference(paper: Dict[str, Any], index: int = 1) -> str:
    """Formats a paper metadata dict into an IEEE citation string."""
    authors = parse_authors_list(paper.get("authors"))
    authors_ieee = format_ieee_authors(authors)
    title = clean_string(paper.get("title") or paper.get("filename") or "Untitled Paper")
    venue = clean_string(paper.get("venue"))
    year = paper.get("year") or "n.d."
    doi = clean_string(paper.get("doi"))

    parts = [f"[{index}]", f'{authors_ieee}, "{title},"', f"in {venue}," if venue else "", f"{year}."]
    if doi:
        parts.append(f"DOI: {doi}.")

    return " ".join([p for p in parts if p])


def format_apa_reference(paper: Dict[str, Any]) -> str:
    """Formats a paper metadata dict into an APA citation string."""
    authors = parse_authors_list(paper.get("authors"))
    authors_apa = format_apa_authors(authors)
    title = clean_string(paper.get("title") or paper.get("filename") or "Untitled Paper")
    venue = clean_string(paper.get("venue"))
    year = paper.get("year") or "n.d."
    doi = clean_string(paper.get("doi"))

    ref = f"{authors_apa} ({year}). {title}."
    if venue:
        ref += f" {venue}."
    if doi:
        ref += f" https://doi.org/{doi}"
    return ref


def format_bibtex_entry(paper: Dict[str, Any]) -> str:
    """Formats paper metadata into a standard BibTeX entry."""
    authors_list = parse_authors_list(paper.get("authors"))
    first_author_surname = authors_list[0].split()[-1].lower() if authors_list else "author"
    first_author_surname = re.sub(r"\W+", "", first_author_surname)

    year = str(paper.get("year") or "2024")
    cite_key = f"{first_author_surname}{year}"

    title = clean_string(paper.get("title") or paper.get("filename") or "Untitled")
    authors_bibtex = " and ".join(authors_list)
    venue = clean_string(paper.get("venue") or "arXiv preprint")
    doi = clean_string(paper.get("doi") or "")

    lines = [
        f"@article{{{cite_key},",
        f"  title     = {{{title}}},",
        f"  author    = {{{authors_bibtex}}},",
        f"  journal   = {{{venue}}},",
        f"  year      = {{{year}}},",
    ]
    if doi:
        lines.append(f"  doi       = {{{doi}}},")
    lines.append("}")

    return "\n".join(lines)


def generate_references_list(
    papers: List[Dict[str, Any]],
    style: str = "ieee",
) -> List[str]:
    """Generates a complete list of formatted reference strings for all provided papers."""
    style = style.lower()
    refs = []
    for idx, p in enumerate(papers, 1):
        if style == "ieee":
            refs.append(format_ieee_reference(p, index=idx))
        elif style == "apa":
            refs.append(format_apa_reference(p))
        elif style == "bibtex":
            refs.append(format_bibtex_entry(p))
        else:
            refs.append(format_ieee_reference(p, index=idx))
    return refs
