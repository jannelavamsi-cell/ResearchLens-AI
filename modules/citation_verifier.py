"""
Citation verification engine for ResearchLens AI.
Splits answers into factual claims, maps inline citations [Paper N, p. X] to retrieved passages,
and evaluates claim grounding via lexical overlap and verification metrics.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from core.schemas import ClaimVerification, SearchResult

logger = logging.getLogger("researchlens.citations")

CITATION_REGEX = re.compile(
    r"\[(?:Paper\s*(\d+)(?:,\s*p(?:p)?\.?\s*(\d+(?:-\d+)?))?|p(?:p)?\.?\s*(\d+(?:-\d+)?))\]",
    re.IGNORECASE,
)


def split_claims(text: str) -> List[str]:
    """
    Splits an answer into discrete claims (sentences) while preserving citation tags
    and protecting abbreviations like 'p.' and 'pp.'.
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()

    # Protect citations during splitting by masking bracketed citation blocks
    citations_found = []

    def mask_match(m):
        idx = len(citations_found)
        citations_found.append(m.group(0))
        return f"__CIT_{idx}__"

    masked_text = CITATION_REGEX.sub(mask_match, clean_text)

    # Split by standard sentence delimiters (period, exclamation, question) followed by whitespace and capital letter
    sentence_pattern = re.compile(r"(?<!\bp)(?<!\bpp)(?<!\be\.g)(?<!\bi\.e)(?<=[.!?])\s+(?=[A-Z0-9_\-])")
    raw_sentences = sentence_pattern.split(masked_text)

    claims = []
    for s in raw_sentences:
        s_stripped = s.strip()
        lines = s_stripped.split("\n")
        for line in lines:
            line_clean = line.strip().lstrip("-*•0123456789. ")
            if len(line_clean) > 5:
                # Unmask citations
                for idx, orig_cit in enumerate(citations_found):
                    line_clean = line_clean.replace(f"__CIT_{idx}__", orig_cit)
                claims.append(line_clean)

    return claims if claims else [clean_text]


def extract_citations_from_claim(claim: str) -> List[Dict[str, Any]]:
    """
    Extracts all citation instances from a single claim string.
    Supports: [Paper 1, p. 5], [Paper 2, pp. 3-4], [Paper 1], [p. 8].
    """
    citations = []
    matches = CITATION_REGEX.finditer(claim)
    for m in matches:
        paper_num_str = m.group(1)
        page_num_str = m.group(2) or m.group(3)

        paper_num = int(paper_num_str) if paper_num_str else None
        page_num = 1
        if page_num_str:
            page_start = page_num_str.split("-")[0]
            try:
                page_num = int(page_start)
            except ValueError:
                page_num = 1

        citations.append({
            "raw": m.group(0),
            "paper_num": paper_num,
            "page_num": page_num,
        })
    return citations


def clean_claim_text_without_citations(claim: str) -> str:
    """Removes inline citation brackets and cleans punctuation whitespace."""
    text = CITATION_REGEX.sub("", claim)
    # Fix whitespace before punctuation (e.g. "word ." -> "word.")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_claims(
    claims: List[str],
    retrieved_chunks: List[SearchResult],
    paper_number_map: Optional[Dict[int, str]] = None,
) -> List[ClaimVerification]:
    """
    Verifies each claim against the retrieved grounding chunks:
    - Checks for explicit citations.
    - Evaluates lexical and fuzzy token overlap against cited chunks.
    - Classifies as 'supported', 'partially_supported', or 'unsupported'.
    """
    verifications: List[ClaimVerification] = []

    chunks_by_paper: Dict[str, List[SearchResult]] = {}
    for c in retrieved_chunks:
        chunks_by_paper.setdefault(c.paper_id, []).append(c)

    for claim in claims:
        citations = extract_citations_from_claim(claim)
        claim_content = clean_claim_text_without_citations(claim)

        # Skip verification for conversational wrappers or disclaimer sentences
        lower_claim = claim_content.lower()
        if any(prefix in lower_claim for prefix in [
            "the uploaded papers do not contain",
            "missing information",
            "in summary",
            "based on the provided documents",
            "here is the analysis",
        ]):
            verifications.append(
                ClaimVerification(
                    claim_text=claim,
                    citation=None,
                    is_supported=True,
                    support_level="supported",
                    reason="System or meta-disclaimer sentence",
                )
            )
            continue

        if not citations:
            verifications.append(
                ClaimVerification(
                    claim_text=claim,
                    citation=None,
                    is_supported=False,
                    support_level="unsupported",
                    reason="No citation provided for this claim",
                )
            )
            continue

        highest_score = 0.0
        best_snippet = None

        for cit in citations:
            target_paper_num = cit["paper_num"]
            target_page = cit["page_num"]

            candidate_chunks = []
            for chunk in retrieved_chunks:
                paper_matches = True
                if target_paper_num and paper_number_map:
                    mapped_id = paper_number_map.get(target_paper_num)
                    if mapped_id and chunk.paper_id != mapped_id:
                        paper_matches = False

                page_matches = abs(chunk.page_start - target_page) <= 1

                if paper_matches and page_matches:
                    candidate_chunks.append(chunk)

            if not candidate_chunks and target_paper_num and paper_number_map:
                mapped_id = paper_number_map.get(target_paper_num)
                if mapped_id:
                    candidate_chunks = chunks_by_paper.get(mapped_id, [])

            if not candidate_chunks:
                candidate_chunks = retrieved_chunks

            for chunk in candidate_chunks:
                score_sort = fuzz.token_sort_ratio(claim_content.lower(), chunk.text.lower())
                score_partial = fuzz.partial_ratio(claim_content.lower(), chunk.text.lower())
                combined_score = max(score_sort, score_partial)

                if combined_score > highest_score:
                    highest_score = combined_score
                    best_snippet = chunk.text[:250] + ("..." if len(chunk.text) > 250 else "")

        if highest_score >= 60.0:
            claim_supported = True
            support_level = "supported"
            reason = f"High textual alignment ({int(highest_score)}% similarity)"
        elif highest_score >= 45.0:
            claim_supported = True
            support_level = "partially_supported"
            reason = f"Moderate textual alignment ({int(highest_score)}% similarity)"
        else:
            claim_supported = False
            support_level = "unsupported"
            reason = f"Low textual alignment ({int(highest_score)}% similarity)"

        raw_cit_str = ", ".join([c["raw"] for c in citations])
        verifications.append(
            ClaimVerification(
                claim_text=claim,
                citation=raw_cit_str,
                is_supported=claim_supported,
                support_level=support_level,
                evidence_snippet=best_snippet,
                reason=reason,
            )
        )

    return verifications


def compute_support_score(verifications: List[ClaimVerification]) -> float:
    """
    Computes overall proportion of verified claims:
    Supported = 1.0, Partially Supported = 0.5, Unsupported = 0.0.
    """
    if not verifications:
        return 1.0

    total_weight = 0.0
    for v in verifications:
        if v.support_level == "supported":
            total_weight += 1.0
        elif v.support_level == "partially_supported":
            total_weight += 0.5

    return round(total_weight / len(verifications), 2)
