"""
Structured research extraction module for ResearchLens AI.
Extracts 9 core academic components with physical page citations and evidence quotes,
and verifies quotes against page text using fuzzy matching.
"""

import json
import logging
import re
from typing import Dict, List

import spacy
from rapidfuzz import fuzz
from spacy.matcher import Matcher

from core.llm import get_llm_client
from core.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from core.schemas import ExtractedField, StructuredPaperExtraction
from database.db_manager import DatabaseManager

logger = logging.getLogger("researchlens.extraction")

_spacy_nlp_extract = None


def get_spacy_nlp_extract():
    """Lazy loader for spaCy pipeline in extraction module."""
    global _spacy_nlp_extract
    if _spacy_nlp_extract is None:
        try:
            _spacy_nlp_extract = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp_extract = spacy.blank("en")
    return _spacy_nlp_extract


def extract_scientific_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts structured scientific entities (metrics, benchmarks, architectures)
    from text using rule-based spaCy pattern matching.
    """
    if not text:
        return {"metrics": [], "architectures": [], "datasets": []}
    nlp = get_spacy_nlp_extract()
    doc = nlp(text[:12000])
    matcher = Matcher(nlp.vocab)

    metric_patterns = [
        [
            {
                "LOWER": {
                    "IN": [
                        "accuracy", "acc", "f1", "precision", "recall", "bleu",
                        "rouge", "auc", "map", "ndcg", "mrr", "loss", "perplexity"
                    ]
                }
            }
        ],
        [{"LOWER": "exact"}, {"LOWER": "match"}],
        [{"LOWER": "top"}, {"LOWER": "-"}, {"IS_DIGIT": True}],
    ]
    matcher.add("METRIC", metric_patterns)

    arch_patterns = [
        [
            {
                "LOWER": {
                    "IN": [
                        "transformer", "bert", "roberta", "llama", "gpt", "resnet",
                        "gnn", "lstm", "rag", "t5", "mistral", "autoencoder"
                    ]
                }
            }
        ],
        [{"IS_TITLE": True}, {"LOWER": "net"}],
    ]
    matcher.add("ARCHITECTURE", arch_patterns)

    dataset_patterns = [
        [
            {
                "LOWER": {
                    "IN": [
                        "glue", "superglue", "squad", "imagenet", "retrievalqa",
                        "mmlongbench", "hotpotqa", "triviaqa", "pubmed", "coqa"
                    ]
                }
            }
        ],
        [
            {"IS_UPPER": True, "LENGTH": {">=": 3}},
            {"LOWER": {"IN": ["dataset", "benchmark", "corpus"]}},
        ],
    ]
    matcher.add("DATASET", dataset_patterns)

    matches = matcher(doc)
    results: Dict[str, List[str]] = {"metrics": [], "architectures": [], "datasets": []}
    seen = set()

    for match_id, start, end in matches:
        span_text = doc[start:end].text.strip()
        label = nlp.vocab.strings[match_id]
        key = label.lower() + "s"
        if key in results and span_text.lower() not in seen:
            seen.add(span_text.lower())
            results[key].append(span_text)

    return results

CORE_FIELDS = [
    "objective",
    "abstract",
    "methodology",
    "datasets",
    "algorithms",
    "metrics",
    "results",
    "limitations",
    "future_work",
]

FIELD_DESCRIPTIONS = {
    "objective": "Primary research objective, hypothesis, or core problem statement",
    "abstract": "Verbatim abstract of the paper",
    "methodology": "Proposed technical approach, architecture, or theoretical framework",
    "datasets": "Datasets used for training, benchmarking, and experimental evaluation",
    "algorithms": "Key machine learning models, statistical methods, or algorithms implemented",
    "metrics": "Quantitative evaluation metrics measured (e.g. Accuracy, F1, BLEU, MAP)",
    "results": "Key quantitative experimental outcomes and comparative findings",
    "limitations": "Explicit author-declared limitations, boundary conditions, or caveats",
    "future_work": "Future research directions explicitly proposed by the authors",
}


def normalize_text_for_matching(text: str) -> str:
    """Normalizes whitespace and casing for robust fuzzy substring matching."""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def verify_quote_against_page(
    quote: str,
    page_text: str,
    threshold: float = 85.0,
) -> bool:
    """
    Verifies if an extracted evidence quote is faithfully grounded in the page text.
    Uses exact substring matching followed by partial ratio fuzzy matching.
    Supports ellipsis-separated multi-clause quotes.
    """
    if not quote or not quote.strip():
        return False
    if not page_text or not page_text.strip():
        return False

    q_norm = normalize_text_for_matching(quote)
    p_norm = normalize_text_for_matching(page_text)

    # 1. Exact normalized substring match
    if q_norm in p_norm:
        return True

    # 1b. Multi-clause quotes with ellipsis
    if "..." in quote or "…" in quote:
        parts = [p.strip() for p in re.split(r"\.{3,}|…", quote) if len(p.strip()) > 8]
        if parts:
            all_parts_matched = True
            for part in parts:
                part_norm = normalize_text_for_matching(part)
                if (
                    part_norm not in p_norm
                    and fuzz.partial_ratio(part_norm, p_norm) < threshold
                ):
                    all_parts_matched = False
                    break
            if all_parts_matched:
                return True

    # 2. Token set / partial fuzzy ratio
    score_partial = fuzz.partial_ratio(q_norm, p_norm)
    if score_partial >= threshold:
        return True

    score_token_set = fuzz.token_set_ratio(q_norm, p_norm)
    if score_token_set >= threshold:
        return True

    return False


def build_extraction_context(
    paper_id: str,
    db: DatabaseManager,
    max_tokens: int = 4000,
) -> str:
    """
    Constructs a targeted prompt context from key paper sections,
    enclosed within security-delimited <documents> tags.
    """
    chunks = db.get_chunks(paper_id)
    if not chunks:
        # Fallback to page text if chunks not available
        pages = db.get_pages(paper_id)
        context_parts = []
        for p in pages[:6]:
            context_parts.append(f"[Page {p['page_number']}]:\n{p['text']}")
        inner_content = "\n\n".join(context_parts)
        return f"<documents>\n{inner_content}\n</documents>"

    # Prioritize key sections: Abstract, Introduction, Methods, Experiments, Results, Conclusion
    priority_sections = [
        "abstract",
        "introduction",
        "method",
        "methodology",
        "approach",
        "experiment",
        "results",
        "discussion",
        "limitation",
        "conclusion",
        "future work",
    ]

    # Sample balanced chunks across paper sections to avoid front-loading
    section_groups: Dict[str, list] = {}
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "reference" in sec:
            continue
        group = "other"
        for p in priority_sections:
            if p in sec:
                group = p
                break
        section_groups.setdefault(group, []).append(c)

    selected_chunks = []
    for group, group_chunks in section_groups.items():
        limit = 3 if group in ["method", "methodology", "approach", "experiment", "results"] else 2
        selected_chunks.extend(group_chunks[:limit])

    # Sort selected chunks in reading order
    selected_chunks.sort(key=lambda x: (x.get("page_start", 1), x.get("chunk_index", 0)))

    context_lines = []
    for c in selected_chunks:
        header = f"[Page {c['page_start']}, Section: {c['section_name']}]:"
        context_lines.append(f"{header}\n{c['text']}")

    inner_text = "\n\n---\n\n".join(context_lines)
    return f"<documents>\n{inner_text}\n</documents>"


def rule_based_fallback_extraction(
    paper_id: str,
    db: DatabaseManager,
) -> Dict[str, ExtractedField]:
    """
    Provides a grounded heuristic fallback extraction when an external LLM
    is unavailable, offline, or lacks an API key. Extracts directly from database pages/chunks.
    """
    paper = db.get_paper(paper_id)
    chunks = db.get_chunks(paper_id)
    pages = db.get_pages(paper_id)

    meta_json = {}
    if paper and paper.get("metadata_json"):
        try:
            meta_json = json.loads(paper["metadata_json"])
        except Exception:
            pass

    page_map = {p["page_number"]: p["text"] for p in pages}
    first_page_text = page_map.get(1, "")

    extractions: Dict[str, ExtractedField] = {}

    # 1. Abstract
    abstract_val = meta_json.get("abstract") or "Not stated in paper"
    abstract_quote = abstract_val if abstract_val != "Not stated in paper" else ""
    extractions["abstract"] = ExtractedField(
        field_name="abstract",
        value=abstract_val,
        evidence_quote=abstract_quote[:300],
        page=1,
        confidence=0.9 if abstract_val != "Not stated in paper" else 0.5,
        source_section="Abstract",
        verified=verify_quote_against_page(abstract_quote[:300], first_page_text) if abstract_quote else True,
    )

    # 2. Objective
    obj_val = "Not stated in paper"
    obj_quote = ""
    obj_page = 1
    # Search introduction or abstract chunk for objective keywords
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "abstract" in sec or "intro" in sec:
            text = c["text"]
            for sentence in text.split(". "):
                s_lower = sentence.lower()
                if any(w in s_lower for w in ["propose", "present", "investigate", "aims to", "we design", "focuses on"]):
                    obj_val = sentence.strip() + "."
                    obj_quote = sentence.strip() + "."
                    obj_page = c["page_start"]
                    break
        if obj_quote:
            break

    extractions["objective"] = ExtractedField(
        field_name="objective",
        value=obj_val,
        evidence_quote=obj_quote,
        page=obj_page if obj_quote else 0,
        confidence=0.85 if obj_quote else 0.4,
        source_section="Introduction" if obj_quote else "Unknown",
        verified=verify_quote_against_page(obj_quote, page_map.get(obj_page, "")) if obj_quote else True,
    )

    # 3. Methodology
    meth_val = "Not stated in paper"
    meth_quote = ""
    meth_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "method" in sec or "approach" in sec or "architecture" in sec:
            sentences = c["text"].split(". ")
            if sentences:
                meth_val = sentences[0].strip() + "."
                meth_quote = sentences[0].strip() + "."
                meth_page = c["page_start"]
                break

    extractions["methodology"] = ExtractedField(
        field_name="methodology",
        value=meth_val,
        evidence_quote=meth_quote,
        page=meth_page if meth_quote else 0,
        confidence=0.8 if meth_quote else 0.4,
        source_section="Methodology" if meth_quote else "Unknown",
        verified=verify_quote_against_page(meth_quote, page_map.get(meth_page, "")) if meth_quote else True,
    )

    # 4. Datasets
    data_val = "Not stated in paper"
    data_quote = ""
    data_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "dataset" in sec or "experiment" in sec or "evaluation" in sec:
            text = c["text"]
            for s in text.split(". "):
                if any(w in s.lower() for w in ["dataset", "corpus", "benchmark", "evaluated on", "collected from"]):
                    data_val = s.strip() + "."
                    data_quote = s.strip() + "."
                    data_page = c["page_start"]
                    break
        if data_quote:
            break

    extractions["datasets"] = ExtractedField(
        field_name="datasets",
        value=data_val,
        evidence_quote=data_quote,
        page=data_page if data_quote else 0,
        confidence=0.8 if data_quote else 0.4,
        source_section="Experiments" if data_quote else "Unknown",
        verified=verify_quote_against_page(data_quote, page_map.get(data_page, "")) if data_quote else True,
    )

    # 5. Algorithms
    algo_val = "Not stated in paper"
    algo_quote = ""
    algo_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "method" in sec or "model" in sec or "algorithm" in sec:
            for s in c["text"].split(". "):
                if any(w in s.lower() for w in ["network", "transformer", "gnn", "model", "algorithm", "classifier", "encoder"]):
                    algo_val = s.strip() + "."
                    algo_quote = s.strip() + "."
                    algo_page = c["page_start"]
                    break
        if algo_quote:
            break

    extractions["algorithms"] = ExtractedField(
        field_name="algorithms",
        value=algo_val,
        evidence_quote=algo_quote,
        page=algo_page if algo_quote else 0,
        confidence=0.8 if algo_quote else 0.4,
        source_section="Methodology" if algo_quote else "Unknown",
        verified=verify_quote_against_page(algo_quote, page_map.get(algo_page, "")) if algo_quote else True,
    )

    # 6. Metrics
    metr_val = "Not stated in paper"
    metr_quote = ""
    metr_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "metric" in sec or "experiment" in sec or "evaluation" in sec or "results" in sec:
            for s in c["text"].split(". "):
                if any(w in s.lower() for w in ["accuracy", "f1", "precision", "recall", "mrr", "ndcg", "auc", "bleu", "rouge", "metric"]):
                    metr_val = s.strip() + "."
                    metr_quote = s.strip() + "."
                    metr_page = c["page_start"]
                    break
        if metr_quote:
            break

    extractions["metrics"] = ExtractedField(
        field_name="metrics",
        value=metr_val,
        evidence_quote=metr_quote,
        page=metr_page if metr_quote else 0,
        confidence=0.8 if metr_quote else 0.4,
        source_section="Experiments" if metr_quote else "Unknown",
        verified=verify_quote_against_page(metr_quote, page_map.get(metr_page, "")) if metr_quote else True,
    )

    # 7. Results
    res_val = "Not stated in paper"
    res_quote = ""
    res_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "result" in sec or "finding" in sec:
            for s in c["text"].split(". "):
                if any(w in s.lower() for w in ["achieve", "outperform", "%", "score", "statistically", "higher than"]):
                    res_val = s.strip() + "."
                    res_quote = s.strip() + "."
                    res_page = c["page_start"]
                    break
        if res_quote:
            break

    extractions["results"] = ExtractedField(
        field_name="results",
        value=res_val,
        evidence_quote=res_quote,
        page=res_page if res_quote else 0,
        confidence=0.85 if res_quote else 0.4,
        source_section="Results" if res_quote else "Unknown",
        verified=verify_quote_against_page(res_quote, page_map.get(res_page, "")) if res_quote else True,
    )

    # 8. Limitations
    lim_val = "Not stated in paper"
    lim_quote = ""
    lim_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "limitation" in sec or "discussion" in sec or "threat" in sec:
            for s in c["text"].split(". "):
                if any(w in s.lower() for w in ["limit", "constraint", "bottleneck", "drawback", "computational cost", "threat to validity"]):
                    lim_val = s.strip() + "."
                    lim_quote = s.strip() + "."
                    lim_page = c["page_start"]
                    break
        if lim_quote:
            break

    extractions["limitations"] = ExtractedField(
        field_name="limitations",
        value=lim_val,
        evidence_quote=lim_quote,
        page=lim_page if lim_quote else 0,
        confidence=0.85 if lim_quote else 0.4,
        source_section="Limitations" if lim_quote else "Unknown",
        verified=verify_quote_against_page(lim_quote, page_map.get(lim_page, "")) if lim_quote else True,
    )

    # 9. Future Work
    fut_val = "Not stated in paper"
    fut_quote = ""
    fut_page = 1
    for c in chunks:
        sec = c.get("section_name", "").lower()
        if "future" in sec or "conclusion" in sec:
            for s in c["text"].split(". "):
                if any(w in s.lower() for w in ["future work", "in the future", "intend to", "plan to", "promising direction", "we will explore"]):
                    fut_val = s.strip() + "."
                    fut_quote = s.strip() + "."
                    fut_page = c["page_start"]
                    break
        if fut_quote:
            break

    extractions["future_work"] = ExtractedField(
        field_name="future_work",
        value=fut_val,
        evidence_quote=fut_quote,
        page=fut_page if fut_quote else 0,
        confidence=0.85 if fut_quote else 0.4,
        source_section="Future Work" if fut_quote else "Unknown",
        verified=verify_quote_against_page(fut_quote, page_map.get(fut_page, "")) if fut_quote else True,
    )

    return extractions


def extract_paper_components(
    paper_id: str,
    db: DatabaseManager,
    force_refresh: bool = False,
) -> Dict[str, ExtractedField]:
    """
    Orchestrates grounded structured extraction for an academic paper:
    1. Checks if extractions already exist in database unless force_refresh is True.
    2. Builds targeted context with physical page citations.
    3. Calls LLM with strict schema output (or falls back to grounded rule extraction).
    4. Verifies all extracted evidence quotes against actual page text.
    5. Saves all fields to the database and returns the dictionary of ExtractedField objects.
    """
    if not force_refresh:
        existing = db.get_extractions(paper_id)
        if existing and len(existing) >= 5:
            logger.info(f"Loaded existing extractions for paper {paper_id}")
            result = {}
            for name, data in existing.items():
                result[name] = ExtractedField(
                    field_name=name,
                    value=data["value"],
                    evidence_quote=data["evidence_quote"] or "",
                    page=data["page_number"] or 0,
                    confidence=data["confidence"] or 1.0,
                    source_section=data["source_section"],
                    verified=bool(data["verified"]),
                    user_override=bool(data["user_override"]),
                )
            return result

    # Build prompt context
    context = build_extraction_context(paper_id, db)
    pages = db.get_pages(paper_id)
    page_map = {p["page_number"]: p["text"] for p in pages}

    extraction_results: Dict[str, ExtractedField] = {}

    try:
        client = get_llm_client()
        user_prompt = (
            f"Analyze the following research paper excerpts and extract the 9 structured research components.\n\n"
            f"{context}\n\n"
            f"Remember: Output strictly in JSON conforming to the StructuredPaperExtraction schema.\n"
            f"Every field must include: value, evidence_quote, page, confidence, and source_section."
        )

        structured_output = client.generate_structured(
            prompt=user_prompt,
            schema=StructuredPaperExtraction,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
        )

        if structured_output:
            for field_name in CORE_FIELDS:
                field: ExtractedField = getattr(structured_output, field_name)
                # Verify quote against page text
                if field.value == "Not stated in paper" or not field.evidence_quote.strip():
                    field.verified = True
                else:
                    page_text = page_map.get(field.page, "")
                    field.verified = verify_quote_against_page(field.evidence_quote, page_text)
                    if not field.verified:
                        # Search other pages in the paper for the exact quote
                        for alt_p_num, alt_text in page_map.items():
                            if alt_p_num != field.page and verify_quote_against_page(field.evidence_quote, alt_text):
                                field.page = alt_p_num
                                field.verified = True
                                break
                extraction_results[field_name] = field

    except Exception as err:
        logger.warning(f"LLM structured extraction unavailable or failed ({err}). Using grounded heuristic extraction.")
        extraction_results = rule_based_fallback_extraction(paper_id, db)

    # If LLM didn't return all fields, fill in with fallback
    if len(extraction_results) < len(CORE_FIELDS):
        fallback = rule_based_fallback_extraction(paper_id, db)
        for name, field in fallback.items():
            if name not in extraction_results:
                extraction_results[name] = field

    # Save to database
    for _field_name, field in extraction_results.items():
        db.insert_or_update_extraction(paper_id, field)

    logger.info(f"Completed and stored extractions for paper {paper_id}")
    return extraction_results
