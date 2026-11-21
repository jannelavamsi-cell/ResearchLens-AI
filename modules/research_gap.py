"""
Research gaps and opportunity discovery module for ResearchLens AI.
Aggregates author-declared limitations and proposed future directions across literature,
performs frequency analysis, and synthesizes grounded, actionable research opportunities.
"""

import logging
from typing import Any, Dict, List, Optional

from core.schemas import ResearchGapItem
from database.db_manager import DatabaseManager
from modules.research_extraction import extract_paper_components

logger = logging.getLogger("researchlens.gaps")


# Known recurrent challenge taxonomies in ML/NLP
TAXONOMY_PATTERNS = {
    "Computational Scalability & Efficiency": [
        "computational cost", "gpu", "memory", "training time", "latency", "scale",
        "parameters", "resource-intensive", "throughput", "inference time",
    ],
    "Domain Generalization & Robustness": [
        "generalization", "out-of-domain", "distribution shift", "transferability",
        "domain-specific", "unseen", "adversarial", "robustness", "cross-domain",
    ],
    "Data Scarcity & Annotation Bottlenecks": [
        "scarcity", "annotation", "labeled data", "small sample", "few-shot",
        "data collection", "data imbalance", "unlabeled", "manual curation",
    ],
    "Explainability & Mechanistic Interpretability": [
        "interpretability", "black-box", "explainability", "transparency",
        "attention weights", "causal", "hallucination", "alignment",
    ],
    "Multimodal & Temporal Fusion": [
        "multimodal", "temporal", "dynamic", "heterogeneous", "time-series",
        "video", "audio", "long-context", "streaming",
    ],
}


def discover_research_gaps(
    collection_id: str,
    paper_ids: Optional[List[str]] = None,
    db: Optional[DatabaseManager] = None,
) -> List[ResearchGapItem]:
    """
    Analyzes all author-declared limitations and future work items across selected papers.
    Clusters common themes and returns ranked ResearchGapItem opportunities.
    """
    if db is None:
        db = DatabaseManager()

    if paper_ids is None:
        all_papers = db.list_papers(collection_id)
        paper_ids = [p["id"] for p in all_papers if p["status"] == "ready"]

    if not paper_ids:
        return []

    # Aggregate extracted limitations and future directions
    extracted_limitations: List[Dict[str, Any]] = []

    for pid in paper_ids:
        paper = db.get_paper(pid)
        if not paper:
            continue
        title = paper["title"] or paper["filename"]

        exts = extract_paper_components(pid, db)
        lim = exts.get("limitations")
        fut = exts.get("future_work")

        if lim and lim.value != "Not stated in paper":
            extracted_limitations.append({
                "paper_id": pid,
                "paper_title": title,
                "field_name": "limitations",
                "value": lim.value,
                "evidence_quote": lim.evidence_quote,
                "page": lim.page,
            })

        if fut and fut.value != "Not stated in paper":
            extracted_limitations.append({
                "paper_id": pid,
                "paper_title": title,
                "field_name": "future_work",
                "value": fut.value,
                "evidence_quote": fut.evidence_quote,
                "page": fut.page,
            })

    # Group by taxonomy pattern
    grouped_gaps: Dict[str, List[Dict[str, Any]]] = {}
    for item in extracted_limitations:
        text = (item["value"] + " " + item["evidence_quote"]).lower()
        matched = False
        for category, terms in TAXONOMY_PATTERNS.items():
            if any(term in text for term in terms):
                grouped_gaps.setdefault(category, []).append(item)
                matched = True
                break
        if not matched:
            grouped_gaps.setdefault("Methodological Constraints & Open Questions", []).append(item)

    gap_items: List[ResearchGapItem] = []

    for category, items in grouped_gaps.items():
        if not items:
            continue

        distinct_papers = []
        seen_pids = set()
        for it in items:
            if it["paper_id"] not in seen_pids:
                seen_pids.add(it["paper_id"])
                distinct_papers.append({
                    "paper_title": it["paper_title"],
                    "page": it["page"],
                    "limitation_quote": it["evidence_quote"] or it["value"],
                })

        freq = len(distinct_papers)

        # Formulate direction based on category
        if "Scalability" in category:
            direction = (
                "Develop model compression, parameter-efficient fine-tuning (PEFT), "
                "or sparse attention architectures to mitigate severe training and memory overheads."
            )
            rationale = (
                "Directly addresses author-acknowledged hardware and computational bottlenecks, "
                "enabling deployment in resource-constrained environments."
            )
        elif "Generalization" in category:
            direction = (
                "Investigate invariant representation learning and multi-domain benchmark stress-testing "
                "to evaluate behavior under real-world domain shifts."
            )
            rationale = (
                "Counteracts the observed degradation in out-of-distribution performance declared by authors."
            )
        elif "Data Scarcity" in category:
            direction = (
                "Explore synthetic data generation, self-supervised contrastive pre-training, "
                "and active learning to reduce dependence on costly manual annotations."
            )
            rationale = (
                "Overcomes small sample sizes and annotation bottlenecks cited in current experimental setups."
            )
        elif "Explainability" in category:
            direction = (
                "Incorporate mechanistic interpretability probes and causal attribution frameworks "
                "to ensure decisions are inspectable by domain experts."
            )
            rationale = (
                "Resolves black-box trust barriers highlighted in clinical and high-stakes deployment discussions."
            )
        else:
            direction = (
                "Design unified hybrid frameworks integrating structured relational reasoning "
                "with dense contextual semantics."
            )
            rationale = (
                "Synthesizes complementary strengths to overcome single-paradigm constraints declared in the literature."
            )

        gap_statement = (
            f"{category}: Multiple authors cite persistent bottlenecks in this domain."
        )

        gap_items.append(
            ResearchGapItem(
                gap_statement=gap_statement,
                supporting_papers=distinct_papers,
                frequency_count=freq,
                suggested_direction=direction,
                rationale=rationale,
            )
        )

    # Sort descending by frequency count
    gap_items.sort(key=lambda g: g.frequency_count, reverse=True)
    return gap_items
