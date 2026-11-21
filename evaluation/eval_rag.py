"""
Evaluation harness: RAG Answer Quality and Citation Grounding.
Evaluates answer correctness, citation accuracy, and insufficient evidence handling.
"""

import logging
from typing import Any, Dict, List

from database.db_manager import DatabaseManager
from modules.rag_pipeline import answer_research_question

logger = logging.getLogger("researchlens.eval.rag")


def evaluate_rag(
    qa_items: List[Dict[str, Any]],
    collection_id: str,
    db: DatabaseManager,
) -> Dict[str, Any]:
    """
    Evaluates conversational QA turns:
    - Answer Faithfulness (% of claims supported by evidence)
    - Citation Accuracy (% of citations accurately verified)
    - Insufficient Evidence Recall (properly flagging unanswerable queries)
    """
    support_scores = []
    citation_accuracies = []
    insufficient_correct = 0
    insufficient_total = 0

    for item in qa_items:
        q = item["question"]
        expected_tokens = item["expected_answer_contains"]
        is_insufficient = item.get("is_insufficient", False)

        resp = answer_research_question(
            query=q,
            collection_id=collection_id,
            db=db,
            top_k=4,
        )

        support_scores.append(resp.overall_support_score)

        # Check citation accuracy
        verified_count = sum(1 for v in resp.verifications if v.is_supported)
        cit_acc = verified_count / max(1, len(resp.verifications))
        citation_accuracies.append(cit_acc)

        # Check unanswerable query detection
        if is_insufficient:
            insufficient_total += 1
            if resp.insufficient_evidence or any(t.lower() in resp.answer.lower() for t in expected_tokens):
                insufficient_correct += 1

    avg_support = round(sum(support_scores) / max(1, len(support_scores)), 3)
    avg_cit_acc = round(sum(citation_accuracies) / max(1, len(citation_accuracies)), 3)
    insufficient_acc = round(insufficient_correct / max(1, insufficient_total), 3) if insufficient_total else 1.0

    return {
        "answer_faithfulness": avg_support,
        "citation_accuracy": avg_cit_acc,
        "insufficient_evidence_accuracy": insufficient_acc,
    }
