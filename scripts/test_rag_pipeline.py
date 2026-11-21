"""
Test script running 5 research questions through the RAG pipeline and Citation Verifier.
Includes 4 factual cross-paper queries and 1 unanswerable negative test query.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from database.db_manager import DatabaseManager
from modules.rag_pipeline import answer_research_question

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("test_rag")

TEST_QUESTIONS = [
    {
        "id": "Q1_PDFTriage",
        "question": "How does PDFTriage represent document structure to enable context retrieval for Large Language Models?",
        "expected_topic": "HTML-like tree metadata, function calling triage, born-digital PDFs",
        "expected_paper": "2024.emnlp-industry.13-paper-1.pdf",
    },
    {
        "id": "Q2_PIER_QA",
        "question": "What preprocessing algorithms and vision models does PIER-QA use to process non-textual PDF elements?",
        "expected_topic": "DBSCAN clustering, Marker markdown, LLaVA image captioning",
        "expected_paper": "research paper-2.pdf",
    },
    {
        "id": "Q3_NeuSym_RAG",
        "question": "What are the three main stages of the NeuSym-RAG framework and what database engine is utilized?",
        "expected_topic": "Parsing, Encoding, Interaction stages with DuckDB and Milvus",
        "expected_paper": "research paper-3.pdf",
    },
    {
        "id": "Q4_Multi_Modal_Benchmark",
        "question": "What common multi-page benchmark dataset is evaluated in both Doc-React and MHier-RAG?",
        "expected_topic": "MMLongBench / MMLongBench-Doc / SlideVQA",
        "expected_paper": "Research paper-4.pdf or research paper-5.pdf",
    },
    {
        "id": "Q5_Unanswerable_Negative",
        "question": "What quantum annealing schedule, coherence time, and qubit topology was utilized to optimize the neural network parameters?",
        "expected_topic": "Insufficient evidence declaration",
        "is_unanswerable": True,
    },
]


def run_rag_tests():
    settings = Settings()
    db = DatabaseManager(settings.database_path)

    collection_id = "default-collection"

    print("\n=======================================================")
    print("RUNNING 5 TEST QUESTIONS THROUGH RESEARCHLENS RAG PIPELINE")
    print(f"Collection: {collection_id} | Model: {settings.gemini_model}")
    print("=======================================================\n")

    results = []

    for idx, q_item in enumerate(TEST_QUESTIONS, 1):
        q_text = q_item["question"]
        print(f"\n[{idx}/5] Question: {q_text}")

        # Execute RAG pipeline
        rag_resp = answer_research_question(
            query=q_text,
            collection_id=collection_id,
            db=db,
            top_k=5,
        )

        answer = rag_resp.answer
        citations = [f"[{c.paper_title}, p. {c.page_number}]" for c in rag_resp.citations]
        verifications = rag_resp.verifications
        support_score = rag_resp.overall_support_score

        supported = [v for v in verifications if v.is_supported]
        unsupported = [v for v in verifications if not v.is_supported]
        partially = [v for v in verifications if v.support_level == "partially_supported"]

        print(f"\n  Answer Preview:\n  {answer[:250]}..." if len(answer) > 250 else f"\n  Answer:\n  {answer}")
        print(f"\n  Citations Extracted: {citations[:4]}")
        print(f"  Evidence Support Rate: {support_score * 100:.1f}%")
        print(f"  Total Claims: {len(verifications)} (Supported: {len(supported)}, Partial: {len(partially)}, Unsupported: {len(unsupported)})")

        is_insufficient = rag_resp.insufficient_evidence or ("do not contain sufficient evidence" in answer.lower())
        if q_item.get("is_unanswerable"):
            print(f"  Negative Query Check: {'PASSED (Properly refused)' if is_insufficient else 'FAILED (Hallucinated)'}")
        else:
            print(f"  Factual Query Check: {'PASSED (Answered with citations)' if citations else 'NO CITATIONS'}")

        results.append({
            "id": q_item["id"],
            "question": q_text,
            "answer": answer,
            "citations": citations,
            "support_score": support_score,
            "claims_count": len(verifications),
            "is_insufficient": is_insufficient,
        })

    out_file = Path("evaluation/real_papers_rag_tests.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] RAG test suite completed. Results saved to {out_file}\n")


if __name__ == "__main__":
    run_rag_tests()
