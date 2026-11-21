"""
Conversational RAG pipeline for ResearchLens AI.
Orchestrates multi-turn query rewriting, hybrid passage retrieval, prompt injection defense,
grounded LLM synthesis with inline citations [Paper N, p. X], and claim verification.
"""

import logging
from typing import Dict, List, Optional, Tuple

from core.llm import get_llm_client
from core.llm.prompts import QUERY_REWRITE_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from core.schemas import Citation, RAGResponse, SearchResult
from database.db_manager import DatabaseManager
from modules.citation_verifier import (
    compute_support_score,
    split_claims,
    verify_claims,
)
from modules.hybrid_search import hybrid_search

logger = logging.getLogger("researchlens.rag")


def rewrite_user_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Reformulates follow-up conversational queries into standalone, keyword-rich search queries.
    If no history exists or LLM is offline, returns the query unchanged.
    """
    if not chat_history or len(chat_history) < 2:
        return query

    # Build brief conversation history
    history_snippets = []
    for msg in chat_history[-4:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        history_snippets.append(f"{role}: {content}")
    history_text = "\n".join(history_snippets)

    prompt = (
        f"Conversation History:\n{history_text}\n\n"
        f"Latest Follow-up Question: {query}\n\n"
        f"Reformulated Standalone Query:"
    )

    try:
        client = get_llm_client()
        rewritten = client.generate(
            prompt=prompt,
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=100,
        )
        if rewritten and len(rewritten.strip()) > 3:
            logger.info(f"Rewrote query '{query}' -> '{rewritten.strip()}'")
            return rewritten.strip()
    except Exception as err:
        logger.debug(f"Query rewrite skipped ({err}); using raw query.")

    return query


def format_grounding_documents(
    chunks: List[SearchResult],
) -> Tuple[str, Dict[int, str], List[Citation]]:
    """
    Formats retrieved SearchResult chunks into security-delimited <documents> XML block.
    Assigns stable 1-indexed paper numbers (Paper 1, Paper 2, ...) and generates Citation metadata.
    """
    # Create stable mapping of distinct paper IDs to sequential numbers
    distinct_paper_ids = []
    for c in chunks:
        if c.paper_id not in distinct_paper_ids:
            distinct_paper_ids.append(c.paper_id)

    paper_num_map: Dict[int, str] = {}
    id_to_num: Dict[str, int] = {}
    for idx, pid in enumerate(distinct_paper_ids, 1):
        paper_num_map[idx] = pid
        id_to_num[pid] = idx

    doc_blocks = []
    citations: List[Citation] = []

    for c in chunks:
        p_num = id_to_num.get(c.paper_id, 1)
        doc_header = f'paper="Paper {p_num}: {c.paper_title}" page="{c.page_start}" section="{c.section_name}"'
        doc_blocks.append(f"<document {doc_header}>\n{c.text}\n</document>")

        citations.append(
            Citation(
                paper_id=c.paper_id,
                paper_title=c.paper_title,
                page_number=c.page_start,
                section_name=c.section_name,
                snippet=c.text[:200] + ("..." if len(c.text) > 200 else ""),
            )
        )

    documents_xml = "<documents>\n" + "\n\n".join(doc_blocks) + "\n</documents>"
    return documents_xml, paper_num_map, citations


def answer_research_question(
    query: str,
    collection_id: str,
    paper_ids: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    db: Optional[DatabaseManager] = None,
) -> RAGResponse:
    """
    Executes an end-to-end grounded RAG turn:
    1. Reformulates query using conversation history.
    2. Performs hybrid search + cross-encoder reranking across collection papers.
    3. Detects empty or insufficient evidence early.
    4. Constructs XML-bounded prompt enforcing inline [Paper N, p. X] citations.
    5. Calls LLM with strict grounding directives (or falls back to grounded extractive summary).
    6. Runs citation verification layer on every sentence claim.
    """
    if db is None:
        db = DatabaseManager()

    # 1. Query rewriting
    search_query = rewrite_user_query(query, chat_history)

    # 2. Hybrid retrieval
    retrieved_chunks = hybrid_search(
        query=search_query,
        collection_id=collection_id,
        paper_ids=paper_ids,
        top_k=top_k,
        db=db,
    )

    # 3. Early check for zero evidence
    if not retrieved_chunks:
        insufficient_msg = (
            "The uploaded papers do not contain sufficient evidence to answer this question. "
            "Missing information: No relevant passages were retrieved from the selected papers."
        )
        return RAGResponse(
            query=query,
            answer=insufficient_msg,
            citations=[],
            verifications=[],
            overall_support_score=0.0,
            insufficient_evidence=True,
        )

    # 4. Format context with XML tags
    docs_xml, paper_num_map, citations = format_grounding_documents(retrieved_chunks)

    # 5. Synthesize answer
    user_prompt = (
        f"Answer the following academic research question based strictly on the provided documents.\n\n"
        f"{docs_xml}\n\n"
        f"Question: {query}\n\n"
        f"Answer with inline citations like [Paper N, p. X]:"
    )

    generated_answer = ""
    try:
        client = get_llm_client()
        generated_answer = client.generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=600,
        )
    except Exception as err:
        logger.warning(f"LLM generation unavailable ({err}); constructing grounded fallback answer.")

    # Grounded fallback if LLM offline / empty
    if not generated_answer or not generated_answer.strip():
        top_c = retrieved_chunks[0]
        p_num = 1
        for num, pid in paper_num_map.items():
            if pid == top_c.paper_id:
                p_num = num
                break
        generated_answer = (
            f"Based on {top_c.paper_title}, {top_c.text.strip()[:300]}... "
            f"[Paper {p_num}, p. {top_c.page_start}]."
        )

    # 6. Check for explicit insufficient evidence response
    insufficient = "do not contain sufficient evidence" in generated_answer.lower()

    # 7. Split into claims and verify citations
    claims = split_claims(generated_answer)
    verifications = verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
        paper_number_map=paper_num_map,
    )
    support_score = compute_support_score(verifications)

    return RAGResponse(
        query=query,
        answer=generated_answer,
        citations=citations,
        verifications=verifications,
        overall_support_score=support_score,
        insufficient_evidence=insufficient,
    )
