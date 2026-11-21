"""
ResearchLens AI - Smart Hybrid Search Page.
Fuses FAISS dense vector search and BM25 sparse search with Reciprocal Rank Fusion (RRF)
and Cross-Encoder reranking, featuring transparent score inspection and metadata filters.
"""

import html
import re
from typing import List

import streamlit as st

from core.config import settings
from core.logging_config import get_logger
from core.schemas import SearchResult
from database.db_manager import DBManager
from modules.bm25_search import BM25SearchEngine
from modules.hybrid_search import HybridSearchEngine
from modules.reranking import CrossEncoderReranker
from modules.semantic_search import SemanticSearchEngine
from ui.components import (
    citation_pill,
    empty_state,
    page_header,
    safe_page_render,
    status_badge,
)
from ui.theme import inject_custom_css

logger = get_logger("page.search")
db = DBManager(settings.database_path)

st.set_page_config(page_title="Smart Hybrid Search | ResearchLens AI", page_icon="⚡", layout="wide")
inject_custom_css()


@st.cache_resource
def get_search_engines():
    """Initializes and caches search engines in Streamlit."""
    semantic = SemanticSearchEngine()
    bm25 = BM25SearchEngine()
    reranker = CrossEncoderReranker()
    hybrid = HybridSearchEngine(semantic_engine=semantic, bm25_engine=bm25, reranker=reranker)
    return semantic, bm25, reranker, hybrid


semantic_engine, bm25_engine, reranker_engine, hybrid_engine = get_search_engines()


def highlight_terms(text: str, query_str: str) -> str:
    """Safely escapes HTML and highlights query terms in passage text."""
    escaped_text = html.escape(text)
    terms = [re.escape(t.strip()) for t in query_str.split() if len(t.strip()) > 2]
    if not terms:
        return escaped_text
    pattern = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(
        r'<mark style="background-color: #FEF08A; color: #1E293B; padding: 1px 4px; border-radius: 3px; font-weight: 500;">\1</mark>',
        escaped_text,
    )


@safe_page_render("Smart Hybrid Search")
def main():
    # Ensure active collection
    collections = db.list_collections()
    if not collections:
        empty_state("No Collections Found", "Create a workspace collection from the library first.", icon="📚")
        return

    col_map = {c["name"]: c["id"] for c in collections}
    if "active_collection_id" not in st.session_state or st.session_state["active_collection_id"] not in col_map.values():
        st.session_state["active_collection_id"] = collections[0]["id"]
        st.session_state["active_collection_name"] = collections[0]["name"]

    current_collection_id = st.session_state["active_collection_id"]
    current_collection_name = st.session_state["active_collection_name"]

    page_header(
        title="Smart Hybrid Search",
        subtitle=f"Multi-stage vector & lexical retrieval with reciprocal rank fusion across {current_collection_name}",
        badge_text="FAISS + BM25 + RRF + Cross-Encoder",
    )

    papers = db.list_papers(current_collection_id)
    all_chunks = db.get_all_chunks(current_collection_id)

    # -----------------------------------------------------------------------
    # Sidebar Controls & Index Status
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Search Controls")
        st.caption(f"Workspace: **{current_collection_name}**")
        st.caption(f"Total Papers: **{len(papers)}** | Chunks: **{len(all_chunks)}**")

        st.markdown("---")
        st.markdown("#### 🔄 Index Management")
        if st.button("Rebuild Search Index", use_container_width=True, type="secondary"):
            with st.spinner("Re-indexing all passages into FAISS and BM25..."):
                if all_chunks:
                    semantic_engine.rebuild_index(current_collection_id, all_chunks)
                    bm25_engine.index_chunks(current_collection_id, all_chunks)
                    st.toast(f"Successfully indexed {len(all_chunks)} passages!", icon="✅")
                else:
                    st.warning("No chunks available in this collection.")
            st.rerun()

    if not papers or not all_chunks:
        empty_state(
            title="Collection is Empty",
            message="Upload research papers or ingest PDFs to index passages for semantic and hybrid search.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="📄",
        )
        return

    # -----------------------------------------------------------------------
    # Search Query & Filters Card
    # -----------------------------------------------------------------------
    with st.container(border=True):
        col_q, col_k = st.columns([4, 1])

        # Preset query handler
        preset_query = st.session_state.pop("preset_search_query", "")
        initial_query = preset_query if preset_query else st.session_state.get("search_query_text", "")

        query = col_q.text_input(
            "Research Query",
            value=initial_query,
            placeholder="e.g. self-attention mechanism, clinical entity extraction, or link prediction benchmark",
            key="search_input_field",
        )
        top_k = col_k.number_input("Top Results", min_value=1, max_value=20, value=5, step=1)

        col_mode, col_paper, col_sec = st.columns([1.5, 1.5, 1.2])

        mode_options = [
            "🚀 Hybrid + Cross-Encoder (Rerank)",
            "⚡ Hybrid RRF (Dense + Lexical)",
            "🧠 Dense Only (FAISS Vectors)",
            "📖 BM25 Only (Keyword Match)",
        ]
        selected_mode = col_mode.selectbox("Retrieval Engine", options=mode_options, index=0)

        # Paper filter
        paper_options = {"All Papers": None}
        for p in papers:
            pid = p.get("id")
            if pid:
                title_label = p.get("title") or p.get("filename", "Untitled Paper")
                paper_options[title_label] = pid
        selected_paper_label = col_paper.selectbox("Filter by Paper", options=list(paper_options.keys()))
        filter_paper_id = paper_options[selected_paper_label]

        # Section filter
        section_options = [
            "All Sections",
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
        ]
        selected_section = col_sec.selectbox("Filter by Section", options=section_options)
        filter_section = None if selected_section == "All Sections" else selected_section

    # -----------------------------------------------------------------------
    # Empty State & Suggestions
    # -----------------------------------------------------------------------
    if not query.strip():
        st.markdown("#### 💡 Quick Query Suggestions")
        st.caption("Click any topic below to run a pre-configured hybrid search:")

        sugg_col1, sugg_col2, sugg_col3 = st.columns(3)
        with sugg_col1:
            if st.button("🔍 Attention Mechanism & Architectures", use_container_width=True):
                st.session_state["preset_search_query"] = "attention mechanism neural network architecture transformer"
                st.rerun()
        with sugg_col2:
            if st.button("📊 Benchmark Performance & Results", use_container_width=True):
                st.session_state["preset_search_query"] = "experimental results benchmark accuracy performance comparison"
                st.rerun()
        with sugg_col3:
            if st.button("⚠️ Limitations & Future Directions", use_container_width=True):
                st.session_state["preset_search_query"] = "limitations failure modes future work research directions"
                st.rerun()

        st.markdown(
            """
            <div style="margin-top: 2rem; padding: 1.5rem; background: var(--rl-surface); border: 1px dashed var(--rl-border); border-radius: 12px; text-align: center;">
                <p style="margin: 0; font-size: 0.95rem; color: var(--rl-text-muted);">
                    Type a query above or click a suggestion to explore grounded passages with instant page citations and score inspections.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # -----------------------------------------------------------------------
    # Retrieval Execution
    # -----------------------------------------------------------------------
    # Auto-ensure BM25 and Semantic indices are ready
    if current_collection_id not in bm25_engine._chunks:
        if all_chunks:
            bm25_engine.index_chunks(current_collection_id, all_chunks)
            semantic_engine.rebuild_index(current_collection_id, all_chunks)

    results: List[SearchResult] = []
    with st.spinner("Retrieving and reranking passages across collection..."):
        if "Cross-Encoder" in selected_mode:
            results = hybrid_engine.search(
                collection_id=current_collection_id,
                query=query.strip(),
                top_k=top_k,
                top_k_dense=15,
                top_k_bm25=15,
                top_k_rrf=20,
                filter_paper_id=filter_paper_id,
                filter_section=filter_section,
                enable_reranker=True,
            )
        elif "Hybrid RRF" in selected_mode:
            results = hybrid_engine.search(
                collection_id=current_collection_id,
                query=query.strip(),
                top_k=top_k,
                top_k_dense=15,
                top_k_bm25=15,
                top_k_rrf=20,
                filter_paper_id=filter_paper_id,
                filter_section=filter_section,
                enable_reranker=False,
            )
        elif "Dense Only" in selected_mode:
            results = semantic_engine.search(
                collection_id=current_collection_id,
                query=query.strip(),
                top_k=top_k,
                filter_paper_id=filter_paper_id,
                filter_section=filter_section,
            )
        elif "BM25 Only" in selected_mode:
            results = bm25_engine.search(
                collection_id=current_collection_id,
                query=query.strip(),
                top_k=top_k,
                filter_paper_id=filter_paper_id,
                filter_section=filter_section,
            )

    # -----------------------------------------------------------------------
    # Results Display
    # -----------------------------------------------------------------------
    res_header_col, count_col = st.columns([4, 1])
    res_header_col.markdown(f"### 🎯 Search Results (`{len(results)}` matches found)")
    count_col.markdown(
        f"<div style='text-align: right; padding-top: 0.5rem;'>{status_badge('ready', f'{len(results)} Retrieved')}</div>",
        unsafe_allow_html=True,
    )

    if not results:
        st.warning("No matching passages found. Try syncing the index or expanding query keywords.")
        return

    for idx, res in enumerate(results):
        page_label = f"Page {res.page_start}" if res.page_start == res.page_end else f"Pages {res.page_start}-{res.page_end}"
        pill_html = citation_pill(res.page_start)

        with st.container(border=True):
            top_col, meta_col = st.columns([3.5, 1.5])

            with top_col:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                        <span style="background: var(--rl-primary); color: white; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; font-weight: 700;">#{idx + 1}</span>
                        <span style="font-size: 1.05rem; font-weight: 600; color: var(--rl-text-primary);">{html.escape(res.paper_title)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with meta_col:
                score_str = ""
                if res.rerank_score is not None:
                    score_str = f"Score: <b>{res.rerank_score:.3f}</b> (Rerank)"
                elif res.rrf_score is not None:
                    score_str = f"Score: <b>{res.rrf_score:.4f}</b> (RRF)"
                elif res.dense_score is not None:
                    score_str = f"Score: <b>{res.dense_score:.3f}</b> (Dense)"
                elif res.bm25_score is not None:
                    score_str = f"Score: <b>{res.bm25_score:.2f}</b> (BM25)"

                st.markdown(
                    f"""
                    <div style="text-align: right; font-size: 0.85rem; color: var(--rl-text-muted);">
                        {pill_html} <span style="margin-left: 0.4rem;">{score_str}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Metadata tags row
            st.markdown(
                f"""
                <div style="display: flex; gap: 0.5rem; margin-bottom: 0.6rem; font-size: 0.8rem;">
                    <span style="background: var(--rl-surface); border: 1px solid var(--rl-border); padding: 2px 8px; border-radius: 4px; color: var(--rl-text-secondary);">Section: <b>{html.escape(res.section_name)}</b></span>
                    <span style="background: var(--rl-surface); border: 1px solid var(--rl-border); padding: 2px 8px; border-radius: 4px; color: var(--rl-text-secondary);">{page_label}</span>
                    <span style="background: var(--rl-surface); border: 1px solid var(--rl-border); padding: 2px 8px; border-radius: 4px; color: var(--rl-text-muted);">Chunk: {res.chunk_id[:16]}...</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Highlighted passage excerpt
            highlighted_body = highlight_terms(res.text, query)
            st.markdown(
                f"""
                <div class="rl-evidence-quote" style="font-size: 0.92rem; line-height: 1.6; margin: 0.4rem 0 0.8rem 0;">
                    {highlighted_body}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Actions & Score Breakdown
            action_col1, action_col2 = st.columns([3, 1])

            with action_col1:
                with st.expander("🔍 Multi-Stage Score Breakdown"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**Dense (FAISS):**<br>`{f'{res.dense_score:.4f}' if res.dense_score is not None else 'N/A'}`", unsafe_allow_html=True)
                    c2.markdown(f"**BM25 (Sparse):**<br>`{f'{res.bm25_score:.4f}' if res.bm25_score is not None else 'N/A'}`", unsafe_allow_html=True)
                    c3.markdown(f"**RRF (k=60):**<br>`{f'{res.rrf_score:.6f}' if res.rrf_score is not None else 'N/A'}`", unsafe_allow_html=True)
                    c4.markdown(f"**Cross-Encoder:**<br>`{f'{res.rerank_score:.4f}' if res.rerank_score is not None else 'N/A'}`", unsafe_allow_html=True)

            with action_col2:
                btn_key = f"chat_passage_{res.chunk_id}_{idx}"
                if st.button("💬 Ask in Chat", key=btn_key, use_container_width=True, type="secondary"):
                    prompt_text = (
                        f"In the paper '{res.paper_title}' (Section '{res.section_name}', {page_label}), "
                        f"it states: \"{res.text[:220]}...\". Can you elaborate on this finding and its implications?"
                    )
                    st.session_state["pending_chat_prompt"] = prompt_text
                    st.switch_page("pages/4_Research_Chat.py")


if __name__ == "__main__":
    main()
