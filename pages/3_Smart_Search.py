"""
ResearchLens AI - Smart Hybrid Search Page.
Fuses FAISS dense vector search and BM25 sparse search with Reciprocal Rank Fusion (RRF)
and Cross-Encoder reranking, featuring transparent score inspection and metadata filters.
"""

import re

import streamlit as st

from core.config import settings
from core.logging_config import get_logger
from database.db_manager import DBManager
from modules.bm25_search import BM25SearchEngine
from modules.hybrid_search import HybridSearchEngine
from modules.reranking import CrossEncoderReranker
from modules.semantic_search import SemanticSearchEngine

logger = get_logger("page.search")
db = DBManager(settings.database_path)

st.set_page_config(page_title="Smart Hybrid Search", page_icon="⚡", layout="wide")

# Ensure active collection
if "active_collection_id" not in st.session_state:
    collections = db.list_collections()
    st.session_state["active_collection_id"] = collections[0]["id"]
    st.session_state["active_collection_name"] = collections[0]["name"]

current_collection_id = st.session_state["active_collection_id"]
current_collection_name = st.session_state["active_collection_name"]

st.title("⚡ Smart Hybrid Search")
st.markdown(
    f"Active Workspace: **{current_collection_name}** | Architecture: **Dense (FAISS) + BM25 + RRF (k=60) + Cross-Encoder**"
)

# ---------------------------------------------------------------------------
# Index Management & Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def get_search_engines():
    """Initializes and caches search engines in Streamlit."""
    semantic = SemanticSearchEngine()
    bm25 = BM25SearchEngine()
    reranker = CrossEncoderReranker()
    hybrid = HybridSearchEngine(semantic_engine=semantic, bm25_engine=bm25, reranker=reranker)
    return semantic, bm25, reranker, hybrid


semantic_engine, bm25_engine, reranker_engine, hybrid_engine = get_search_engines()

# Sync index button in sidebar
with st.sidebar:
    st.subheader("🔍 Index Controls")
    if st.button("🔄 Sync & Rebuild Collection Index", use_container_width=True):
        with st.spinner("Indexing all collection passages into FAISS and BM25..."):
            all_chunks = db.get_all_chunks(current_collection_id)
            if all_chunks:
                semantic_engine.rebuild_index(current_collection_id, all_chunks)
                bm25_engine.index_chunks(current_collection_id, all_chunks)
                st.success(f"Indexed {len(all_chunks)} chunks!")
            else:
                st.warning("No chunks found in this collection to index.")
        st.rerun()

# ---------------------------------------------------------------------------
# Search Filters & Query Form
# ---------------------------------------------------------------------------

filter_container = st.container(border=True)
with filter_container:
    col_q, col_k = st.columns([4, 1])
    query = col_q.text_input(
        "Enter Research Query",
        placeholder="e.g. attention mechanism clinical document intelligence or link prediction",
    )
    top_k = col_k.number_input("Top Results", min_value=1, max_value=20, value=5)

    col_f1, col_f2, col_f3 = st.columns(3)

    # Paper Filter
    papers = db.list_papers(current_collection_id)
    paper_options = {"All Papers": None}
    for p in papers:
        pid = p.get("id")
        if pid:
            title_label = p.get("title") or p.get("filename", "Untitled Paper")
            paper_options[title_label] = pid
    selected_paper_label = col_f1.selectbox("Filter by Paper", options=list(paper_options.keys()))
    filter_paper_id = paper_options[selected_paper_label]

    # Section Filter
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
    selected_section = col_f2.selectbox("Filter by Section", options=section_options)
    filter_section = None if selected_section == "All Sections" else selected_section

    # Reranker Toggle
    enable_rerank = col_f3.checkbox("Enable Cross-Encoder Reranker", value=True)

# ---------------------------------------------------------------------------
# Query Execution & Results Display
# ---------------------------------------------------------------------------

def highlight_terms(text: str, query_str: str) -> str:
    """Highlights query terms in passage text with yellow background."""
    terms = [re.escape(t) for t in query_str.split() if len(t) > 2]
    if not terms:
        return text
    pattern = re.compile(r"(" + "|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark style='background-color: #FEF08A; padding: 1px 3px; border-radius: 2px;'>\1</mark>", text)


if query.strip():
    # Auto-ensure BM25 is populated if not yet in memory
    if current_collection_id not in bm25_engine._chunks:
        all_chunks = db.get_all_chunks(current_collection_id)
        if all_chunks:
            bm25_engine.index_chunks(current_collection_id, all_chunks)
            semantic_engine.rebuild_index(current_collection_id, all_chunks)

    with st.spinner("Executing hybrid retrieval with RRF and Cross-Encoder..."):
        results = hybrid_engine.search(
            collection_id=current_collection_id,
            query=query.strip(),
            top_k=top_k,
            top_k_dense=15,
            top_k_bm25=15,
            top_k_rrf=20,
            filter_paper_id=filter_paper_id,
            filter_section=filter_section,
            enable_reranker=enable_rerank,
        )

    st.subheader(f"Retrieved Passages ({len(results)} matches)")

    if not results:
        st.warning("No matching passages found. Try syncing the index or using broader query keywords.")
    else:
        for idx, res in enumerate(results):
            with st.container(border=True):
                head_col, score_col = st.columns([4, 1])

                with head_col:
                    page_label = f"Page {res.page_start}" if res.page_start == res.page_end else f"Pages {res.page_start}-{res.page_end}"
                    st.markdown(f"### #{idx + 1}. {res.paper_title}")
                    st.markdown(
                        f"`Section: {res.section_name}` | `📍 {page_label}` | `ID: {res.chunk_id}`"
                    )

                with score_col:
                    if res.rerank_score is not None:
                        st.metric("Rerank Score", f"{res.rerank_score:.3f}")
                    elif res.rrf_score is not None:
                        st.metric("RRF Score", f"{res.rrf_score:.4f}")

                # Passage text with highlighted query terms
                highlighted = highlight_terms(res.text, query)
                st.markdown(
                    f"<div style='background-color: #F8FAFC; border-left: 4px solid #2563EB; padding: 12px; margin-top: 8px; border-radius: 4px;'>{highlighted}</div>",
                    unsafe_allow_html=True,
                )

                # Expandable multi-stage score inspection
                with st.expander("🔍 Multi-Stage Score Breakdown"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.write(f"**Dense Score (Cosine):** {f'{res.dense_score:.4f}' if res.dense_score is not None else 'N/A'}")
                    c2.write(f"**BM25 Score:** {f'{res.bm25_score:.4f}' if res.bm25_score is not None else 'N/A'}")
                    c3.write(f"**RRF Score (k=60):** {f'{res.rrf_score:.6f}' if res.rrf_score is not None else 'N/A'}")
                    c4.write(f"**Cross-Encoder Score:** {f'{res.rerank_score:.4f}' if res.rerank_score is not None else 'N/A'}")
