"""
ResearchLens AI - Intelligent Research-Paper Analysis & Comparative Literature Review System.
Main Streamlit application entry point and workspace dashboard.
"""

import streamlit as st

from core.config import settings
from core.llm import get_llm_client
from core.logging_config import setup_logging
from database.db_manager import DBManager

# Initialize logging and database
setup_logging()
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: Collection Management & Provider Health Check
# ---------------------------------------------------------------------------

st.sidebar.title("🔬 ResearchLens AI")
st.sidebar.markdown("**Grounded Literature Review & Analysis**")
st.sidebar.divider()

# Collections
collections = db.list_collections()
collection_names = [c.get("name", "Default Workspace") for c in collections]
if not collection_names:
    db.create_collection("Default Workspace", "Initial research collection")
    collections = db.list_collections()
    collection_names = [c.get("name", "Default Workspace") for c in collections]

# Manage active collection in session state
if "active_collection_id" not in st.session_state:
    st.session_state["active_collection_id"] = collections[0].get("id", "")
    st.session_state["active_collection_name"] = collections[0].get("name", "Default Workspace")

selected_name = st.sidebar.selectbox(
    "Active Collection",
    options=collection_names,
    index=collection_names.index(st.session_state["active_collection_name"])
    if st.session_state["active_collection_name"] in collection_names
    else 0,
)

# Sync selection
for c in collections:
    if c.get("name") == selected_name:
        st.session_state["active_collection_id"] = c.get("id", "")
        st.session_state["active_collection_name"] = c.get("name", selected_name)
        break

# Create new collection expander
with st.sidebar.expander("➕ New Collection", expanded=False):
    new_coll_name = st.text_input("Collection Name", key="new_coll_input")
    new_coll_desc = st.text_input("Description (optional)", key="new_coll_desc")
    if st.button("Create", use_container_width=True):
        if new_coll_name.strip():
            new_id = db.create_collection(new_coll_name.strip(), new_coll_desc.strip())
            st.session_state["active_collection_id"] = new_id
            st.session_state["active_collection_name"] = new_coll_name.strip()
            st.success(f"Created '{new_coll_name}'")
            st.rerun()

st.sidebar.divider()

# LLM Health & Settings in Sidebar
st.sidebar.markdown("### ⚙️ System Status")
llm_client = get_llm_client()
health = llm_client.health_check()

h_status = health.get("status", "error")
h_model = health.get("model", "Unknown")
h_msg = health.get("message", "")

if h_status == "healthy":
    st.sidebar.success(
        f"**LLM Connected**: {h_model}\n\n*Provider: {settings.llm_provider.upper()}*"
    )
elif h_status == "warning":
    st.sidebar.warning(f"**LLM Notice**: {h_msg}")
elif h_status == "unconfigured":
    st.sidebar.warning(f"**LLM Unconfigured**: {h_msg}")
else:
    st.sidebar.error(f"**LLM Error**: {h_msg}")

with st.sidebar.expander("🔧 Model Configuration", expanded=False):
    st.markdown(f"**Provider**: `{settings.llm_provider}`")
    st.markdown(f"**Embedding**: `{settings.embedding_model_name}`")
    st.markdown(
        f"**Reranker**: `{settings.reranker_model_name if settings.enable_reranker else 'Disabled'}`"
    )
    st.markdown(f"**Database**: `{settings.database_path}`")
    st.caption("Change settings in `.env` or set environment variables.")

# ---------------------------------------------------------------------------
# Main Page Dashboard & Overview
# ---------------------------------------------------------------------------

st.title("🔬 Welcome to ResearchLens AI")
st.markdown(
    """
    **ResearchLens AI** is an intelligent, evidence-grounded research-paper analysis and comparative
    literature review system. Every extracted finding, answer, comparison cell, and research gap is
    strictly verifiable and traced to source paper pages.
    """
)

# Active Collection Stats Bar
current_col_id = st.session_state["active_collection_id"]
papers = db.list_papers(current_col_id)
ready_papers = [p for p in papers if p.get("status") == "ready"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Collection", st.session_state["active_collection_name"])
c2.metric("Total Papers", len(papers))
c3.metric("Ready for Analysis", len(ready_papers))
c4.metric("LLM Provider", settings.llm_provider.upper())

st.divider()

# Module Navigation Grid
st.markdown("### 📑 Platform Capabilities")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("1. 📄 Paper Upload & Ingestion")
    st.write(
        "Upload multi-page research PDFs with two-column reading-order parsing, "
        "section detection, dehyphenation, SHA-256 deduplication, and OCR fallback."
    )

    st.subheader("2. 🔍 Structured Paper Analysis")
    st.write(
        "Extract 9 grounded components (objective, methodology, datasets, results, limitations, etc.) "
        "with page citations, fuzzy quote verification badges, and researcher overrides."
    )

    st.subheader("3. ⚡ Smart Hybrid Search")
    st.write(
        "Execute hybrid semantic search fusing FAISS dense embeddings and BM25 lexical ranking "
        "via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking."
    )

    st.subheader("4. 💬 Grounded Research Chat")
    st.write(
        "Ask questions across individual papers or collections. Responses include inline citations "
        "`[Paper N, p. X]` and automated claim-support verification."
    )

with col_b:
    st.subheader("5. 📊 Multi-Paper Comparison")
    st.write(
        "Generate side-by-side matrices comparing methodologies, datasets, and results. "
        "Includes a comparability warning guard when papers use distinct benchmarks."
    )

    st.subheader("6. 📝 Literature Review Synthesis")
    st.write(
        "Synthesize structured review drafts with thematic clustering, methodological comparisons, "
        "and auto-formatted bibliographies in IEEE, APA, and BibTeX styles."
    )

    st.subheader("7. 💡 Research Gap Opportunities")
    st.write(
        "Aggregate author-declared limitations across papers to discover unaddressed research gaps "
        "with direct evidence provenance and proposed directions."
    )

    st.subheader("8. 📈 Trends & Export Center")
    st.write(
        "Interactive Plotly visualizations for keyword frequencies, publication timelines, and "
        "similarity heatmaps, plus export to Markdown, DOCX, and PDF."
    )

st.divider()
st.info(
    "👉 Get started by navigating to **1_Paper_Upload** in the sidebar to upload your research papers."
)
