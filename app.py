"""
ResearchLens AI - Intelligent Research-Paper Analysis & Comparative Literature Review System.
Main Streamlit application entry point and workspace dashboard.
"""

import streamlit as st

from core.config import settings
from core.llm import get_llm_client
from core.logging_config import setup_logging
from database.db_manager import DBManager
from ui.components import render_footer, render_header, render_metric_card
from ui.theme import inject_custom_css

# Initialize logging and database
setup_logging()
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

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

render_header(
    title="ResearchLens AI",
    subtitle="Evidence-grounded scientific document intelligence and comparative literature review platform.",
    icon="🔬",
)

# Active Collection Stats Bar
current_col_id = st.session_state["active_collection_id"]
papers = db.list_papers(current_col_id)
ready_papers = [p for p in papers if p.get("status") == "ready"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Active Collection", st.session_state["active_collection_name"], "Workspace Scope")
with c2:
    render_metric_card("Total Papers", len(papers), "Uploaded in Library")
with c3:
    render_metric_card("Ready for Analysis", len(ready_papers), "Fully Processed")
with c4:
    render_metric_card("LLM Provider", settings.llm_provider.upper(), f"Model: {health.get('model', 'Active')}")

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# Module Navigation Grid
st.markdown("### 📑 Platform Capabilities")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">1. 📄 Paper Upload & Ingestion</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Upload multi-page research PDFs with two-column reading-order parsing, section detection, dehyphenation, SHA-256 deduplication, and OCR fallback.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">2. 🔍 Structured Paper Analysis</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Extract 9 grounded components (objective, methodology, datasets, results, limitations, etc.) with page citations, fuzzy quote verification badges, and researcher overrides.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">3. ⚡ Smart Hybrid Search</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Execute hybrid semantic search fusing FAISS dense embeddings and BM25 lexical ranking via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">4. 💬 Grounded Research Chat</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Ask questions across individual papers or collections. Responses include inline citations <code>[Paper N, p. X]</code> and automated claim-support verification.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">5. 📊 Multi-Paper Comparison</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Generate side-by-side matrices comparing methodologies, datasets, and results. Includes a comparability warning guard when papers use distinct benchmarks.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">6. 📝 Literature Review Synthesis</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Synthesize structured review drafts with thematic clustering, methodological comparisons, and auto-formatted bibliographies in IEEE, APA, and BibTeX styles.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">7. 💡 Research Gap Opportunities</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Aggregate author-declared limitations across papers to discover unaddressed research gaps with direct evidence provenance and proposed directions.</p>
        </div>
        <div class="rl-card">
            <div class="rl-card-header"><span class="rl-card-title">8. 📈 Trends & Export Center</span></div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0;">Interactive Plotly visualizations for keyword frequencies, publication timelines, and similarity heatmaps, plus export to Markdown, DOCX, and PDF.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer()
