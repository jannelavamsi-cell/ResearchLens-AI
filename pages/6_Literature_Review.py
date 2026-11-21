"""
Streamlit Page 6: Literature Review Synthesis.
Drafts multi-paper literature reviews structured into Introduction, Thematic Synthesis,
Methodological Comparison, Research Gaps, and Conclusion, accompanied by IEEE/APA/BibTeX references.
"""

import logging

import streamlit as st

from database.db_manager import DatabaseManager
from modules.literature_review import (
    review_to_markdown,
    synthesize_literature_review,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.lit_review")

st.set_page_config(
    page_title="Literature Review | ResearchLens AI",
    page_icon="📖",
    layout="wide",
)
inject_custom_css()

db = DatabaseManager()

st.sidebar.title("📖 Literature Review")

collections = db.list_collections()
col_options = {c.get("name", "Default"): c.get("id", "") for c in collections}

if not collections:
    st.warning("No collections found.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Workspace Collection", list(col_options.keys()))
active_col_id = col_options[selected_col_name]

papers = db.list_papers(active_col_id)
if not papers:
    st.info("No papers available in this collection.")
    st.markdown("👉 Head over to **Paper Upload** in the sidebar to upload research papers or load sample PDFs.")
    st.stop()

# Selection Scope
st.sidebar.subheader("Review Scope")
scope = st.sidebar.radio("Scope", ["All Papers in Collection", "Select Specific Papers"])
selected_pids = None

if scope == "Select Specific Papers":
    paper_map = {
        f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('year') or 'N/A'})": p.get("id")
        for p in papers
    }
    selected_labels = st.sidebar.multiselect("Select Target Papers", list(paper_map.keys()), default=list(paper_map.keys())[:2])
    if not selected_labels:
        st.sidebar.warning("Please select at least 1 paper.")
        st.stop()
    selected_pids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]

ref_style = st.sidebar.selectbox("Citation Style", ["IEEE", "APA", "BibTeX"])

st.title("📖 Academic Literature Review Draft")
st.caption(
    "Automated, thematic literature review synthesis with physical page citations "
    "and verifiable bibliographic references."
)

# Mandatory Academic Disclaimer Banner
st.warning(
    "⚠️ **Mandatory Academic Notice:** This is an AI-generated synthesis draft. "
    "Researchers must independently verify all citations and claims against original source papers "
    "before academic submission or publication."
)

if "lit_review_draft" not in st.session_state:
    st.session_state.lit_review_draft = None

generate_btn = st.button("✨ Synthesize Literature Review", type="primary")

if generate_btn or st.session_state.lit_review_draft is None:
    with st.spinner("Analyzing thematic clusters and drafting literature review..."):
        try:
            draft = synthesize_literature_review(
                collection_id=active_col_id,
                paper_ids=selected_pids,
                db=db,
                ref_style=ref_style.lower(),
            )
            st.session_state.lit_review_draft = draft
        except Exception as err:
            st.error(f"Error generating literature review: {err}")
            st.stop()

draft = st.session_state.lit_review_draft

# Review Tabs
tab_full, tab_themes, tab_methods, tab_gaps, tab_refs = st.tabs([
    "📑 Full Review",
    "🎯 Thematic Synthesis",
    "⚙️ Methodological Comparison",
    "🚀 Research Gaps",
    "📚 Bibliographic References",
])

with tab_full:
    st.subheader(draft.title)
    st.markdown("### 1. Introduction")
    st.markdown(draft.introduction)

    st.markdown("### 2. Thematic Synthesis")
    for t_name, t_text in draft.thematic_synthesis.items():
        st.markdown(f"#### {t_name}")
        st.markdown(t_text)

    st.markdown("### 3. Methodological Comparison")
    st.markdown(draft.methodological_comparison)

    st.markdown("### 4. Research Gaps & Open Challenges")
    st.markdown(draft.research_gaps)

    st.markdown("### 5. Conclusion")
    st.markdown(draft.conclusion)

    # Export Download Button
    md_text = review_to_markdown(draft)
    st.download_button(
        label="📥 Download Complete Literature Review (Markdown)",
        data=md_text.encode("utf-8"),
        file_name="literature_review_draft.md",
        mime="text/markdown",
    )

with tab_themes:
    st.subheader("Thematic Synthesis by Cluster")
    for t_name, t_text in draft.thematic_synthesis.items():
        with st.container(border=True):
            st.markdown(f"#### {t_name}")
            st.markdown(t_text)

with tab_methods:
    st.subheader("Methodological Comparison")
    st.markdown(draft.methodological_comparison)

with tab_gaps:
    st.subheader("Synthesized Research Gaps")
    st.markdown(draft.research_gaps)

with tab_refs:
    st.subheader(f"Formatted References ({ref_style.upper()})")
    if ref_style.lower() == "bibtex":
        for r in draft.references:
            st.code(r, language="bibtex")
    else:
        for _idx, r in enumerate(draft.references, 1):
            st.markdown(f"{r}")
