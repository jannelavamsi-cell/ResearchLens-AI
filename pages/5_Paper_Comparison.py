"""
Streamlit Page 5: Paper Comparison & Benchmarking.
Side-by-side comparative matrix across 2 to 5 papers with physical page citations,
prominent comparability warnings, and grounded comparative narrative synthesis.
"""

import logging

import streamlit as st

from database.db_manager import DatabaseManager
from modules.paper_comparison import (
    build_comparison_matrix,
    check_comparability,
    generate_comparison_narrative,
    matrix_to_dataframe,
    matrix_to_markdown,
)
from modules.research_extraction import FIELD_DESCRIPTIONS
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.comparison")

st.set_page_config(
    page_title="Paper Comparison | ResearchLens AI",
    page_icon="⚖️",
    layout="wide",
)
inject_custom_css()

db = DatabaseManager()

st.sidebar.title("⚖️ Paper Comparison")

collections = db.list_collections()
col_options = {c.get("name", "Default"): c.get("id", "") for c in collections}

if not collections:
    st.warning("No collections found.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Workspace Collection", list(col_options.keys()))
active_col_id = col_options[selected_col_name]

papers = db.list_papers(active_col_id)
if len(papers) < 2:
    st.info("You need at least 2 uploaded papers in this collection to perform a comparative study.")
    st.markdown("👉 Head over to **Paper Upload** in the sidebar to upload more papers or load sample PDFs.")
    st.stop()

paper_map = {
    f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('year') or 'N/A'})": p.get("id")
    for p in papers
}
default_selected = list(paper_map.keys())[:min(3, len(paper_map))]

selected_labels = st.sidebar.multiselect(
    "Select Papers to Compare (2 to 5)",
    options=list(paper_map.keys()),
    default=default_selected,
)

if len(selected_labels) < 2:
    st.warning("Please select at least 2 papers to generate a comparative analysis.")
    st.stop()

if len(selected_labels) > 5:
    st.warning("Please select at most 5 papers at a time for optimal side-by-side readability.")
    selected_labels = selected_labels[:5]

selected_pids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]

# Main Page Header
st.title("⚖️ Cross-Paper Comparative Analysis")
st.caption(
    "Rigorous, side-by-side component comparison with physical page provenance "
    "and scientific comparability guards."
)

with st.spinner("Building comparative matrix..."):
    try:
        matrix = build_comparison_matrix(selected_pids, db)
        warning = check_comparability(matrix)
        df_matrix = matrix_to_dataframe(matrix)
    except Exception as err:
        st.error(f"Error constructing comparison matrix: {err}")
        st.stop()

# Comparability Warning Alert Banner
if warning.has_warning:
    with st.container(border=True):
        st.error(f"### {warning.message}")
        st.markdown("**Discrepancy Details:**")
        for r in warning.reasons:
            st.markdown(f"- ⚠️ {r}")
        st.caption(
            "Scientific Rigor Notice: Academic machine learning models evaluated on different "
            "datasets or varying metric definitions cannot be ranked by raw numerical scores."
        )
else:
    st.success("✅ **Fair Benchmark Alignment:** The compared papers evaluate on identical or directly comparable benchmarks.")

# Tabs: Grid View, Component Breakdown, Narrative Synthesis
tab_table, tab_cards, tab_narrative = st.tabs([
    "📊 Comparative Matrix",
    "🔍 Component Deep-Dive",
    "📝 Comparative Narrative",
])

with tab_table:
    st.subheader("Side-by-Side Research Component Matrix")
    st.caption("Each cell contains the extracted findings along with exact physical page citation pills `[p. X]`.")
    st.dataframe(df_matrix, use_container_width=True, height=500)

    # Export Buttons
    exp_col1, exp_col2, _ = st.columns([1, 1, 2])
    with exp_col1:
        csv_bytes = df_matrix.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Matrix (CSV)",
            data=csv_bytes,
            file_name="researchlens_paper_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        narrative_preview = "Comparative study of selected research literature."
        md_content = matrix_to_markdown(matrix, warning, narrative_preview)
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=md_content.encode("utf-8"),
            file_name="researchlens_comparison_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

with tab_cards:
    st.subheader("Field-by-Field Component Comparison")
    rows = matrix.get("rows", {})
    papers_meta = matrix.get("papers", [])

    for field_name, cells in rows.items():
        with st.expander(f"📌 {field_name.replace('_', ' ').title()} — {FIELD_DESCRIPTIONS.get(field_name, '')}", expanded=False):
            card_cols = st.columns(len(cells))
            for idx, cell in enumerate(cells):
                p = papers_meta[idx] if idx < len(papers_meta) else {}
                with card_cols[idx]:
                    st.markdown(f"**{p.get('title', 'Untitled Paper')}**")
                    st.caption(f"Authors: {p.get('authors', 'Unknown')} | Year: {p.get('year', 'N/A')}")

                    if cell.value == "Not stated in paper":
                        st.info("⚪ *Not stated in paper*")
                    else:
                        st.markdown(f"> {cell.value}")
                        if cell.page > 0:
                            st.caption(f"📍 **Cited Page {cell.page}**")

                    if cell.evidence_quote and cell.value != "Not stated in paper":
                        st.caption(f"*Quote:* \"{cell.evidence_quote[:120]}...\"")

with tab_narrative:
    st.subheader("Synthesized Comparative Literature Narrative")
    st.caption("Grounded, multi-perspective synthesis highlighting trade-offs and benchmark caveats.")

    if "comparison_narrative" not in st.session_state:
        st.session_state.comparison_narrative = ""

    synth_btn = st.button("✨ Synthesize Comparative Narrative", type="primary")

    if synth_btn or not st.session_state.comparison_narrative:
        with st.spinner("Analyzing comparative dimensions and synthesizing narrative..."):
            st.session_state.comparison_narrative = generate_comparison_narrative(matrix, warning)

    st.markdown(st.session_state.comparison_narrative)

    # Full report download including narrative
    full_md = matrix_to_markdown(matrix, warning, st.session_state.comparison_narrative)
    st.download_button(
        label="📥 Download Full Comparative Analysis (Markdown)",
        data=full_md.encode("utf-8"),
        file_name="comparative_literature_study.md",
        mime="text/markdown",
    )
