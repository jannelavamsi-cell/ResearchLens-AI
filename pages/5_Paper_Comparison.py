"""
ResearchLens AI - Paper Comparison & Benchmarking Page.
Side-by-side comparative matrix across 2 to 5 papers with physical page citations,
prominent comparability warnings, and grounded comparative narrative synthesis.
"""

import html
import logging

import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.paper_comparison import (
    build_comparison_matrix,
    check_comparability,
    generate_comparison_narrative,
    matrix_to_dataframe,
    matrix_to_markdown,
)
from modules.research_extraction import FIELD_DESCRIPTIONS
from ui.components import (
    banner,
    citation_pill,
    empty_state,
    evidence_panel,
    page_header,
    safe_page_render,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.comparison")
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="Paper Comparison | ResearchLens AI",
    page_icon="⚖️",
    layout="wide",
)
inject_custom_css()


@safe_page_render("Paper Comparison & Benchmarking")
def main():
    collections = db.list_collections()
    if not collections:
        empty_state("No Collections Found", "Create a workspace collection from the library first.", icon="📚")
        return

    col_map = {c["name"]: c["id"] for c in collections}
    if "active_collection_id" not in st.session_state or st.session_state["active_collection_id"] not in col_map.values():
        st.session_state["active_collection_id"] = collections[0]["id"]
        st.session_state["active_collection_name"] = collections[0]["name"]

    active_col_id = st.session_state["active_collection_id"]
    active_col_name = st.session_state["active_collection_name"]

    papers = db.list_papers(active_col_id)

    page_header(
        title="Cross-Paper Comparative Analysis",
        subtitle=f"Side-by-side benchmarking and component contrast across literature in {active_col_name}",
        badge_text="Grounded Matrix + Comparability Guard",
    )

    if len(papers) < 2:
        empty_state(
            title="At Least 2 Papers Required",
            message=f"The active workspace '{active_col_name}' currently has {len(papers)} paper(s). Upload at least 2 papers to perform comparative analysis.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="⚖️",
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚖️ Comparison Scope")
        st.caption(f"Workspace: **{active_col_name}** ({len(papers)} papers)")

        paper_map = {
            f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('year') or 'N/A'})": p.get("id")
            for p in papers
        }
        default_selected = list(paper_map.keys())[: min(3, len(paper_map))]

        selected_labels = st.multiselect(
            "Select Papers to Compare (2 to 5)",
            options=list(paper_map.keys()),
            default=default_selected,
        )

        if len(selected_labels) < 2:
            st.warning("Select at least 2 papers to generate a comparative analysis.")
            return

        if len(selected_labels) > 5:
            st.warning("Display capped at 5 papers for optimal side-by-side readability.")
            selected_labels = selected_labels[:5]

        selected_pids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]

    # -----------------------------------------------------------------------
    # Build Matrix & Check Comparability
    # -----------------------------------------------------------------------
    with st.spinner("Building comparative matrix and validating benchmark alignment..."):
        try:
            matrix = build_comparison_matrix(selected_pids, db)
            warning = check_comparability(matrix)
            df_matrix = matrix_to_dataframe(matrix)
        except Exception as err:
            st.error(f"Error constructing comparison matrix: {err}")
            return

    # Comparability Alert Banner
    if warning.has_warning:
        banner(
            alert_type="warning",
            title=warning.message,
            message="Machine learning models and empirical pipelines evaluated on divergent datasets or distinct metrics cannot be directly ranked by raw metrics:",
            details=warning.reasons,
        )
    else:
        banner(
            alert_type="success",
            title="Fair Benchmark Alignment Verified",
            message="The selected papers evaluate across consistent benchmarks and metric definitions, allowing fair comparative analysis.",
        )

    # -----------------------------------------------------------------------
    # Multi-Tab Comparative Presentation
    # -----------------------------------------------------------------------
    tab_matrix, tab_cards, tab_narrative = st.tabs([
        "📊 Comparative Matrix",
        "🔍 Component Deep-Dive",
        "📝 Comparative Narrative",
    ])

    # Tab 1: Structured Grid Table
    with tab_matrix:
        st.markdown("#### Side-by-Side Research Component Matrix")
        st.caption("Each cell includes extracted findings along with verifiable physical page citations [p. X].")

        st.dataframe(df_matrix, use_container_width=True, height=480)

        exp_c1, exp_c2, _ = st.columns([1.2, 1.4, 2])
        with exp_c1:
            csv_bytes = df_matrix.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"comparison_matrix_{active_col_name[:12]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_c2:
            narrative_preview = "Comparative study of selected research literature."
            md_content = matrix_to_markdown(matrix, warning, narrative_preview)
            st.download_button(
                label="📥 Download Markdown Table",
                data=md_content.encode("utf-8"),
                file_name=f"comparison_table_{active_col_name[:12]}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Tab 2: Component Deep-Dive
    with tab_cards:
        st.markdown("#### Field-by-Field Component Comparison")
        rows = matrix.get("rows", {})
        papers_meta = matrix.get("papers", [])

        for field_name, cells in rows.items():
            field_label = field_name.replace("_", " ").title()
            field_desc = FIELD_DESCRIPTIONS.get(field_name, "")
            with st.expander(f"📌 {field_label} — {field_desc}", expanded=False):
                card_cols = st.columns(len(cells))
                for idx, cell in enumerate(cells):
                    p = papers_meta[idx] if idx < len(papers_meta) else {}
                    with card_cols[idx]:
                        st.markdown(
                            f"""
                            <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                                <div style="font-weight: 600; font-size: 0.95rem; color: var(--rl-primary); margin-bottom: 4px;">
                                    {html.escape(p.get('title', 'Untitled Paper'))}
                                </div>
                                <div style="font-size: 0.75rem; color: var(--rl-text-muted); margin-bottom: 8px;">
                                    {html.escape(p.get('authors', 'Unknown'))} ({p.get('year', 'N/A')})
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if cell.value == "Not stated in paper":
                            st.caption("⚪ *Not stated in paper*")
                        else:
                            st.markdown(f"> {cell.value}")
                            if cell.page > 0:
                                pill_html = citation_pill(cell.page)
                                st.markdown(f"<div style='margin-top: 4px;'>{pill_html}</div>", unsafe_allow_html=True)

                        if cell.evidence_quote and cell.value != "Not stated in paper":
                            evidence_panel(cell.evidence_quote, cell.page)

    # Tab 3: Synthesized Narrative
    with tab_narrative:
        st.markdown("#### Synthesized Comparative Literature Narrative")
        st.caption("Multi-perspective, grounded technical comparison highlighting trade-offs and methodological differences.")

        if "comparison_narrative" not in st.session_state:
            st.session_state.comparison_narrative = ""

        narr_col1, narr_col2 = st.columns([1.5, 3])
        with narr_col1:
            synth_btn = st.button("✨ Synthesize Comparative Narrative", type="primary", use_container_width=True)

        if synth_btn or not st.session_state.comparison_narrative:
            with st.spinner("Analyzing comparative dimensions and synthesizing narrative..."):
                st.session_state.comparison_narrative = generate_comparison_narrative(matrix, warning)

        st.markdown(
            f"""
            <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 12px; padding: 1.5rem; margin-top: 1rem; line-height: 1.7;">
                {st.session_state.comparison_narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        full_md = matrix_to_markdown(matrix, warning, st.session_state.comparison_narrative)
        st.download_button(
            label="📥 Download Full Comparative Study (Markdown)",
            data=full_md.encode("utf-8"),
            file_name=f"comparative_literature_study_{active_col_name[:12]}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
