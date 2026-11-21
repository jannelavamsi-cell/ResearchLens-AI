"""
ResearchLens AI - Research Gaps & Opportunity Discovery Page.
Aggregates author-declared limitations and future work across research literature,
quantifies challenge recurrence frequency, and synthesizes grounded research proposals.
"""

import html
import logging

import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.research_gap import discover_research_gaps
from ui.components import (
    citation_pill,
    empty_state,
    evidence_panel,
    page_header,
    safe_page_render,
    stat_card,
    status_badge,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.gaps")
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="Research Gaps | ResearchLens AI",
    page_icon="🚀",
    layout="wide",
)
inject_custom_css()


@safe_page_render("Research Gap Discovery")
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
        title="Research Gaps & Opportunity Discovery",
        subtitle=f"Synthesizes author-acknowledged limitations, boundary conditions, and recurrent challenges across {active_col_name}",
        badge_text="Limitation Mining + Thesis Proposals",
    )

    if not papers:
        empty_state(
            title="No Research Papers Available",
            message=f"The active workspace '{active_col_name}' contains no papers. Upload papers and extract components to discover research gaps.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="📄",
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar: Summary & Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🚀 Discovery Controls")
        st.caption(f"Workspace: **{active_col_name}** ({len(papers)} papers)")
        if st.button("🔄 Refresh Gap Analysis", use_container_width=True, type="secondary"):
            st.rerun()

    # -----------------------------------------------------------------------
    # Discover Gaps
    # -----------------------------------------------------------------------
    with st.spinner("Analyzing author-declared limitations and computing recurrence frequency..."):
        gap_items = discover_research_gaps(active_col_id, db=db)

    if not gap_items:
        empty_state(
            title="No Research Gaps Detected",
            message="No author-declared limitations or future work statements were detected in this collection yet. Make sure structured extraction has completed on the Paper Analysis page.",
            action_label="Go to Paper Analysis",
            action_page="pages/2_Paper_Analysis.py",
            icon="🔍",
        )
        return

    # -----------------------------------------------------------------------
    # Header KPI Metric Cards
    # -----------------------------------------------------------------------
    multi_paper_count = sum(1 for g in gap_items if g.frequency_count > 1)
    highest_freq = max(g.frequency_count for g in gap_items) if gap_items else 0

    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        stat_card(
            label="Identified Gap Domains",
            value=str(len(gap_items)),
            delta="Extracted from limitation sections",
            icon="🚀",
        )
    with met_col2:
        stat_card(
            label="Multi-Paper Recurrent Gaps",
            value=str(multi_paper_count),
            delta="Shared across >= 2 papers",
            icon="🔥",
        )
    with met_col3:
        stat_card(
            label="Peak Challenge Recurrence",
            value=f"{highest_freq} papers",
            delta="Most persistent bottleneck",
            icon="📈",
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Action Row: Export Report
    # -----------------------------------------------------------------------
    act_col1, act_col2 = st.columns([3.5, 1.5])
    with act_col1:
        st.markdown(f"### 📋 Discovered Opportunity Landscape (`{len(gap_items)}` gaps ranked by frequency)")
    with act_col2:
        # Prepare Markdown export
        md_report_lines = [
            "# ResearchLens AI: Grounded Research Gaps Report",
            f"**Workspace Collection:** {active_col_name}  ",
            f"**Total Identified Domains:** {len(gap_items)}  ",
            f"**Recurrent Challenges:** {multi_paper_count}  ",
            "\n---\n",
        ]
        for idx, g in enumerate(gap_items, 1):
            md_report_lines.append(f"## {idx}. {g.gap_statement}\n")
            md_report_lines.append(f"- **Recurrence:** {g.frequency_count} paper(s)\n")
            md_report_lines.append(f"- **Suggested Direction:** {g.suggested_direction}\n")
            md_report_lines.append(f"- **Rationale:** {g.rationale}\n")
            md_report_lines.append("### Supporting Author Evidence:")
            for sp in g.supporting_papers:
                sp_title = sp.get("paper_title", "Unknown Paper")
                sp_page = sp.get("page", "?")
                sp_quote = sp.get("limitation_quote", "")
                md_report_lines.append(f"- **{sp_title}** (p. {sp_page}): \"{sp_quote}\"")
            md_report_lines.append("\n---\n")

        st.download_button(
            label="📥 Download Gaps Report (MD)",
            data="\n".join(md_report_lines).encode("utf-8"),
            file_name=f"research_gaps_{active_col_name[:12]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Render Gap Opportunity Cards
    # -----------------------------------------------------------------------
    for idx, gap in enumerate(gap_items, 1):
        with st.container(border=True):
            header_col1, header_col2 = st.columns([3.2, 1.8])

            with header_col1:
                st.markdown(
                    f"""
                    <div style="font-size: 1.15rem; font-weight: 700; color: var(--rl-primary); margin-bottom: 0.25rem;">
                        {idx}. {html.escape(gap.gap_statement)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with header_col2:
                if gap.frequency_count > 1:
                    freq_badge = status_badge("warning", f"🔥 Recurrent in {gap.frequency_count} Papers")
                else:
                    freq_badge = status_badge("neutral", "📌 Single Paper")

                st.markdown(
                    f"<div style='text-align: right;'>{freq_badge}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="background: var(--rl-surface); border-left: 4px solid var(--rl-primary); padding: 10px 14px; border-radius: 4px; margin: 0.5rem 0 0.8rem 0;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: var(--rl-text-muted); text-transform: uppercase;">Actionable Research Direction:</div>
                    <div style="font-size: 0.95rem; font-weight: 500; color: var(--rl-text-primary); margin-top: 2px;">{html.escape(gap.suggested_direction)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(f"**Scientific Rationale:** {gap.rationale}")

            # Supporting Papers Accordion
            with st.expander(f"📚 View Author-Declared Evidence Excerpts ({len(gap.supporting_papers)} Paper citations)"):
                for sp in gap.supporting_papers:
                    sp_title = sp.get("paper_title", "Unknown Paper")
                    sp_page = sp.get("page", 1)
                    sp_quote = sp.get("limitation_quote", "")
                    pill_html = citation_pill(sp_page)

                    st.markdown(
                        f"""
                        <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-weight: 600; font-size: 0.9rem; color: var(--rl-text-primary);">{html.escape(sp_title)}</span>
                                {pill_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    evidence_panel(sp_quote, sp_page)

            # Draft Thesis / Proposal Paragraph Generator
            with st.expander("📝 Generate Proposal / Thesis Abstract Draft"):
                cite_tokens = ", ".join([f"{sp.get('paper_title', 'Unknown Paper')} (p. {sp.get('page', '?')})" for sp in gap.supporting_papers])
                first_sp = gap.supporting_papers[0] if gap.supporting_papers else {}
                first_title = first_sp.get("paper_title", "Prior literature")
                first_quote = first_sp.get("limitation_quote", "unresolved challenges remain")
                proposal_draft = (
                    f"While significant recent strides have been reported across the literature ({cite_tokens}), "
                    f"a central unresolved challenge remains: {gap.gap_statement.lower()} "
                    f"Specifically, {first_title} acknowledges that '{first_quote}' "
                    f"To overcome these explicit boundaries, this proposed work investigates: {gap.suggested_direction.lower()} "
                    f"This approach is motivated by the necessity to {gap.rationale.lower()}"
                )
                st.text_area(
                    "Draft Proposal Text (Ready for Thesis or Grant Application)",
                    value=proposal_draft,
                    height=130,
                    key=f"proposal_text_{idx}",
                )


if __name__ == "__main__":
    main()
