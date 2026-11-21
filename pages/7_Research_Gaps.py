"""
Streamlit Page 7: Research Gaps & Opportunity Discovery.
Aggregates author-declared limitations and future work across research literature,
quantifies challenge recurrence frequency, and synthesizes grounded research proposals.
"""

import logging

import streamlit as st

from database.db_manager import DatabaseManager
from modules.research_gap import discover_research_gaps
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.gaps")

st.set_page_config(
    page_title="Research Gaps | ResearchLens AI",
    page_icon="🚀",
    layout="wide",
)
inject_custom_css()

db = DatabaseManager()

st.sidebar.title("🚀 Research Gaps")

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

st.title("🚀 Research Gaps & Opportunity Discovery")
st.caption(
    "Synthesizes author-acknowledged limitations, boundary conditions, and future directions "
    "into grounded research project and thesis opportunities."
)

with st.spinner("Analyzing author-declared limitations and computing challenge recurrence..."):
    gap_items = discover_research_gaps(active_col_id, db=db)

if not gap_items:
    st.info("No author-declared limitations or future work statements detected in this collection yet.")
    st.markdown("👉 Make sure extractions are computed on the **Paper Analysis** page.")
    st.stop()

# Header Metrics
met_col1, met_col2, met_col3 = st.columns(3)
with met_col1:
    st.metric("Total Identified Gap Domains", len(gap_items))
with met_col2:
    multi_paper_count = sum(1 for g in gap_items if g.frequency_count > 1)
    st.metric("Multi-Paper Recurrent Gaps", multi_paper_count)
with met_col3:
    highest_freq = max(g.frequency_count for g in gap_items) if gap_items else 0
    st.metric("Peak Challenge Recurrence", f"{highest_freq} papers")

st.markdown("---")

# Render Gap Cards
for idx, gap in enumerate(gap_items, 1):
    with st.container(border=True):
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(f"### {idx}. {gap.gap_statement}")
        with header_col2:
            badge_color = "🔥" if gap.frequency_count > 1 else "📌"
            st.markdown(f"**{badge_color} Mentioned in {gap.frequency_count} Paper(s)**")

        st.markdown(f"**Actionable Research Direction:**\n> {gap.suggested_direction}")
        st.caption(f"**Scientific Rationale:** {gap.rationale}")

        # Supporting Papers Accordion
        with st.expander(f"📚 View Author-Declared Evidence Excerpts ({len(gap.supporting_papers)} Paper citations)"):
            for sp in gap.supporting_papers:
                sp_title = sp.get("paper_title", "Unknown Paper")
                sp_page = sp.get("page", "?")
                sp_quote = sp.get("limitation_quote", "")
                st.markdown(f"**{sp_title}** (Page {sp_page}):")
                st.info(f'"{sp_quote}"')

        # Draft Thesis / Proposal Paragraph
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
            st.text_area("Draft Proposal Text (Ready for Thesis or Grant Application)", value=proposal_draft, height=120)

# Export Full Gap Analysis Report
md_report_lines = ["# ResearchLens AI: Grounded Research Gaps Report\n"]
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
    label="📥 Download Research Gaps Analysis (Markdown)",
    data="\n".join(md_report_lines).encode("utf-8"),
    file_name="researchlens_gaps_analysis.md",
    mime="text/markdown",
)
