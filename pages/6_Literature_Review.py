"""
ResearchLens AI - Academic Literature Review Synthesis Page.
Drafts multi-paper literature reviews structured into Introduction, Thematic Synthesis,
Methodological Comparison, Research Gaps, and Conclusion, accompanied by IEEE/APA/BibTeX references.
"""

import logging
from datetime import datetime
from typing import List, Optional

import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.literature_review import (
    review_to_markdown,
    synthesize_literature_review,
)
from ui.components import (
    banner,
    empty_state,
    page_header,
    safe_page_render,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.lit_review")
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="Literature Review | ResearchLens AI",
    page_icon="📖",
    layout="wide",
)
inject_custom_css()


@safe_page_render("Literature Review Synthesis")
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
        title="Academic Literature Review Draft",
        subtitle=f"Automated thematic clustering and synthesis with verifiable citations across {active_col_name}",
        badge_text="Thematic Clustering + Multi-Style Citations",
    )

    if not papers:
        empty_state(
            title="No Research Papers Available",
            message=f"The active workspace '{active_col_name}' contains no papers. Upload research papers to synthesize a literature review.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="📄",
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar: Scope, Citation Style & Triggers
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📖 Review Configuration")
        st.caption(f"Workspace: **{active_col_name}** ({len(papers)} papers)")

        st.markdown("#### 🎯 Target Papers")
        scope = st.radio("Review Scope", ["All Papers in Workspace", "Select Specific Papers"], label_visibility="collapsed")
        selected_pids: Optional[List[str]] = None

        if scope == "Select Specific Papers":
            paper_map = {
                f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('year') or 'N/A'})": p.get("id")
                for p in papers
            }
            default_selection = list(paper_map.keys())[: min(3, len(paper_map))]
            selected_labels = st.multiselect("Target Papers", list(paper_map.keys()), default=default_selection)
            if not selected_labels:
                st.warning("Please select at least 1 paper.")
                return
            selected_pids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]

        st.markdown("#### 🖋️ Citation Style")
        ref_style = st.selectbox("Style", ["IEEE", "APA", "BibTeX"], index=0, label_visibility="collapsed")

        st.markdown("---")
        generate_btn = st.button("✨ Synthesize Literature Review", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Mandatory Academic Notice Banner
    # -----------------------------------------------------------------------
    banner(
        alert_type="warning",
        title="Mandatory Academic & Peer-Review Notice",
        message="This literature review is an AI-synthesized draft intended for exploratory outlining. Researchers and authors must independently verify all cited claims, statistical findings, and bibliographic entries against original publication PDFs before submission or publication.",
    )

    # -----------------------------------------------------------------------
    # Review Synthesis State & Execution
    # -----------------------------------------------------------------------
    review_cache_key = f"lit_review_{active_col_id}_{ref_style}_{len(selected_pids) if selected_pids else len(papers)}"

    if generate_btn or review_cache_key not in st.session_state:
        with st.status("Synthesizing comprehensive academic literature review...", expanded=True) as status:
            st.write("Clustering thematic concepts and identifying shared paradigms...")
            try:
                draft = synthesize_literature_review(
                    collection_id=active_col_id,
                    paper_ids=selected_pids,
                    db=db,
                    ref_style=ref_style.lower(),
                )
                st.session_state[review_cache_key] = draft
                status.update(label="Literature review synthesized successfully!", state="complete", expanded=False)
            except Exception as err:
                status.update(label=f"Synthesis failed: {err}", state="error")
                st.error(f"Error during synthesis: {err}")
                return

    draft = st.session_state.get(review_cache_key)
    if not draft:
        st.info("Click 'Synthesize Literature Review' to generate a structured draft.")
        return

    # Metadata & Actions Header
    meta_c1, meta_c2 = st.columns([3, 1.5])
    with meta_c1:
        st.markdown(
            f"""
            <div style="font-size: 1.25rem; font-weight: 700; color: var(--rl-primary); margin: 0.5rem 0 0.25rem 0;">
                {draft.title}
            </div>
            <div style="font-size: 0.82rem; color: var(--rl-text-muted);">
                Generated for <b>{active_col_name}</b> | Citation Style: <b>{ref_style}</b> | Date: {datetime.now().strftime('%b %d, %Y')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with meta_c2:
        md_text = review_to_markdown(draft)
        st.download_button(
            label="📥 Download Markdown",
            data=md_text.encode("utf-8"),
            file_name=f"literature_review_{active_col_name[:12]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Tabs: Full Review, Thematic, Methods, Gaps, References
    # -----------------------------------------------------------------------
    tab_full, tab_themes, tab_methods, tab_gaps, tab_refs = st.tabs([
        "📑 Full Review",
        "🎯 Thematic Synthesis",
        "⚙️ Methodological Comparison",
        "🚀 Research Gaps",
        "📚 Bibliographic References",
    ])

    # Tab 1: Full Document View
    with tab_full:
        with st.container(border=True):
            st.markdown("### 1. Introduction & Background")
            st.markdown(draft.introduction)

            st.markdown("---")
            st.markdown("### 2. Thematic Synthesis")
            for t_name, t_text in draft.thematic_synthesis.items():
                st.markdown(f"#### 2.{t_name}")
                st.markdown(t_text)

            st.markdown("---")
            st.markdown("### 3. Methodological Comparison")
            st.markdown(draft.methodological_comparison)

            st.markdown("---")
            st.markdown("### 4. Identified Research Gaps & Open Challenges")
            st.markdown(draft.research_gaps)

            st.markdown("---")
            st.markdown("### 5. Conclusion & Synthesis")
            st.markdown(draft.conclusion)

    # Tab 2: Thematic Synthesis
    with tab_themes:
        st.markdown("#### Thematic Synthesis by Conceptual Cluster")
        for t_name, t_text in draft.thematic_synthesis.items():
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-weight: 600; font-size: 1.05rem; color: var(--rl-primary); margin-bottom: 0.5rem;'>📌 {t_name}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(t_text)

    # Tab 3: Methodological Comparison
    with tab_methods:
        with st.container(border=True):
            st.markdown("#### Methodological Paradigms & Trade-Offs")
            st.markdown(draft.methodological_comparison)

    # Tab 4: Research Gaps
    with tab_gaps:
        with st.container(border=True):
            st.markdown("#### Synthesized Research Gaps & Open Opportunities")
            st.markdown(draft.research_gaps)

    # Tab 5: References
    with tab_refs:
        st.markdown(f"#### Formatted Bibliographic References ({ref_style.upper()})")
        if ref_style.lower() == "bibtex":
            for r in draft.references:
                st.code(r, language="bibtex")
        else:
            for idx, r in enumerate(draft.references, 1):
                st.markdown(
                    f"""
                    <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; font-size: 0.9rem;">
                        <b>[{idx}]</b> {r}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
