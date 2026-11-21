"""
Streamlit Page 2: Paper Analysis & Grounded Extraction.
Displays the 9 core structured research components with verifiable physical page citations,
evidence quote grounding badges (Verified / Unverified / Not Stated),
researcher manual override editing, extracted experimental results tables, and section chunk inspection.
"""

import logging

import pandas as pd
import streamlit as st

from database.db_manager import DatabaseManager
from modules.research_extraction import (
    CORE_FIELDS,
    FIELD_DESCRIPTIONS,
    extract_paper_components,
)
from modules.results_tables import (
    extract_tables_from_pdf,
    get_paper_results_tables,
)
from ui.components import (
    citation_pill,
    empty_state,
    evidence_panel,
    page_header,
    render_footer,
    safe_page_render,
    status_badge,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.analysis")
db = DatabaseManager()
inject_custom_css()


@st.dialog("Re-run Structured Extraction")
def dialog_rerun_extraction(paper_id: str, paper_title: str) -> None:
    st.write(f"Re-run AI structured component extraction on **{paper_title}**?")
    st.caption("This will re-query the configured LLM to re-extract all 9 components and re-verify quotes.")
    col_c, col_r = st.columns(2)
    with col_c:
        if st.button("Cancel", use_container_width=True, key="cancel_rerun_btn"):
            st.rerun()
    with col_r:
        if st.button("Confirm Re-run", type="primary", use_container_width=True, key="confirm_rerun_btn"):
            with st.spinner("Extracting research components..."):
                extract_paper_components(paper_id=paper_id, db=db, force_refresh=True)
            st.toast("Extraction refreshed successfully!", icon="✨")
            st.rerun()


@safe_page_render
def render_analysis_page() -> None:
    current_col_id = st.session_state.get("active_collection_id", "")
    current_col_name = st.session_state.get("active_collection_name", "Default Workspace")

    page_header(
        title="Grounded Paper Analysis",
        subtitle=f"Verifiable, quote-backed component extraction with physical page citations in workspace: {current_col_name}.",
        icon="🔬",
        badge="Evidence Grounding",
    )

    papers = db.list_papers(current_col_id) if current_col_id else []

    if not papers:
        empty_state(
            title="No Papers in Workspace",
            message="Please upload research papers in PDF Upload before analyzing structured components.",
            icon="📄",
            action_label="🚀 Go to PDF Upload",
            action_page="pages/1_Paper_Upload.py",
        )
        render_footer()
        return

    # Paper Selector Dropdown
    paper_options = {
        f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('n_pages', 0)}p)": p.get("id")
        for p in papers
    }
    selected_label = st.selectbox("Select Research Paper", list(paper_options.keys()), key="analysis_paper_select")
    active_paper_id = paper_options[selected_label]

    current_paper = db.get_paper(active_paper_id)
    if not current_paper:
        st.error("Paper record not found.")
        return

    # Paper Overview Metadata Card
    paper_title = current_paper.get("title") or current_paper.get("filename", "Untitled Paper")
    paper_pages = current_paper.get("n_pages", 0)
    paper_status = current_paper.get("status", "ready")
    authors = current_paper.get("authors") or "Authors not detected"
    venue = current_paper.get("venue") or "Venue not specified"
    year = str(current_paper.get("year")) if current_paper.get("year") else "Year unknown"
    doi = current_paper.get("doi") or "No DOI"

    with st.container(border=True):
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.markdown(f"### {paper_title}")
            st.markdown(f"**Authors:** {authors} • **Year:** {year} • **Venue:** {venue}")
            st.caption(f"**DOI:** `{doi}` • **Pages:** {paper_pages} • **File:** `{current_paper.get('filename', '')}`")
        with col_m2:
            st.markdown(status_badge(paper_status), unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Re-run Extraction", use_container_width=True, key="btn_trigger_rerun"):
                dialog_rerun_extraction(active_paper_id, paper_title)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Tabs: Overview, Extracted Fields, Results Tables, Sections
    tab_overview, tab_components, tab_tables, tab_pages = st.tabs([
        "📖 Abstract & Summary",
        "📋 9 Grounded Components",
        "📊 Experimental Tables",
        "📑 Sections & Passages",
    ])

    extractions = extract_paper_components(paper_id=active_paper_id, db=db, force_refresh=False)

    with tab_overview:
        st.subheader("Verbatim Abstract & Topic Overview")
        abs_field = extractions.get("abstract")
        if abs_field and abs_field.value != "Not stated in paper":
            st.markdown(f"> {abs_field.value}")
            evidence_panel(abs_field.evidence_quote, page=abs_field.page, paper_title=paper_title)
        else:
            st.info("Abstract not explicitly demarcated in paper text.")

        st.markdown("<hr style='margin: 1.25rem 0;'>", unsafe_allow_html=True)
        col_kw_info, col_kw_btn = st.columns([3, 1])
        with col_kw_info:
            st.markdown("##### 🏷️ Scientific Keywords for this Paper")
            kws = db.get_keywords(active_paper_id)
            if kws:
                kw_str = ", ".join([f"`{k.get('keyword')}`" for k in kws[:10]])
                st.markdown(f"Top terms: {kw_str}")
            else:
                st.caption("No keywords extracted yet.")
        with col_kw_btn:
            st.write("")
            if st.button("Explore Keyword Hub", use_container_width=True, key="goto_keywords_hub"):
                st.switch_page("pages/2b_Keyword_Extraction.py")

    with tab_components:
        st.subheader("Grounded Component Matrix")
        st.caption("Every component is mapped to an exact physical page number and verifiable verbatim evidence quote.")

        for field_name in CORE_FIELDS:
            if field_name == "abstract":
                continue  # Displayed in tab_overview

            field = extractions.get(field_name)
            if not field:
                continue

            desc = FIELD_DESCRIPTIONS.get(field_name, "")
            title_formatted = field_name.replace("_", " ").title()

            with st.container(border=True):
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.markdown(f"#### {title_formatted}")
                    st.caption(desc)

                with head_col2:
                    if field.user_override:
                        st.markdown(status_badge("info", "Researcher Override"), unsafe_allow_html=True)
                    elif field.value == "Not stated in paper":
                        st.markdown(status_badge("neutral", "Not Stated"), unsafe_allow_html=True)
                    elif field.verified:
                        st.markdown(status_badge("verified", "Verified Quote"), unsafe_allow_html=True)
                    else:
                        st.markdown(status_badge("warning", "Unverified Quote"), unsafe_allow_html=True)

                    if field.page > 0:
                        st.markdown(citation_pill(field.page), unsafe_allow_html=True)

                # Main extracted value
                if field.value == "Not stated in paper":
                    st.warning("⚠️ *This component was not explicitly identified or stated in the paper text.*")
                else:
                    st.markdown(f"> {field.value}")

                # Evidence quote panel
                if field.evidence_quote and field.value != "Not stated in paper":
                    evidence_panel(field.evidence_quote, page=field.page, paper_title=paper_title)

                # Researcher Override Form
                with st.expander("✏️ Correct / Override Extracted Field"):
                    with st.form(key=f"edit_form_{field_name}"):
                        new_val = st.text_area("Corrected Value", value=field.value, height=90)
                        new_page = st.number_input("Physical Page Number", min_value=1, max_value=max(1, paper_pages), value=max(1, field.page))
                        if st.form_submit_button("Save Researcher Override", type="primary"):
                            db.update_user_override(active_paper_id, field_name, new_val.strip(), int(new_page))
                            st.toast(f"Updated {title_formatted}!", icon="✏️")
                            st.rerun()

    with tab_tables:
        st.subheader("Extracted Experimental Tables")
        stored_tables = get_paper_results_tables(active_paper_id, db)

        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.caption(f"Detected **{len(stored_tables)}** tables from PDF layout.")
        with col_t2:
            if st.button("📊 Extract / Rebuild Tables", use_container_width=True, key="btn_extract_tables"):
                file_path = current_paper.get("file_path")
                if file_path:
                    with st.status("Analyzing document layout...", expanded=True) as status:
                        extract_tables_from_pdf(file_path, active_paper_id, db=db)
                        status.update(label="Tables extracted successfully!", state="complete")
                    st.toast("Tables updated!", icon="📊")
                    st.rerun()

        if not stored_tables:
            st.info("No tables currently extracted for this paper. Click 'Extract / Rebuild Tables' above.")
        else:
            for t in stored_tables:
                with st.container(border=True):
                    try:
                        t_page = t.get("page_number", t.get("page", "?"))
                        t_idx = t.get("table_index", 1)
                        is_res = t.get("is_results_table", False)
                        badge_label = "🏆 Scientific Results Table" if is_res else "📋 Layout Table"
                        caption_val = t.get("caption", "")

                        rows = t.get("rows", [])
                        headers = t.get("headers", t.get("header", []))
                        row_count = t.get("row_count") or (len(rows) if isinstance(rows, list) else 0)
                        col_count = t.get("col_count") or (len(headers) if isinstance(headers, list) else 0)

                        st.markdown(f"##### {badge_label} (Page {t_page}, Table {t_idx})")
                        if caption_val:
                            st.caption(f"**Caption:** {caption_val}")
                        st.caption(f"Shape: {row_count} rows × {col_count} cols")

                        if headers and rows and isinstance(rows, list):
                            st.dataframe(pd.DataFrame(rows, columns=headers), use_container_width=True, hide_index=True)
                        elif rows and isinstance(rows, list):
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception as t_err:
                        st.warning(f"Error rendering table: {t_err}")

    with tab_pages:
        st.subheader("Raw Document Chunks & Sections")
        chunks = db.get_chunks(active_paper_id)
        if not chunks:
            st.info("No chunks indexed for this paper.")
        else:
            st.caption(f"Showing **{len(chunks)}** section-bounded passages across **{paper_pages}** pages:")
            for idx, c in enumerate(chunks):
                p_start = c.get("page_start", "?")
                p_end = c.get("page_end", "?")
                sec_name = c.get("section_name", "General")
                tok_count = c.get("token_count", 0)
                chunk_txt = c.get("text", "")
                with st.expander(f"Chunk #{idx+1} | Page {p_start}–{p_end} | Section: {sec_name} ({tok_count} tokens)"):
                    st.code(chunk_txt, language=None)

    render_footer()


render_analysis_page()
