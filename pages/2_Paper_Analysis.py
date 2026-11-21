"""
Streamlit Page 2: Paper Analysis & Grounded Extraction.
Displays the 9 core structured research components with verifiable physical page citations,
evidence quote grounding badges (Verified / Unverified / Not Stated),
researcher manual override editing, and extracted experimental results tables.
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

logger = logging.getLogger("researchlens.ui.analysis")

st.set_page_config(
    page_title="Paper Analysis | ResearchLens AI",
    page_icon="🔬",
    layout="wide",
)

db = DatabaseManager()

# Sidebar: Collection and Paper Selection
st.sidebar.title("🔬 Paper Analysis")

collections = db.list_collections()
col_options = {c.get("name", "Default"): c.get("id", "") for c in collections}

if not collections:
    st.warning("No collections found. Please initialize a collection on the Home page.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Workspace Collection", list(col_options.keys()))
active_col_id = col_options[selected_col_name]

papers = db.list_papers(active_col_id)
if not papers:
    st.info("No papers uploaded in this collection yet.")
    st.markdown("👉 Head over to **Paper Upload** in the sidebar to upload research papers or load sample PDFs.")
    st.stop()

paper_options = {
    f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('n_pages', 0)}p)": p.get("id")
    for p in papers
}
selected_paper_label = st.sidebar.selectbox("Select Research Paper", list(paper_options.keys()))
active_paper_id = paper_options[selected_paper_label]

current_paper = db.get_paper(active_paper_id)
if not current_paper:
    st.error("Paper record not found.")
    st.stop()

# Header Information
st.title("🔬 Grounded Paper Analysis")
st.caption("Verifiable, quote-backed research component extraction with physical page citations.")

# Metadata Card
meta_col1, meta_col2, meta_col3 = st.columns([3, 1, 1])

paper_title = current_paper.get("title") or current_paper.get("filename", "Untitled Paper")
paper_filename = current_paper.get("filename", "unknown.pdf")
paper_pages = current_paper.get("n_pages", 0)
paper_status = current_paper.get("status", "ready")

with meta_col1:
    st.subheader(paper_title)
    authors_str = current_paper.get("authors") or "Authors not detected"
    venue_str = current_paper.get("venue") or "Venue not specified"
    year_str = str(current_paper.get("year")) if current_paper.get("year") else "Year unknown"
    doi_str = current_paper.get("doi") or "No DOI"
    st.markdown(f"**Authors:** {authors_str} | **Year:** {year_str} | **Venue:** {venue_str}")
    st.caption(f"**DOI:** `{doi_str}` | **Pages:** {paper_pages} | **File:** `{paper_filename}`")

with meta_col2:
    st.metric("Total Pages", paper_pages)

with meta_col3:
    status_color = "🟢" if paper_status == "ready" else "🟡"
    st.metric("Status", f"{status_color} {paper_status.capitalize()}")

st.markdown("---")

# Extraction Trigger & Toolbar
toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([2, 1, 1])

with toolbar_col1:
    st.write("**Structured Research Extraction**")
    st.caption("Extracts Objective, Abstract, Methodology, Datasets, Algorithms, Metrics, Results, Limitations, and Future Work.")

with toolbar_col2:
    refresh_btn = st.button("🔄 Re-run Extraction", use_container_width=True)

with toolbar_col3:
    extract_tables_btn = st.button("📊 Extract Results Tables", use_container_width=True)

# Handle Results Tables Extraction Trigger
if extract_tables_btn:
    file_path = current_paper.get("file_path")
    if file_path:
        with st.status("Analyzing document layout and extracting tables...", expanded=True) as status:
            tables = extract_tables_from_pdf(file_path, active_paper_id, db=db)
            status.update(label=f"Extracted {len(tables)} tables!", state="complete")
        st.rerun()
    else:
        st.error("Paper file path not available for table extraction.")

# Run or Fetch Extractions
with st.spinner("Loading research extractions..."):
    extractions = extract_paper_components(
        paper_id=active_paper_id,
        db=db,
        force_refresh=refresh_btn,
    )

# Tabs: Components, Results Tables, Raw Pages
tab_components, tab_tables, tab_pages = st.tabs([
    "📋 9 Core Components",
    "📊 Experimental Tables",
    "📄 Page & Chunk Inspector",
])

with tab_components:
    st.markdown("### Grounded Component Matrix")
    st.caption("Every component is mapped to an exact physical page number and verifiable verbatim evidence quote.")

    for field_name in CORE_FIELDS:
        field = extractions.get(field_name)
        if not field:
            continue

        desc = FIELD_DESCRIPTIONS.get(field_name, "")

        with st.container(border=True):
            head_col1, head_col2 = st.columns([3, 1])
            with head_col1:
                st.markdown(f"#### {field_name.replace('_', ' ').title()}")
                st.caption(desc)

            with head_col2:
                # Badges
                badges = []
                if field.user_override:
                    badges.append("✏️ **Researcher Edit**")

                if field.value == "Not stated in paper":
                    badges.append("⚪ **Not Stated**")
                elif field.verified:
                    badges.append("🟢 **Verified Quote**")
                else:
                    badges.append("🟡 **Unverified Quote**")

                for b in badges:
                    st.markdown(b)

                if field.page > 0:
                    st.caption(f"📍 **Cited Page {field.page}** (Conf: {int(field.confidence * 100)}%)")

            # Main Value
            if field.value == "Not stated in paper":
                st.warning("⚠️ *This component was not explicitly identified or stated in the paper text.*")
            else:
                st.markdown(f"> {field.value}")

            # Evidence Quote Drawer
            if field.evidence_quote and field.value != "Not stated in paper":
                with st.expander(f"🔍 View Verbatim Evidence Quote (Page {field.page})"):
                    st.markdown(f"**Section:** `{field.source_section or 'General'}`")
                    st.info(f'"{field.evidence_quote}"')
                    if field.verified:
                        st.caption("✅ **Quote Verified:** This quote was verified verbatim in the physical page text.")
                    else:
                        st.caption("⚠️ **Verification Warning:** High-confidence exact quote match was not confirmed on cited page text.")

            # Researcher Override Form
            with st.expander("✏️ Override or Correct Field"):
                with st.form(key=f"edit_form_{field_name}"):
                    new_val = st.text_area("Corrected Value", value=field.value, height=100)
                    new_page = st.number_input("Physical Page Number", min_value=1, max_value=max(1, paper_pages), value=max(1, field.page))
                    save_btn = st.form_submit_button("Save Researcher Override")
                    if save_btn:
                        db.update_user_override(active_paper_id, field_name, new_val.strip(), int(new_page))
                        st.success(f"Updated {field_name.title()}! Marked as researcher override.")
                        st.rerun()

with tab_tables:
    st.markdown("### Extracted Experimental Tables")
    stored_tables = get_paper_results_tables(active_paper_id, db)

    if not stored_tables:
        st.info("No tables currently extracted for this paper. Click 'Extract Results Tables' above to detect tables.")
    else:
        st.write(f"Detected **{len(stored_tables)}** tables from PDF layout.")
        for t in stored_tables:
            with st.container(border=True):
                try:
                    t_page = t.get("page_number", t.get("page", "?"))
                    t_idx = t.get("table_index", 1)
                    is_res_badge = "🏆 Results Table" if t.get("is_results_table") else "📋 Layout Table"
                    caption_val = t.get("caption", "")
                    caption_str = f" - {caption_val}" if caption_val else ""

                    rows = t.get("rows", [])
                    headers = t.get("headers", t.get("header", []))

                    row_count = t.get("row_count")
                    if row_count is None and isinstance(rows, list):
                        row_count = len(rows)

                    col_count = t.get("col_count")
                    if col_count is None:
                        if isinstance(headers, list) and headers:
                            col_count = len(headers)
                        elif isinstance(rows, list) and rows and isinstance(rows[0], list):
                            col_count = len(rows[0])

                    t_col1, t_col2 = st.columns([3, 1])
                    with t_col1:
                        st.markdown(f"##### {is_res_badge} (Page {t_page}, Table {t_idx}){caption_str}")
                    with t_col2:
                        if row_count is not None and col_count is not None:
                            st.caption(f"Shape: {row_count} rows × {col_count} cols")
                        else:
                            st.caption("Shape unavailable")

                    if headers and rows and isinstance(rows, list):
                        df = pd.DataFrame(rows, columns=headers)
                        st.dataframe(df, use_container_width=True)
                    elif rows and isinstance(rows, list):
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.caption("No tabular rows available for this table.")
                except Exception as table_err:
                    st.warning(f"Could not render table: {table_err}")

with tab_pages:
    st.markdown("### Raw Document Chunks & Sections")
    chunks = db.get_chunks(active_paper_id)
    if not chunks:
        st.info("No chunk records found for this paper.")
    else:
        st.write(f"Showing **{len(chunks)}** indexed chunks across **{paper_pages}** pages:")
        for idx, c in enumerate(chunks):
            p_start = c.get("page_start", "?")
            p_end = c.get("page_end", "?")
            sec_name = c.get("section_name", "General")
            tok_count = c.get("token_count", 0)
            chunk_txt = c.get("text", "")
            with st.expander(f"Chunk #{idx+1} | Page {p_start}–{p_end} | Section: {sec_name} ({tok_count} words)"):
                st.code(chunk_txt, language=None)
