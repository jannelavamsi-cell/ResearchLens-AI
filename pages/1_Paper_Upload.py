"""
ResearchLens AI - Paper Upload & Library Management Page.
Supports multi-file PDF upload, SHA-256 deduplication, two-column parsing,
section detection, chunking, metadata extraction, and collection management with confirmation dialogs.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.chunking import SectionAwareChunker
from modules.metadata_extraction import MetadataExtractor
from modules.pdf_processor import PDFProcessor
from modules.results_tables import normalize_table_dict
from modules.section_detection import detect_sections_from_pages
from ui.components import (
    banner,
    empty_state,
    page_header,
    render_footer,
    safe_page_render,
    status_badge,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.upload")
db = DBManager(settings.database_path)
inject_custom_css()


# ---------------------------------------------------------------------------
# Ingestion Routine with Granular Multi-Stage Feedback
# ---------------------------------------------------------------------------

def process_single_pdf(file_bytes: bytes, filename: str, collection_id: str) -> Tuple[bool, str]:
    """Processes a single PDF file through the document ingestion pipeline."""
    # Stage 1: Validate & Compute SHA-256
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = db.get_paper_by_hash(collection_id, file_hash)
    if existing:
        return False, f"Paper '{filename}' already exists in this workspace (Duplicate SHA-256)."

    # Save to disk
    save_dir = Path(settings.upload_dir) / collection_id
    save_dir.mkdir(parents=True, exist_ok=True)
    dest_path = save_dir / f"{file_hash[:12]}_{filename}"

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    # Register in DB as processing
    paper_id = db.insert_paper(
        collection_id=collection_id,
        file_hash=file_hash,
        filename=filename,
        file_path=str(dest_path),
        title=filename,
    )
    db.update_paper_status(paper_id, "processing", "Extracting text and layout...")

    try:
        # Stage 2: Extract text and layout
        processor = PDFProcessor(enable_ocr=True)
        pages, tables, scan_report = processor.process_pdf(str(dest_path))

        if not pages:
            db.update_paper_status(paper_id, "failed", "No extractable text found in PDF.")
            return False, f"Failed to extract text from '{filename}'."

        db.insert_pages(paper_id, pages)

        # Stage 3: Normalize and insert results tables
        for tab in tables:
            norm_tab = normalize_table_dict(tab)
            db.insert_results_table(
                paper_id=paper_id,
                page_number=norm_tab.get("page_number", 1),
                table_index=norm_tab.get("table_index", 1),
                table_json=json.dumps(norm_tab),
            )

        # Stage 4: Detect academic sections
        page_tuples = [(p.page_number, p.text) for p in pages]
        sections = detect_sections_from_pages(page_tuples)

        # Stage 5: Section-aware chunking
        chunker = SectionAwareChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            exclude_references=True,
        )
        chunks = chunker.chunk_paper(paper_id, pages, sections)
        db.insert_chunks(chunks)

        # Stage 6: Metadata extraction
        extractor = MetadataExtractor(enable_online_enrichment=False)
        metadata = extractor.extract_metadata(str(dest_path))

        db.update_paper_metadata(
            paper_id=paper_id,
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            venue=metadata.venue,
            doi=metadata.doi,
            n_pages=len(pages),
            metadata_json={"scan_report": scan_report},
        )

        db.update_paper_status(paper_id, "ready", f"Successfully indexed {len(chunks)} chunks.")
        logger.info(f"Successfully processed paper '{filename}' (ID: {paper_id})")
        return True, f"Successfully processed '{filename}' ({len(pages)} pages, {len(chunks)} chunks)."

    except Exception as e:
        logger.error(f"Error processing '{filename}': {e}", exc_info=True)
        db.update_paper_status(paper_id, "failed", str(e))
        return False, f"Error processing '{filename}': {str(e)}"


# ---------------------------------------------------------------------------
# Dialogs: Deletion & Replacement
# ---------------------------------------------------------------------------

@st.dialog("Confirm Paper Deletion")
def dialog_delete_paper(paper_id: str, paper_title: str) -> None:
    st.write(f"Are you sure you want to permanently delete **{paper_title}**?")
    st.caption("This will cascade delete all associated passages, extractions, keywords, and tables.")
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Cancel", use_container_width=True, key="cancel_del_btn"):
            st.rerun()
    with col_d:
        if st.button("Delete Permanently", type="primary", use_container_width=True, key="confirm_del_btn"):
            p = db.get_paper(paper_id)
            if p and p.get("file_path"):
                f = Path(p["file_path"])
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            db.delete_paper(paper_id)
            st.toast(f"Deleted '{paper_title}'", icon="🗑️")
            st.rerun()


@st.dialog("Confirm Workspace Deletion")
def dialog_delete_collection(col_id: str, col_name: str) -> None:
    st.write(f"Are you sure you want to delete workspace **{col_name}**?")
    st.error("⚠️ All papers, extractions, chat history, and reports in this workspace will be deleted.")
    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("Cancel", use_container_width=True, key="cancel_col_del"):
            st.rerun()
    with col_d:
        if st.button("Delete Workspace", type="primary", use_container_width=True, key="confirm_col_del"):
            db.delete_collection(col_id)
            all_cols = db.list_collections()
            if all_cols:
                st.session_state["active_collection_id"] = all_cols[0]["id"]
                st.session_state["active_collection_name"] = all_cols[0]["name"]
            st.toast(f"Workspace '{col_name}' deleted.", icon="🗑️")
            st.rerun()


# ---------------------------------------------------------------------------
# Main Page View
# ---------------------------------------------------------------------------

@safe_page_render
def render_upload_page() -> None:
    current_col_id = st.session_state.get("active_collection_id", "")
    current_col_name = st.session_state.get("active_collection_name", "Default Workspace")

    page_header(
        title="PDF Upload & Library Management",
        subtitle=f"Upload academic PDFs to workspace: {current_col_name}. Two-column parsing, section detection, and OCR fallback enabled.",
        icon="📄",
        badge="Ingestion Engine",
    )

    # 1. Upload Container
    with st.container(border=True):
        st.markdown("#### 📥 Add Research Papers")
        st.write("Upload multi-page research PDFs. Files are validated, deduplicated via SHA-256, and parsed into verifiable passages.")

        uploaded_files = st.file_uploader(
            "Select PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Maximum 50MB per paper. Only valid scientific PDFs supported.",
        )

        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 1])
        with btn_col1:
            process_btn = st.button("🚀 Ingest Selected Files", type="primary", disabled=not uploaded_files, use_container_width=True)
        with btn_col2:
            samples_btn = st.button("📚 Load 3 Academic Samples", use_container_width=True)
        with btn_col3:
            if st.button("🗑️ Delete Workspace", use_container_width=True, key="btn_del_col_trigger"):
                dialog_delete_collection(current_col_id, current_col_name)

    # Ingestion handler with multi-stage progress
    if process_btn and uploaded_files:
        p_bar = st.progress(0.0)
        total = len(uploaded_files)

        with st.status(f"Ingesting {total} research paper(s)...", expanded=True) as status:
            for idx, uf in enumerate(uploaded_files):
                st.write(f"**Stage 1:** Validating `{uf.name}` ({idx + 1}/{total})...")
                if uf.size > (50 * 1024 * 1024):
                    st.error(f"'{uf.name}' exceeds the 50MB size limit.")
                    continue

                st.write(f"**Stage 2:** Extracting text & two-column layout from `{uf.name}`...")
                data = uf.read()
                ok, msg = process_single_pdf(data, uf.name, current_col_id)

                if ok:
                    st.write(f"✅ {msg}")
                    st.toast(f"Ingested '{uf.name}'", icon="✅")
                else:
                    st.write(f"⚠️ {msg}")
                    st.toast(msg, icon="⚠️")

                p_bar.progress((idx + 1) / total)

            status.update(label="All selected papers processed!", state="complete")
        st.rerun()

    # Sample loader handler
    if samples_btn:
        fixtures_dir = Path("tests/fixtures")
        sample_files = list(fixtures_dir.glob("*.pdf"))
        if not sample_files:
            from tests.fixtures.create_sample_pdfs import generate_all_sample_pdfs
            generate_all_sample_pdfs()
            sample_files = list(fixtures_dir.glob("*.pdf"))

        with st.status("Ingesting benchmark sample papers...", expanded=True) as status:
            for sf in sample_files:
                st.write(f"Ingesting `{sf.name}`...")
                with open(sf, "rb") as f:
                    data = f.read()
                ok, msg = process_single_pdf(data, sf.name, current_col_id)
                st.write(f"{'✅' if ok else 'ℹ️'} {msg}")
            status.update(label="Sample papers ready in workspace!", state="complete")
        st.toast("3 Academic samples loaded!", icon="📚")
        st.rerun()

    # 2. Paper Library Section
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📚 Workspace Paper Library")

    papers = db.list_papers(current_col_id)

    if not papers:
        empty_state(
            title="Library is Empty",
            message="No research papers have been uploaded to this workspace yet. Drag and drop PDFs above or load sample papers.",
            icon="📭",
        )
    else:
        for p in papers:
            pid = p.get("id", "")
            title = p.get("title") or p.get("filename", "Untitled Paper")
            authors = p.get("authors") or "Authors not detected"
            year = p.get("year") or "Year unknown"
            pages = p.get("n_pages", 0)
            status = p.get("status", "ready")
            doi = p.get("doi") or "No DOI"

            with st.container(border=True):
                col_meta, col_status, col_actions = st.columns([3, 1, 1])

                with col_meta:
                    st.markdown(f"#### {title}")
                    st.caption(f"**Authors:** {authors} | **Year:** {year} | **Pages:** {pages} | **DOI:** `{doi}`")
                    if p.get("metadata_json"):
                        try:
                            meta = json.loads(p.get("metadata_json") or "{}")
                            scan = meta.get("scan_report", {})
                            if scan.get("has_scanned_pages"):
                                banner("warning", "Scanned Pages Detected", f"Contains {scan.get('scanned_pages_count', 0)} scanned pages. OCR was applied.")
                        except Exception:
                            pass

                with col_status:
                    st.write("")
                    st.markdown(status_badge(status), unsafe_allow_html=True)

                with col_actions:
                    st.write("")
                    act_col1, act_col2 = st.columns(2)
                    with act_col1:
                        if st.button("🔍 View", key=f"view_{pid}", use_container_width=True):
                            st.switch_page("pages/2_Paper_Analysis.py")
                    with act_col2:
                        if st.button("🗑️", key=f"del_{pid}", use_container_width=True, help="Delete paper"):
                            dialog_delete_paper(pid, title)

    render_footer()


render_upload_page()
