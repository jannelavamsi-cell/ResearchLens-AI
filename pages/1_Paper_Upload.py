"""
ResearchLens AI - Paper Upload & Library Management Page.
Supports multi-file PDF upload, SHA-256 deduplication, two-column parsing,
section detection, chunking, metadata extraction, and collection management.
"""

import hashlib
from pathlib import Path
from typing import Tuple

import streamlit as st

from core.config import settings
from core.logging_config import get_logger
from database.db_manager import DBManager
from modules.chunking import SectionAwareChunker
from modules.metadata_extraction import MetadataExtractor
from modules.pdf_processor import PDFProcessor
from modules.results_tables import normalize_table_dict
from modules.section_detection import detect_sections_from_pages
from ui.theme import inject_custom_css

logger = get_logger("page.upload")
db = DBManager(settings.database_path)

st.set_page_config(page_title="Paper Upload & Library", page_icon="📄", layout="wide")
inject_custom_css()

# Ensure active collection
if "active_collection_id" not in st.session_state:
    collections = db.list_collections()
    if collections:
        st.session_state["active_collection_id"] = collections[0]["id"]
        st.session_state["active_collection_name"] = collections[0]["name"]
    else:
        cid = db.create_collection("Default Workspace")
        st.session_state["active_collection_id"] = cid
        st.session_state["active_collection_name"] = "Default Workspace"

current_collection_id = st.session_state["active_collection_id"]
current_collection_name = st.session_state["active_collection_name"]

st.title("📄 Paper Upload & Library Management")
st.markdown(
    f"Active Workspace: **{current_collection_name}** | Target Processing: Section-Aware, Two-Column Grounded"
)

# ---------------------------------------------------------------------------
# Section 1: Upload Widget
# ---------------------------------------------------------------------------

upload_container = st.container(border=True)
with upload_container:
    st.subheader("Upload Research Papers")
    st.write(
        "Upload one or more academic research papers in PDF format. Files are validated, deduplicated, "
        "and processed into verifiable section-bounded passages."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum file size: 50MB per paper. Only standard and academic PDFs supported.",
    )

    col_btn, col_sample = st.columns([1, 1])
    process_clicked = col_btn.button(
        "🚀 Process Uploaded Papers", type="primary", disabled=not uploaded_files
    )
    load_samples = col_sample.button("📚 Load 3 Pre-Generated Academic Samples")

# ---------------------------------------------------------------------------
# Helper: Processing Routine
# ---------------------------------------------------------------------------


def process_single_pdf(file_bytes: bytes, filename: str, collection_id: str) -> Tuple[bool, str]:
    """Processes a single PDF file through the document ingestion pipeline."""
    # Compute SHA-256
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check deduplication
    existing = db.get_paper_by_hash(collection_id, file_hash)
    if existing:
        return False, f"Paper '{filename}' already exists in this collection (Duplicate SHA-256)."

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
        # Step 1: Process PDF text and tables
        processor = PDFProcessor(enable_ocr=True)
        pages, tables, scan_report = processor.process_pdf(str(dest_path))

        if not pages:
            db.update_paper_status(paper_id, "failed", "No extractable text found in PDF.")
            return False, f"Failed to extract text from '{filename}'."

        # Insert pages into DB
        db.insert_pages(paper_id, pages)

        # Insert extracted results tables if any
        import json

        for tab in tables:
            norm_tab = normalize_table_dict(tab)
            db.insert_results_table(
                paper_id=paper_id,
                page_number=norm_tab.get("page_number", 1),
                table_index=norm_tab.get("table_index", 1),
                table_json=json.dumps(norm_tab),
            )

        # Step 2: Detect academic sections
        page_tuples = [(p.page_number, p.text) for p in pages]
        sections = detect_sections_from_pages(page_tuples)

        # Step 3: Section-aware chunking
        chunker = SectionAwareChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            exclude_references=True,
        )
        chunks = chunker.chunk_paper(paper_id, pages, sections)
        db.insert_chunks(chunks)

        # Step 4: Metadata extraction
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
        return (
            True,
            f"Successfully processed '{filename}' ({len(pages)} pages, {len(chunks)} chunks).",
        )

    except Exception as e:
        logger.error(f"Error processing '{filename}': {e}", exc_info=True)
        db.update_paper_status(paper_id, "failed", str(e))
        return False, f"Error processing '{filename}': {str(e)}"


# Handle pre-generated sample button
if load_samples:
    fixtures_dir = Path("tests/fixtures")
    sample_files = list(fixtures_dir.glob("*.pdf"))
    if not sample_files:
        from tests.fixtures.create_sample_pdfs import generate_all_sample_pdfs

        generate_all_sample_pdfs()
        sample_files = list(fixtures_dir.glob("*.pdf"))

    with st.status("Loading and indexing sample research papers...", expanded=True) as status:
        for sf in sample_files:
            st.write(f"Reading {sf.name}...")
            with open(sf, "rb") as f:
                data = f.read()
            ok, msg = process_single_pdf(data, sf.name, current_collection_id)
            if ok:
                st.write(f"✅ {msg}")
            else:
                st.write(f"ℹ️ {msg}")
        status.update(label="Sample papers ready in active workspace!", state="complete")
    st.rerun()

# Handle upload processing
if process_clicked and uploaded_files:
    progress_bar = st.progress(0.0)
    total_files = len(uploaded_files)

    with st.status("Processing uploaded papers...", expanded=True) as status:
        for idx, uploaded_file in enumerate(uploaded_files):
            # Size limit check (50 MB)
            if uploaded_file.size > (50 * 1024 * 1024):
                st.error(f"File '{uploaded_file.name}' exceeds 50MB limit.")
                continue

            st.write(f"Processing `{uploaded_file.name}` ({idx + 1}/{total_files})...")
            file_bytes = uploaded_file.read()
            success, message = process_single_pdf(
                file_bytes, uploaded_file.name, current_collection_id
            )

            if success:
                st.write(f"✅ {message}")
            else:
                st.write(f"⚠️ {message}")

            progress_bar.progress((idx + 1) / total_files)

        status.update(label="All uploads processed!", state="complete")
    st.rerun()

# ---------------------------------------------------------------------------
# Section 2: Library Table & Action Controls
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📚 Active Collection Papers")

papers = db.list_papers(current_collection_id)

if not papers:
    st.info(
        "No papers uploaded in this workspace yet. Upload PDFs above or click 'Load 3 Pre-Generated Academic Samples'."
    )
else:
    for p in papers:
        with st.container(border=True):
            col_info, col_stat, col_act = st.columns([3, 1, 1])

            with col_info:
                st.markdown(f"#### {p.get('title') or p.get('filename', 'Unknown Paper')}")
                st.caption(
                    f"**Authors:** {p.get('authors') or 'Unknown'} | "
                    f"**Year:** {p.get('year') or 'N/A'} | "
                    f"**Pages:** {p.get('n_pages', 0)} | "
                    f"**DOI:** {p.get('doi') or 'None'}"
                )
                if p.get("metadata_json"):
                    import json

                    try:
                        meta = json.loads(p.get("metadata_json") or "{}")
                    except Exception:
                        meta = {}
                    scan = meta.get("scan_report", {})
                    if scan.get("has_scanned_pages"):
                        st.warning(
                            f"⚠️ Contains {scan.get('scanned_pages_count', 0)} scanned pages: "
                            f"{scan.get('scanned_pages', [])}. OCR was {'applied' if scan.get('ocr_applied') else 'unavailable'}."
                        )

            with col_stat:
                st.write("")
                st.write("")
                status = p.get("status", "unknown")
                if status == "ready":
                    st.success("🟢 Ready")
                elif status == "processing":
                    st.info("⏳ Processing")
                elif status == "queued":
                    st.warning("⏱️ Queued")
                else:
                    st.error("❌ Failed")
                    if p.get("status_message"):
                        st.caption(p.get("status_message", ""))

            with col_act:
                st.write("")
                st.write("")
                pid = p.get("id", "")
                pfilename = p.get("filename", "paper")
                if pid and st.button("🗑️ Delete", key=f"del_{pid}", use_container_width=True):
                    # Delete file from disk
                    if p.get("file_path"):
                        fpath = Path(p["file_path"])
                        if fpath.exists():
                            try:
                                fpath.unlink()
                            except Exception:
                                pass
                    db.delete_paper(pid)
                    st.success(f"Removed '{pfilename}'")
                    st.rerun()
