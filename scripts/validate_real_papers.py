"""
ResearchLens AI: Real-Paper Validation Runner.
Ingests real academic PDFs from data/uploaded_papers/ into SQLite and vector/BM25 indexes,
inspects layout parsing, sections, chunks, and metadata.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from database.db_manager import DatabaseManager
from modules.chunking import SectionAwareChunker
from modules.metadata_extraction import MetadataExtractor
from modules.pdf_processor import PDFProcessor
from modules.section_detection import detect_sections_from_pages

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_real_papers")


def ingest_and_audit_papers():
    settings = Settings()
    db = DatabaseManager(settings.database_path)

    # Ensure default collection exists
    collections = db.list_collections()
    if not collections:
        col_id = db.create_collection("Default Workspace", "Workspace for uploaded academic papers")
    else:
        col_id = collections[0]["id"]

    papers_dir = Path("data/uploaded_papers")
    pdf_files = sorted([f for f in papers_dir.glob("*.pdf") if f.is_file()])

    print("\n=======================================================")
    print(f"FOUND {len(pdf_files)} REAL RESEARCH PAPERS IN data/uploaded_papers/")
    print(f"Collection ID: {col_id}")
    print("=======================================================\n")

    summary_records = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name
        file_bytes = pdf_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        print(f"\n[{idx}/{len(pdf_files)}] Processing: {filename} ({len(file_bytes):,} bytes)")

        # Check if already in DB
        existing = db.get_paper_by_hash(col_id, file_hash)
        if existing:
            paper_id = existing["id"]
            print(f"  -> Already registered in DB as paper_id: {paper_id}")
        else:
            paper_id = db.insert_paper(
                collection_id=col_id,
                file_hash=file_hash,
                filename=filename,
                file_path=str(pdf_path),
                title=filename,
            )
            print(f"  -> Inserted into DB as paper_id: {paper_id}")

        # Process PDF layout
        processor = PDFProcessor(enable_ocr=True)
        pages, tables, scan_report = processor.process_pdf(str(pdf_path))
        print(f"  -> Extracted {len(pages)} pages, {len(tables)} tables (scanned: {scan_report.get('scanned_count', 0)})")

        # Save pages to DB
        db.insert_pages(paper_id, pages)

        # Save tables
        for tab in tables:
            db.insert_results_table(
                paper_id=paper_id,
                page_number=tab["page_number"],
                table_index=tab["table_index"],
                table_json=json.dumps(tab),
            )

        # Detect sections
        page_tuples = [(p.page_number, p.text) for p in pages]
        sections = detect_sections_from_pages(page_tuples)

        # Chunk paper
        chunker = SectionAwareChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            exclude_references=True,
        )
        chunks = chunker.chunk_paper(paper_id, pages, sections)
        db.insert_chunks(chunks)

        # Extract metadata
        extractor = MetadataExtractor(enable_online_enrichment=False)
        metadata = extractor.extract_metadata(str(pdf_path))

        db.update_paper_metadata(
            paper_id=paper_id,
            title=metadata.title or filename,
            authors=metadata.authors or "Not detected",
            year=metadata.year,
            venue=metadata.venue or "Unknown",
            doi=metadata.doi,
            n_pages=len(pages),
            metadata_json={"scan_report": scan_report},
        )
        db.update_paper_status(paper_id, "ready", f"Indexed {len(chunks)} chunks.")

        # Check layout quality metrics
        # 1. Hyphenation inspection: count words ending in hyphens
        raw_hyphen_breaks = sum(p.text.count("-\n") for p in pages)
        # 2. Header/footer check: inspect top/bottom lines
        # 3. Two-column check: check if two-column logic triggered
        # 4. Sections coverage
        section_ranges = [(s.section_name, s.page_start, s.page_end) for s in sections]

        record = {
            "index": idx,
            "filename": filename,
            "paper_id": paper_id,
            "title": metadata.title or filename,
            "authors": metadata.authors or "Not detected",
            "year": metadata.year,
            "venue": metadata.venue or "Unknown",
            "doi": metadata.doi or "None",
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "tables_found": len(tables),
            "sections": section_ranges,
            "raw_hyphen_breaks": raw_hyphen_breaks,
        }
        summary_records.append(record)

    # Save summary report to JSON
    out_json = Path("evaluation/real_papers_audit.json")
    out_json.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    print(f"\n[DONE] Ingestion and layout audit completed for {len(summary_records)} papers.")
    print(f"Audit log saved to: {out_json}")


if __name__ == "__main__":
    ingest_and_audit_papers()
