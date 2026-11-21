import hashlib
from pathlib import Path

from database.db_manager import DBManager
from modules.chunking import SectionAwareChunker
from modules.metadata_extraction import MetadataExtractor
from modules.pdf_processor import PDFProcessor
from modules.section_detection import detect_sections_from_pages

db = DBManager("data/researchlens.db")
col_id = "default-collection"
fixtures = list(Path("tests/fixtures").glob("*.pdf"))
print(f"Processing {len(fixtures)} sample PDFs...")

for sf in fixtures:
    with open(sf, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    if db.get_paper_by_hash(col_id, h):
        print(f"Already in DB: {sf.name}")
        continue
    pid = db.insert_paper(col_id, h, sf.name, str(sf), sf.stem)
    proc = PDFProcessor(enable_ocr=False)
    pages, tables, scan = proc.process_pdf(str(sf))
    db.insert_pages(pid, pages)
    sections = detect_sections_from_pages([(p.page_number, p.text) for p in pages])
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_paper(pid, pages, sections)
    db.insert_chunks(chunks)
    meta = MetadataExtractor(enable_online_enrichment=False).extract_metadata(str(sf))
    db.update_paper_metadata(
        pid,
        title=meta.title,
        authors=meta.authors,
        year=meta.year,
        venue=meta.venue,
        doi=meta.doi,
        n_pages=len(pages),
        metadata_json={"scan_report": scan},
    )
    db.update_paper_status(pid, "ready")
    print(f"Processed {sf.name}: {len(pages)} pages, {len(chunks)} chunks, title: {meta.title}")

papers = db.list_papers(col_id)
print(f"Total papers in collection: {len(papers)}")
for p in papers:
    chunks = db.get_chunks(p["id"])
    print(f"- {p['title']} ({p['n_pages']} pages, {len(chunks)} chunks, status: {p['status']})")
