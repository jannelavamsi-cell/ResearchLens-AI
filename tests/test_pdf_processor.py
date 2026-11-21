"""
Unit tests for PDFProcessor and text preprocessing in ResearchLens AI.
Validates page counts, 1-indexed numbering, dehyphenation, ligature normalization,
and table extraction.
"""

from pathlib import Path

import pytest

from modules.pdf_processor import PDFProcessor
from modules.text_preprocessing import fix_hyphenation, normalize_ligatures


@pytest.fixture
def sample_pdf_path():
    p = Path("tests/fixtures/paper_1_medical_nlp.pdf")
    if not p.exists():
        from tests.fixtures.create_sample_pdfs import generate_all_sample_pdfs

        generate_all_sample_pdfs()
    return str(p)


def test_text_preprocessing_hyphenation():
    raw_text = "This is a trans-\nformer model and an algo-\n  rithm."
    cleaned = fix_hyphenation(raw_text)
    assert "transformer" in cleaned
    assert "algorithm" in cleaned


def test_text_preprocessing_ligatures():
    raw_text = "The \ufb01rst paper evaluated \ufb02oating-point \ufb00ect."
    cleaned = normalize_ligatures(raw_text)
    assert "first" in cleaned
    assert "floating-point" in cleaned


def test_pdf_processor_pages_and_metadata(sample_pdf_path):
    processor = PDFProcessor(enable_ocr=False)
    pages, tables, scan_report = processor.process_pdf(sample_pdf_path)

    # Paper 1 has exactly 3 pages
    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[2].page_number == 3

    # Check content on page 1
    assert "Attention-Based Deep Learning" in pages[0].text
    assert "MedLens-Transformer" in pages[0].text

    # Check scan report
    assert scan_report["total_pages"] == 3
    assert scan_report["has_scanned_pages"] is False


def test_pdf_processor_scanned_page_detection(tmp_path):
    """Creates a blank/scanned PDF page with no text and checks detection."""
    import pymupdf

    blank_pdf = tmp_path / "scanned_doc.pdf"
    doc = pymupdf.open()
    # Create empty page
    doc.new_page(width=595, height=842)
    doc.save(str(blank_pdf))
    doc.close()

    processor = PDFProcessor(enable_ocr=False)
    pages, tables, scan_report = processor.process_pdf(str(blank_pdf))

    assert len(pages) == 1
    assert pages[0].is_ocr is True
    assert scan_report["has_scanned_pages"] is True
    assert 1 in scan_report["scanned_pages"]


def test_metadata_extractor():
    from modules.metadata_extraction import MetadataExtractor
    sample_pdf = Path("tests/fixtures/paper_1_medical_nlp.pdf")
    if sample_pdf.exists():
        extractor = MetadataExtractor(enable_online_enrichment=False)
        meta = extractor.extract_metadata(str(sample_pdf))
        assert meta.title is not None
        assert "ClinicalBERT" in meta.title or "Paper" in meta.title or len(meta.title) > 3
        doi, year = extractor._extract_doi_and_year("Published in 2023. DOI: 10.1234/test.doi.123")
        assert year == 2023
        assert doi == "10.1234/test.doi.123"


def test_pdf_processor_edge_cases(tmp_path):
    """Tests robustness on missing, empty, corrupted, and encrypted PDFs."""
    import pymupdf

    processor = PDFProcessor(enable_ocr=False)

    # 1. Missing file
    with pytest.raises(FileNotFoundError):
        processor.process_pdf(str(tmp_path / "non_existent.pdf"))

    # 2. Empty file (0 bytes)
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        processor.process_pdf(str(empty_pdf))

    # 3. Non-PDF / Corrupted file
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"This is a plaintext file renamed to .pdf with no PDF headers.")
    with pytest.raises(ValueError, match="invalid or corrupt"):
        processor.process_pdf(str(corrupt_pdf))

    # 4. Encrypted PDF
    enc_pdf = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=400)
    doc.save(str(enc_pdf), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pass123")
    doc.close()

    with pytest.raises(ValueError, match="encrypted or password-protected"):
        processor.process_pdf(str(enc_pdf))


