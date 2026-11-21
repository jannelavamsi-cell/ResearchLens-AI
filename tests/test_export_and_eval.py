"""
Tests for Phase 8: Multi-Format Document Export and Evaluation Metrics.
"""

import io
import math
import os

from evaluation.eval_retrieval import compute_dcg, compute_ndcg, is_chunk_relevant
from modules.export import (
    export_bibtex_file,
    export_to_docx,
    export_to_markdown,
    export_to_pdf,
)


def test_markdown_and_bibtex_export():
    papers = [
        {
            "title": "ClinicalBERT",
            "authors": "Emily Alsentzer and John Murphy",
            "year": 2019,
            "venue": "Clinical NLP Workshop",
            "doi": "10.18653/v1/W19-1909",
        }
    ]

    # Markdown export
    md = export_to_markdown(
        title="Literature Analysis",
        content="This report synthesizes medical transformer findings.",
        references=["[1] E. Alsentzer, ClinicalBERT, 2019."],
    )
    assert "# Literature Analysis" in md
    assert "This report synthesizes" in md
    assert "[1] E. Alsentzer" in md

    # BibTeX export
    bib = export_bibtex_file(papers)
    assert "@article{alsentzer2019" in bib
    assert "title     = {ClinicalBERT}" in bib
    assert "doi       = {10.18653/v1/W19-1909}" in bib


def test_docx_export():
    sections = [
        {"title": "1. Introduction", "content": "Introductory overview of deep learning.\n- Point 1\n- Point 2"},
        {"title": "2. Comparative Study", "content": "> Quoted comparative trade-off observation."},
    ]
    table = [
        ["Model", "Accuracy", "Metric"],
        ["ClinicalBERT", "88.4%", "Macro F1"],
    ]

    docx_stream = export_to_docx(
        title="Comparative Report",
        sections=sections,
        table_data=table,
        references=["[1] Reference entry."],
        disclaimer="AI-generated draft for testing.",
    )
    assert isinstance(docx_stream, io.BytesIO)
    assert docx_stream.getbuffer().nbytes > 1000


def test_pdf_export():
    sections = [
        {"title": "1. Introduction", "content": "Overview text for PDF compilation."},
        {"title": "2. Results Summary", "content": "- Bullet item 1\n- Bullet item 2"},
    ]
    pdf_stream = export_to_pdf(
        title="Evaluation Report",
        sections=sections,
        references=["[1] Reference entry."],
        disclaimer="AI-generated draft.",
    )
    assert isinstance(pdf_stream, io.BytesIO)
    pdf_bytes = pdf_stream.getvalue()
    assert pdf_bytes.startswith(b"%PDF-")


def test_retrieval_evaluation_math():
    # Test DCG and nDCG
    relevances = [1, 0, 1]
    dcg = compute_dcg(relevances, k=3)
    expected_dcg = (1.0 / math.log2(2)) + (0.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert math.isclose(dcg, expected_dcg, rel_tol=1e-3)

    ndcg = compute_ndcg(relevances, k=3)
    assert 0.0 < ndcg <= 1.0

    # Perfect ranking nDCG should be 1.0
    perfect_ndcg = compute_ndcg([1, 1, 0], k=3)
    assert math.isclose(perfect_ndcg, 1.0, rel_tol=1e-3)

    # Empty ranking nDCG should be 0.0
    zero_ndcg = compute_ndcg([0, 0, 0], k=3)
    assert zero_ndcg == 0.0

    # Relevance keyword matcher
    text = "ClinicalBERT is evaluated on MIMIC-III diagnostic codes."
    assert is_chunk_relevant(text, ["mimic", "clinical"]) is True
    assert is_chunk_relevant(text, ["quantum", "astronomy"]) is False


def test_evaluation_suite_run():
    import os
    import tempfile

    from evaluation.report import run_evaluation_suite

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        out_path = tmp.name

    try:
        results = run_evaluation_suite(output_path=out_path)
        assert "retrieval" in results
        assert "extraction" in results
        assert "rag" in results
        assert os.path.exists(out_path)
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass


def test_generate_label_template(tmp_path):
    from evaluation.generate_label_template import (
        generate_paper_json_template,
        generate_paper_markdown_sheet,
        generate_templates_for_paper_path,
    )

    out_dir = str(tmp_path)
    md_file = generate_paper_markdown_sheet(
        filename="test_paper.pdf",
        title="Test Paper Title",
        authors="Test Author",
        num_pages=3,
        detected_sections=["Introduction", "Methodology"],
        output_dir=out_dir,
    )
    assert os.path.exists(md_file)
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test Paper Title" in content
    assert "2.1 Objective" in content

    json_file = generate_paper_json_template("test_paper.pdf", output_dir=out_dir)
    assert os.path.exists(json_file)

    pdf_path = "tests/fixtures/paper_1_medical_nlp.pdf"
    if os.path.exists(pdf_path):
        res = generate_templates_for_paper_path(pdf_path, output_dir=out_dir)
        assert os.path.exists(res["markdown"])
        assert os.path.exists(res["json"])

