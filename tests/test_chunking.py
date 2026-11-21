"""
Unit tests for SectionAwareChunker in ResearchLens AI.
Validates section boundary constraints, page range tracking, deterministic IDs,
and References section exclusion.
"""

import pytest

from core.schemas import PageSchema, SectionSpan
from modules.chunking import SectionAwareChunker


@pytest.fixture
def sample_pages_and_sections():
    pages = [
        PageSchema(
            page_number=1,
            text="Abstract\n\nThis paper investigates neural attention architectures.\n\n1. Introduction\n\nDeep learning has revolutionized natural language processing across multiple benchmarks.",
        ),
        PageSchema(
            page_number=2,
            text="2. Methodology\n\nOur architecture consists of 12 layers with relative multi-head self-attention mechanisms.\n\n3. Limitations\n\nOur evaluation is restricted to English text.",
        ),
        PageSchema(
            page_number=3,
            text="References\n\n[1] Vaswani et al. Attention Is All You Need.\n[2] Devlin et al. BERT.",
        ),
    ]
    sections = [
        SectionSpan(section_name="Abstract", page_start=1, page_end=1),
        SectionSpan(section_name="Introduction", page_start=1, page_end=1),
        SectionSpan(section_name="Methodology", page_start=2, page_end=2),
        SectionSpan(section_name="Limitations", page_start=2, page_end=2),
        SectionSpan(section_name="References", page_start=3, page_end=3),
    ]
    return pages, sections


def test_chunking_respects_section_boundaries(sample_pages_and_sections):
    pages, sections = sample_pages_and_sections
    chunker = SectionAwareChunker(
        chunk_size_tokens=100, chunk_overlap_tokens=10, exclude_references=True
    )

    chunks = chunker.chunk_paper("paper-test-123456", pages, sections)

    assert len(chunks) >= 4
    # Ensure every chunk belongs to exactly one section
    section_names = {c["section_name"] for c in chunks}
    assert "Abstract" in section_names
    assert "Introduction" in section_names
    assert "Methodology" in section_names
    assert "Limitations" in section_names

    # Ensure References section is excluded from retrieval chunks
    assert "References" not in section_names


def test_chunking_deterministic_ids_and_pages(sample_pages_and_sections):
    pages, sections = sample_pages_and_sections
    chunker = SectionAwareChunker(chunk_size_tokens=100)

    chunks = chunker.chunk_paper("paper-abcdef99", pages, sections)

    for c in chunks:
        assert c["id"].startswith("chk-paper-ab-")
        assert c["page_start"] in [1, 2, 3]
        assert c["page_end"] >= c["page_start"]
        assert len(c["text"]) > 0
