"""
Unit tests for Section Detection in ResearchLens AI.
Validates pattern matching for canonical academic headings and continuous span generation.
"""

from modules.section_detection import classify_heading_line, detect_sections_from_pages


def test_classify_heading_line():
    assert classify_heading_line("Abstract") == "Abstract"
    assert classify_heading_line("1. Introduction") == "Introduction"
    assert classify_heading_line("2. Related Work") == "Related Work"
    assert classify_heading_line("3. Methodology and Architecture") == "Methodology"
    assert classify_heading_line("4. Experimental Results") == "Results"
    assert classify_heading_line("5. Limitations") == "Limitations"
    assert classify_heading_line("6. Future Work") == "Future Work"
    assert classify_heading_line("7. Conclusion") == "Conclusion"
    assert classify_heading_line("References") == "References"

    # Roman numerals and multi-level numbering
    assert classify_heading_line("III. METHODOLOGY") == "Methodology"
    assert classify_heading_line("IV. EMPIRICAL EVALUATION") == "Experiments"
    assert classify_heading_line("3.1 Proposed Framework") == "Methodology"
    assert classify_heading_line("4.2 Quantitative Results") == "Results"
    assert classify_heading_line("Section 5: Threats to Validity") == "Limitations"
    assert classify_heading_line("Background and Context") == "Related Work"

    # Non-heading line
    assert classify_heading_line("This sentence mentions attention mechanisms.") is None


def test_detect_sections_from_pages():
    pages = [
        (1, "Title of Paper\n\nAbstract\nThis is the abstract.\n\n1. Introduction\nIntro text."),
        (2, "3. Methodology\nMethod description here.\n\n4. Results\nResults text here."),
        (3, "5. Limitations\nSome limitations.\n\nReferences\n[1] Reference one."),
    ]

    spans = detect_sections_from_pages(pages)
    span_names = [s.section_name for s in spans]

    assert "Abstract" in span_names
    assert "Introduction" in span_names
    assert "Methodology" in span_names
    assert "Results" in span_names
    assert "Limitations" in span_names
    assert "References" in span_names

    # Check page spans
    method_span = next(s for s in spans if s.section_name == "Methodology")
    assert method_span.page_start == 2
