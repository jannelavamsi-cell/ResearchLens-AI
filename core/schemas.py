"""
Data schemas for ResearchLens AI using Pydantic v2.
Defines strongly-typed models for all documents, chunks, extractions,
citations, and comparative analyses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Document & Ingestion Schemas
# ---------------------------------------------------------------------------


class PageSchema(BaseModel):
    """Represents an extracted single page from a PDF document."""

    page_number: int = Field(..., description="1-indexed physical page number")
    text: str = Field(..., description="Cleaned, dehyphenated page text")
    is_ocr: bool = Field(default=False, description="Whether page was processed via OCR")


class SectionSpan(BaseModel):
    """Represents a recognized section span within a paper."""

    section_name: str = Field(..., description="Detected section heading")
    page_start: int = Field(..., description="Page where section begins")
    page_end: int = Field(..., description="Page where section ends")


class ChunkSchema(BaseModel):
    """Represents a section-bounded text chunk for indexing and retrieval."""

    id: str = Field(..., description="Deterministic chunk UUID or hash")
    paper_id: str = Field(..., description="ID of source paper")
    chunk_index: int = Field(..., description="Zero-indexed chunk sequence number")
    page_start: int = Field(..., description="Start page number (1-indexed)")
    page_end: int = Field(..., description="End page number (1-indexed)")
    section_name: str = Field(..., description="Parent section heading")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Approximate token count")


class PaperMetadata(BaseModel):
    """Bibliographic metadata for a research paper."""

    title: Optional[str] = Field(default="Untitled Paper", description="Paper title")
    authors: Optional[str] = Field(default="Unknown Authors", description="Author list")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Conference or journal venue")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    abstract: Optional[str] = Field(default=None, description="Verbatim abstract")


class PaperSchema(BaseModel):
    """Represents a full research paper entity in ResearchLens AI."""

    id: str = Field(..., description="Unique paper identifier (UUID)")
    collection_id: str = Field(..., description="Workspace collection identifier")
    file_hash: str = Field(..., description="SHA-256 hash of PDF file")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Local filesystem path to PDF")
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    n_pages: int = Field(default=0, description="Total page count")
    status: str = Field(default="queued", description="Status: queued/processing/ready/failed")
    status_message: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Structured Research Extraction Schemas
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A grounded extracted research component with page citation and evidence quote."""

    field_name: str = Field(..., description="Name of the extracted component")
    value: str = Field(
        ...,
        description="Extracted synthesis or value. Must be 'Not stated in paper' if absent.",
    )
    evidence_quote: str = Field(
        ..., description="Short verbatim excerpt from the paper supporting the value"
    )
    page: int = Field(..., description="1-indexed page number containing the quote")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence rating between 0 and 1"
    )
    source_section: Optional[str] = Field(
        default=None, description="Section from which evidence was extracted"
    )
    verified: bool = Field(
        default=False, description="Whether evidence quote was verified against page text"
    )
    user_override: bool = Field(
        default=False, description="Whether value was manually edited by the researcher"
    )


class StructuredPaperExtraction(BaseModel):
    """Complete grounded structured extraction schema for an academic paper."""

    objective: ExtractedField
    abstract: ExtractedField
    methodology: ExtractedField
    datasets: ExtractedField
    algorithms: ExtractedField
    metrics: ExtractedField
    results: ExtractedField
    limitations: ExtractedField
    future_work: ExtractedField


# ---------------------------------------------------------------------------
# Retrieval & RAG Schemas
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """Represents a retrieved passage with citation metadata and fusion scores."""

    chunk_id: str
    paper_id: str
    paper_title: str
    section_name: str
    page_start: int
    page_end: int
    text: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    """Represents a verifiable page-level citation in an answer."""

    paper_id: str
    paper_title: str
    page_number: int
    section_name: Optional[str] = None
    snippet: Optional[str] = None


class ClaimVerification(BaseModel):
    """Verification analysis for an individual claim made in an answer."""

    claim_text: str = Field(..., description="Individual sentence or claim from response")
    citation: Optional[str] = Field(default=None, description="Attached inline citation")
    is_supported: bool = Field(
        default=False, description="Whether claim is supported by cited evidence"
    )
    support_level: str = Field(
        default="unsupported", description="supported / partially_supported / unsupported"
    )
    evidence_snippet: Optional[str] = None
    reason: Optional[str] = None


class RAGResponse(BaseModel):
    """Complete verifiable RAG response object."""

    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    verifications: List[ClaimVerification] = Field(default_factory=list)
    overall_support_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Proportion of verified claims"
    )
    insufficient_evidence: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Comparison & Review Schemas
# ---------------------------------------------------------------------------


class ComparabilityWarning(BaseModel):
    """Warning issued when comparing papers evaluated on distinct benchmarks."""

    has_warning: bool = False
    reasons: List[str] = Field(default_factory=list)
    message: str = ""


class ComparisonCell(BaseModel):
    """A cell in the cross-paper comparison table."""

    paper_id: str
    field_name: str
    value: str
    evidence_quote: str
    page: int
    verified: bool = False


class LiteratureReviewDraft(BaseModel):
    """Structured literature review synthesis."""

    title: str
    introduction: str
    thematic_synthesis: Dict[str, str] = Field(
        default_factory=dict, description="Theme name -> synthesized text"
    )
    methodological_comparison: str
    research_gaps: str
    conclusion: str
    references: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "AI-generated draft. Verify against the original papers and add proper "
        "academic referencing before submission."
    )


class ResearchGapItem(BaseModel):
    """A single grounded research gap opportunity."""

    gap_statement: str
    supporting_papers: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of {paper_title, page, limitation_quote}"
    )
    frequency_count: int = Field(default=1)
    suggested_direction: str
    rationale: str


class CanonicalTable(BaseModel):
    """Canonical representation of an extracted table with strictly validated keys."""

    page_number: int = Field(..., description="1-indexed physical page number")
    table_index: int = Field(default=1, description="1-indexed table sequence on the page")
    headers: List[str] = Field(default_factory=list, description="Column header strings")
    rows: List[List[str]] = Field(default_factory=list, description="Table body row cell values")
    row_count: int = Field(default=0, description="Number of data rows")
    col_count: int = Field(default=0, description="Number of columns")
    caption: str = Field(default="", description="Table caption or title")
    is_results_table: bool = Field(default=False, description="Whether table contains empirical results")
    paper_id: Optional[str] = Field(default=None, description="Associated paper ID")
    id: Optional[str] = Field(default=None, description="Database row ID if stored")

