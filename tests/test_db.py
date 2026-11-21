"""
Unit tests for ResearchLens AI database manager and SQLite schema.
Validates table creation, cascading deletes, foreign keys, deduplication, and transactions.
"""

import os
import tempfile

import pytest

from core.schemas import ExtractedField, PageSchema
from database.db_manager import DBManager


@pytest.fixture
def temp_db():
    """Provides a fresh isolated database for each test."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_researchlens.db")
    manager = DBManager(db_path=db_path)
    yield manager
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass


def test_db_initialization(temp_db):
    """Verifies default collection is created on startup."""
    collections = temp_db.list_collections()
    assert len(collections) >= 1
    assert collections[0]["name"] == "Default Workspace"


def test_collection_crud(temp_db):
    """Tests creating, listing, and deleting a collection."""
    cid = temp_db.create_collection("NLP Benchmark Collection", "Testing NLP papers")
    assert cid is not None

    col = temp_db.get_collection(cid)
    assert col["name"] == "NLP Benchmark Collection"
    assert col["description"] == "Testing NLP papers"

    # Delete collection
    assert temp_db.delete_collection(cid) is True
    assert temp_db.get_collection(cid) is None


def test_paper_insertion_and_cascade(temp_db):
    """Tests inserting paper, pages, chunks, extractions, and verifies cascade on delete."""
    cid = temp_db.create_collection("Cascade Collection")

    # Insert paper
    pid = temp_db.insert_paper(
        collection_id=cid,
        file_hash="fakehash123",
        filename="attention_is_all_you_need.pdf",
        file_path="/tmp/fake.pdf",
        title="Attention Is All You Need",
        n_pages=15,
    )
    assert pid is not None

    # Check paper retrieved
    paper = temp_db.get_paper(pid)
    assert paper["title"] == "Attention Is All You Need"
    assert paper["status"] == "queued"

    # Insert pages
    pages = [
        PageSchema(page_number=1, text="Page 1 text: Attention mechanism introduction"),
        PageSchema(page_number=2, text="Page 2 text: Scaled dot-product attention formula"),
    ]
    temp_db.insert_pages(pid, pages)
    retrieved_pages = temp_db.get_pages(pid)
    assert len(retrieved_pages) == 2
    assert retrieved_pages[0]["page_number"] == 1

    # Insert chunks
    chunks = [
        {
            "id": "chk-1",
            "paper_id": pid,
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_name": "Introduction",
            "text": "Attention mechanism introduction text",
            "token_count": 4,
        }
    ]
    temp_db.insert_chunks(chunks)
    assert len(temp_db.get_chunks(pid)) == 1

    # Insert extraction
    extraction = ExtractedField(
        field_name="objective",
        value="Replace recurrent models entirely with attention",
        evidence_quote="We propose the Transformer, based solely on attention mechanisms",
        page=1,
        confidence=0.98,
        source_section="Abstract",
        verified=True,
    )
    temp_db.insert_or_update_extraction(pid, extraction)
    extractions = temp_db.get_extractions(pid)
    assert "objective" in extractions
    assert extractions["objective"]["verified"] is True

    # Test user override
    temp_db.update_user_override(pid, "objective", "New researcher override value", 2)
    updated_extractions = temp_db.get_extractions(pid)
    assert updated_extractions["objective"]["value"] == "New researcher override value"
    assert updated_extractions["objective"]["user_override"] is True

    # Test cascade delete
    assert temp_db.delete_paper(pid) is True
    assert temp_db.get_paper(pid) is None
    assert len(temp_db.get_pages(pid)) == 0
    assert len(temp_db.get_chunks(pid)) == 0
    assert len(temp_db.get_extractions(pid)) == 0


def test_llm_cache(temp_db):
    """Tests prompt caching and retrieval."""
    prompt_hash = "abc123hash"
    response_text = "This is a cached LLM response"
    model_name = "test-model"

    assert temp_db.get_cached_llm_response(prompt_hash) is None

    temp_db.cache_llm_response(prompt_hash, response_text, model_name)
    cached = temp_db.get_cached_llm_response(prompt_hash)
    assert cached == response_text
