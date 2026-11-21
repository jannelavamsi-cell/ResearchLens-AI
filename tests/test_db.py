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


def test_schema_version_and_migrations(temp_db):
    """Tests schema_version table tracking and automated migration runner."""
    # 1. Fresh DB should have schema_version 1
    assert temp_db.get_schema_version() >= 1

    # 2. Test legacy migration path
    cid = temp_db.create_collection("Migration Test")
    pid = temp_db.insert_paper(
        collection_id=cid,
        file_hash="hash_mig_1",
        filename="mig_paper.pdf",
        file_path="/tmp/mig.pdf",
        title="Migration Paper",
        n_pages=5,
    )

    # Insert legacy formatted table record (with 'header' and missing 'row_count')
    import json
    legacy_json = json.dumps({
        "page_number": 2,
        "table_index": 1,
        "header": ["ColA", "ColB"],
        "rows": [["1", "2"], ["3", "4"]],
    })
    temp_db.insert_results_table(
        paper_id=pid,
        page_number=2,
        table_index=1,
        table_json=legacy_json,
    )

    # Reset schema version to 0 to simulate pre-migration database
    with temp_db.get_connection() as conn:
        conn.execute("DELETE FROM schema_version")

    # Re-run migration
    with temp_db.get_connection() as conn:
        temp_db._run_migrations(conn)

    # Verify migration applied
    assert temp_db.get_schema_version() == 1
    tables = temp_db.get_results_tables(pid)
    assert len(tables) == 1
    migrated_data = json.loads(tables[0]["table_json"])
    assert migrated_data["row_count"] == 2
    assert migrated_data["col_count"] == 2
    assert migrated_data["headers"] == ["ColA", "ColB"]
    assert "header" not in migrated_data
    assert "page" not in migrated_data


def test_get_collection_stats(temp_db):
    """Tests aggregate collection metrics calculation."""
    cid = temp_db.create_collection("Stats Collection")
    empty_stats = temp_db.get_collection_stats(cid)
    assert empty_stats["papers_count"] == 0
    assert empty_stats["pages_count"] == 0
    assert empty_stats["chunks_count"] == 0
    assert empty_stats["keywords_count"] == 0
    assert empty_stats["extractions_count"] == 0
    assert empty_stats["verified_extractions"] == 0
    assert empty_stats["unverified_extractions"] == 0
    assert empty_stats["tables_count"] == 0
    assert empty_stats["chat_questions"] == 0

    # Add a paper and verify stats update
    pid = temp_db.insert_paper(
        collection_id=cid,
        file_hash="hash999",
        filename="stats_paper.pdf",
        file_path="stats_paper.pdf",
        title="Stats Paper",
        n_pages=10,
    )
    temp_db.update_paper_status(pid, "ready")
    temp_db.insert_chunks([
        {"id": "c1", "paper_id": pid, "chunk_index": 0, "page_start": 1, "page_end": 1, "section_name": "Abstract", "text": "text", "token_count": 5}
    ])
    temp_db.insert_keywords(pid, [{"keyword": "stats", "score": 0.9, "method": "keybert"}])
    temp_db.insert_or_update_extraction(pid, ExtractedField(field_name="objective", value="Test stats", evidence_quote="Test", page=1, confidence=1.0, verified=True))
    temp_db.insert_or_update_extraction(pid, ExtractedField(field_name="limitations", value="Limited data", evidence_quote="", page=0, confidence=0.5, verified=False))
    sess_id = temp_db.create_chat_session(cid, "Test Chat")
    temp_db.insert_chat_message(sess_id, "user", "What is the objective?")

    updated_stats = temp_db.get_collection_stats(cid)
    assert updated_stats["papers_count"] == 1
    assert updated_stats["pages_count"] == 10
    assert updated_stats["ready_papers"] == 1
    assert updated_stats["chunks_count"] == 1
    assert updated_stats["keywords_count"] == 1
    assert updated_stats["extractions_count"] == 2
    assert updated_stats["verified_extractions"] == 1
    assert updated_stats["unverified_extractions"] == 1
    assert updated_stats["chat_questions"] == 1
    assert len(updated_stats["recent_papers"]) == 1
    assert len(updated_stats["recent_questions"]) == 1

