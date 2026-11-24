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
    """Tests schema_version table tracking and automated migration runner across v1 and v2."""
    # 1. Fresh DB should have schema_version >= 2
    assert temp_db.get_schema_version() >= 2

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

    # Insert legacy formatted table record (with 'header' and missing 'row_count', 'classification', etc.)
    import json
    legacy_json = json.dumps({
        "page_number": 2,
        "table_index": 1,
        "header": ["Model", "Accuracy"],
        "rows": [["Model A", "89.5"], ["Model B", "91.2"]],
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

    # Re-run migrations sequentially (v0 -> v1 -> v2)
    with temp_db.get_connection() as conn:
        temp_db._run_migrations(conn)

    # Verify migration applied up to version 2
    assert temp_db.get_schema_version() == 2
    tables = temp_db.get_results_tables(pid)
    assert len(tables) == 1
    migrated_data = json.loads(tables[0]["table_json"])

    # Verify v1 migration canonical properties
    assert migrated_data["row_count"] == 2
    assert migrated_data["col_count"] == 2
    assert migrated_data["headers"] == ["Model", "Accuracy"]
    assert "header" not in migrated_data
    assert "page" not in migrated_data

    # Verify v2 migration canonical properties
    assert migrated_data["is_accepted"] is True
    assert migrated_data["classification"] == "results"
    assert migrated_data["is_results_table"] is True
    assert isinstance(migrated_data["normalized_records"], list)
    assert len(migrated_data["normalized_records"]) == 2
    assert migrated_data["normalized_records"][0]["model"] == "Model A"
    assert migrated_data["normalized_records"][0]["value"] == "89.5"


def test_reextract_tables_action(temp_db, tmp_path):
    """Tests the reextract_tables_for_paper and reextract_all_tables actions."""
    import json

    import pymupdf

    cid = temp_db.create_collection("Reextract Workspace")

    # 1. Test fallback when source PDF does not exist on disk
    pid_no_pdf = temp_db.insert_paper(
        collection_id=cid,
        file_hash="hash_no_pdf",
        filename="missing.pdf",
        file_path=str(tmp_path / "non_existent.pdf"),
        title="Missing PDF Paper",
        n_pages=1,
    )
    temp_db.insert_results_table(
        paper_id=pid_no_pdf,
        page_number=1,
        table_index=1,
        table_json=json.dumps({"headers": ["M", "Score"], "rows": [["A", "95"]]}),
    )

    re_res = temp_db.reextract_tables_for_paper(pid_no_pdf)
    assert len(re_res) == 1
    assert re_res[0]["headers"] == ["M", "Score"]
    assert re_res[0]["is_accepted"] is True
    assert re_res[0]["classification"] == "results"

    # 2. Test re-extraction when real source PDF exists on disk
    pdf_path = str(tmp_path / "real_paper.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 100), "Table 1: Benchmark Accuracy on GLUE")
    page.draw_rect(pymupdf.Rect(50, 130, 200, 160))
    page.insert_text((60, 150), "Model")
    page.draw_rect(pymupdf.Rect(200, 130, 350, 160))
    page.insert_text((210, 150), "Accuracy")
    page.draw_rect(pymupdf.Rect(50, 160, 200, 190))
    page.insert_text((60, 180), "Model X")
    page.draw_rect(pymupdf.Rect(200, 160, 350, 190))
    page.insert_text((210, 180), "96.4")
    doc.save(pdf_path)
    doc.close()

    pid_with_pdf = temp_db.insert_paper(
        collection_id=cid,
        file_hash="hash_with_pdf",
        filename="real_paper.pdf",
        file_path=pdf_path,
        title="Real PDF Paper",
        n_pages=1,
    )

    # Insert old junk record that should be cleared on re-extraction
    temp_db.insert_results_table(
        paper_id=pid_with_pdf,
        page_number=1,
        table_index=99,
        table_json=json.dumps({"headers": ["OldJunk"], "rows": [["Old"]]}),
    )

    fresh_tables = temp_db.reextract_tables_for_paper(pid_with_pdf)
    assert len(fresh_tables) >= 1
    assert fresh_tables[0]["is_accepted"] is True
    assert fresh_tables[0]["classification"] == "results"

    # Verify old junk record is gone
    stored = temp_db.get_results_tables(pid_with_pdf)
    assert len(stored) == len(fresh_tables)

    # 3. Test reextract_all_tables
    summary = temp_db.reextract_all_tables(collection_id=cid)
    assert summary["papers_count"] == 2
    assert summary["tables_count"] >= 2


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

