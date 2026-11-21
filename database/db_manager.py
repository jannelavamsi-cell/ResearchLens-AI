"""
Database Manager for ResearchLens AI.
Handles all relational persistence using SQLite with parameterised queries,
cascading deletes, transaction management, and collection/paper/chunk/extraction CRUD.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from core.logging_config import get_logger
from core.schemas import ExtractedField, PageSchema

logger = get_logger("db_manager")


class DBManager:
    """Manages SQLite connections, schema initialization, and transactional operations."""

    def __init__(self, db_path: str = "data/researchlens.db") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides a managed SQLite connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction error: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def init_db(self, schema_file: Optional[str] = None) -> None:
        """Executes schema.sql to ensure all tables, indexes, and constraints exist."""
        if schema_file is None:
            schema_file = str(Path(__file__).parent / "schema.sql")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            # Create default collection if none exists
            cursor = conn.execute("SELECT COUNT(*) FROM collections")
            if cursor.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO collections (id, name, description) VALUES (?, ?, ?)",
                    (
                        "default-collection",
                        "Default Workspace",
                        "Initial default collection for uploaded research papers.",
                    ),
                )
        logger.info(f"Database initialized successfully at {self.db_path}")

    # -----------------------------------------------------------------------
    # Collections Operations
    # -----------------------------------------------------------------------

    def create_collection(self, name: str, description: Optional[str] = None) -> str:
        """Creates a new named workspace collection."""
        collection_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO collections (id, name, description) VALUES (?, ?, ?)",
                (collection_id, name.strip(), description),
            )
        logger.info(f"Created collection '{name}' (ID: {collection_id})")
        return collection_id

    def list_collections(self) -> List[Dict[str, Any]]:
        """Returns all registered workspace collections."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, name, description, created_at FROM collections ORDER BY created_at ASC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves details of a single collection by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_collection(self, collection_id: str) -> bool:
        """Deletes a collection and cascades removal of all associated papers and data."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Papers Operations
    # -----------------------------------------------------------------------

    def insert_paper(
        self,
        collection_id: str,
        file_hash: str,
        filename: str,
        file_path: str,
        title: Optional[str] = None,
        n_pages: int = 0,
    ) -> str:
        """Registers a newly uploaded research paper in queued status."""
        paper_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO papers (id, collection_id, file_hash, filename, file_path, title, n_pages, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')
                """,
                (
                    paper_id,
                    collection_id,
                    file_hash,
                    filename,
                    file_path,
                    title or filename,
                    n_pages,
                ),
            )
        logger.info(f"Registered paper '{filename}' (ID: {paper_id}) in collection {collection_id}")
        return paper_id

    def get_paper_by_hash(self, collection_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Checks if a paper with identical hash already exists in the given collection."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM papers WHERE collection_id = ? AND file_hash = ?",
                (collection_id, file_hash),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetches complete paper record by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_papers(self, collection_id: str) -> List[Dict[str, Any]]:
        """Returns all papers belonging to a collection."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM papers WHERE collection_id = ? ORDER BY created_at DESC",
                (collection_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_paper_status(
        self, paper_id: str, status: str, status_message: Optional[str] = None
    ) -> None:
        """Updates the processing status of a paper ('queued', 'processing', 'ready', 'failed')."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE papers SET status = ?, status_message = ? WHERE id = ?",
                (status, status_message, paper_id),
            )

    def update_paper_metadata(
        self,
        paper_id: str,
        title: Optional[str] = None,
        authors: Optional[str] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        doi: Optional[str] = None,
        n_pages: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates bibliographic metadata fields on a paper."""
        meta_str = json.dumps(metadata_json) if metadata_json else None
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE papers SET
                    title = COALESCE(?, title),
                    authors = COALESCE(?, authors),
                    year = COALESCE(?, year),
                    venue = COALESCE(?, venue),
                    doi = COALESCE(?, doi),
                    n_pages = COALESCE(?, n_pages),
                    metadata_json = COALESCE(?, metadata_json)
                WHERE id = ?
                """,
                (title, authors, year, venue, doi, n_pages, meta_str, paper_id),
            )

    def delete_paper(self, paper_id: str) -> bool:
        """Deletes a paper and automatically cascades removal of pages, chunks, extractions."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Pages Operations
    # -----------------------------------------------------------------------

    def insert_pages(self, paper_id: str, pages: List[PageSchema]) -> None:
        """Bulk inserts extracted page texts for a paper."""
        records = [
            (str(uuid.uuid4()), paper_id, p.page_number, p.text, 1 if p.is_ocr else 0)
            for p in pages
        ]
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO pages (id, paper_id, page_number, text, is_ocr)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )

    def get_pages(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieves all pages for a paper in page_number order."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM pages WHERE paper_id = ? ORDER BY page_number ASC",
                (paper_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_page_text(self, paper_id: str, page_number: int) -> Optional[str]:
        """Retrieves the extracted text of a specific page."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT text FROM pages WHERE paper_id = ? AND page_number = ?",
                (paper_id, page_number),
            )
            row = cursor.fetchone()
            return row["text"] if row else None

    # -----------------------------------------------------------------------
    # Chunks Operations
    # -----------------------------------------------------------------------

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Bulk inserts section-aware chunks into the database."""
        records = [
            (
                c["id"],
                c["paper_id"],
                c["chunk_index"],
                c["page_start"],
                c["page_end"],
                c["section_name"],
                c["text"],
                c.get("token_count", len(c["text"].split())),
            )
            for c in chunks
        ]
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (id, paper_id, chunk_index, page_start, page_end, section_name, text, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def get_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chunks belonging to a specific paper."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index ASC",
                (paper_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_chunks(self, collection_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chunks for all papers inside a collection (for BM25 / FAISS)."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.*, p.title as paper_title, p.authors as paper_authors, p.year as paper_year
                FROM chunks c
                JOIN papers p ON c.paper_id = p.id
                WHERE p.collection_id = ? AND p.status = 'ready'
                ORDER BY p.id, c.chunk_index ASC
                """,
                (collection_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Extractions Operations
    # -----------------------------------------------------------------------

    def insert_or_update_extraction(self, paper_id: str, extraction: ExtractedField) -> None:
        """Inserts or updates a structured research field extraction."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO extractions
                (id, paper_id, field_name, value, evidence_quote, page_number, confidence, source_section, verified, user_override)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, field_name) DO UPDATE SET
                    value = excluded.value,
                    evidence_quote = excluded.evidence_quote,
                    page_number = excluded.page_number,
                    confidence = excluded.confidence,
                    source_section = excluded.source_section,
                    verified = excluded.verified,
                    user_override = excluded.user_override
                WHERE extractions.user_override = 0 OR excluded.user_override = 1
                """,
                (
                    str(uuid.uuid4()),
                    paper_id,
                    extraction.field_name,
                    extraction.value,
                    extraction.evidence_quote,
                    extraction.page,
                    extraction.confidence,
                    extraction.source_section,
                    1 if extraction.verified else 0,
                    1 if extraction.user_override else 0,
                ),
            )

    def get_extractions(self, paper_id: str) -> Dict[str, Dict[str, Any]]:
        """Returns all structured extractions for a paper mapped by field_name."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM extractions WHERE paper_id = ?",
                (paper_id,),
            )
            results = {}
            for row in cursor.fetchall():
                data = dict(row)
                data["verified"] = bool(data["verified"])
                data["user_override"] = bool(data["user_override"])
                results[data["field_name"]] = data
            return results

    def update_user_override(
        self, paper_id: str, field_name: str, new_value: str, page_number: int
    ) -> None:
        """Saves a manual researcher edit as a verified user override."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE extractions SET
                    value = ?,
                    page_number = ?,
                    verified = 1,
                    user_override = 1
                WHERE paper_id = ? AND field_name = ?
                """,
                (new_value, page_number, paper_id, field_name),
            )

    # -----------------------------------------------------------------------
    # Keywords Operations
    # -----------------------------------------------------------------------

    def insert_keywords(self, paper_id: str, keywords_list: List[Dict[str, Any]]) -> None:
        """Inserts extracted keywords and concept rankings for a paper."""
        records = [
            (
                str(uuid.uuid4()),
                paper_id,
                k["keyword"],
                float(k["score"]),
                k.get("method", "merged"),
            )
            for k in keywords_list
        ]
        with self.get_connection() as conn:
            conn.executemany(
                "INSERT INTO keywords (id, paper_id, keyword, score, method) VALUES (?, ?, ?, ?, ?)",
                records,
            )

    def get_keywords(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieves scored keywords for a given paper."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT keyword, score, method FROM keywords WHERE paper_id = ? ORDER BY score DESC",
                (paper_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_collection_keywords(self, collection_id: str) -> List[Dict[str, Any]]:
        """Returns all keywords across an entire collection for trend analysis."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT k.keyword, k.score, p.title as paper_title, p.year as paper_year
                FROM keywords k
                JOIN papers p ON k.paper_id = p.id
                WHERE p.collection_id = ?
                """,
                (collection_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Results Tables Operations
    # -----------------------------------------------------------------------

    def insert_results_table(
        self, paper_id: str, page_number: int, table_index: int, table_json: str
    ) -> None:
        """Stores an extracted results table JSON."""
        table_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO results_tables (id, paper_id, page_number, table_index, table_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (table_id, paper_id, page_number, table_index, table_json),
            )

    def get_results_tables(self, paper_id: str) -> List[Dict[str, Any]]:
        """Fetches all extracted results tables for a paper."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM results_tables WHERE paper_id = ? ORDER BY page_number, table_index ASC",
                (paper_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Chat History Operations
    # -----------------------------------------------------------------------

    def create_chat_session(self, collection_id: str, title: str = "New Research Chat") -> str:
        """Creates a new conversational chat session in a collection."""
        session_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, collection_id, title) VALUES (?, ?, ?)",
                (session_id, collection_id, title),
            )
        return session_id

    def list_chat_sessions(self, collection_id: str) -> List[Dict[str, Any]]:
        """Lists chat sessions in a collection ordered by recency."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chat_sessions WHERE collection_id = ? ORDER BY created_at DESC",
                (collection_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        verification: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Appends a user or assistant message with citation metadata to a session."""
        message_id = str(uuid.uuid4())
        cit_str = json.dumps(citations) if citations else None
        ver_str = json.dumps(verification) if verification else None
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content, citations_json, verification_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, role, content, cit_str, ver_str),
            )
        return message_id

    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches all messages in chronological order for a session."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            messages = []
            for row in cursor.fetchall():
                data = dict(row)
                if data["citations_json"]:
                    data["citations"] = json.loads(data["citations_json"])
                else:
                    data["citations"] = []
                if data["verification_json"]:
                    data["verification"] = json.loads(data["verification_json"])
                else:
                    data["verification"] = None
                messages.append(data)
            return messages

    # -----------------------------------------------------------------------
    # Reports & Exports Operations
    # -----------------------------------------------------------------------

    def save_report(
        self,
        collection_id: str,
        report_type: str,
        title: str,
        content_markdown: str,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persists a generated literature review, comparison matrix, or gap report."""
        report_id = str(uuid.uuid4())
        meta_str = json.dumps(metadata_json) if metadata_json else None
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reports (id, collection_id, report_type, title, content_markdown, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (report_id, collection_id, report_type, title, content_markdown, meta_str),
            )
        return report_id

    def list_reports(self, collection_id: str) -> List[Dict[str, Any]]:
        """Lists all reports saved for a collection."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM reports WHERE collection_id = ? ORDER BY created_at DESC",
                (collection_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # LLM Cache Operations
    # -----------------------------------------------------------------------

    def get_cached_llm_response(self, prompt_hash: str) -> Optional[str]:
        """Retrieves a cached LLM response by prompt SHA-256 hash."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT response_text FROM llm_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            row = cursor.fetchone()
            return row["response_text"] if row else None

    def cache_llm_response(self, prompt_hash: str, response_text: str, model_name: str) -> None:
        """Caches an LLM generation result to avoid redundant API queries."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (prompt_hash, response_text, model_name)
                VALUES (?, ?, ?)
                """,
                (prompt_hash, response_text, model_name),
            )


# Alias for readability
DatabaseManager = DBManager

