-- =====================================================================
-- ResearchLens AI: SQLite Relational Schema
-- =====================================================================

PRAGMA foreign_keys = ON;

-- Named Collections / Workspaces
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Uploaded Research Papers
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    doi TEXT,
    n_pages INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',  -- 'queued', 'processing', 'ready', 'failed'
    status_message TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    UNIQUE(collection_id, file_hash)
);

-- Per-Page Text Store (Preserves 1-indexed page mapping)
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    is_ocr INTEGER DEFAULT 0,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(paper_id, page_number)
);

-- Section-Aware Document Chunks
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

-- Structured Research Extractions (Objective, Methodology, Results, etc.)
CREATE TABLE IF NOT EXISTS extractions (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    value TEXT NOT NULL,
    evidence_quote TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_section TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    user_override INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(paper_id, field_name)
);

-- Extracted Keywords & Concepts
CREATE TABLE IF NOT EXISTS keywords (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,  -- 'keybert', 'tfidf', 'spacy', 'merged'
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

-- Interactive Chat Sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

-- Chat Messages & Verifiable Citations
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant'
    content TEXT NOT NULL,
    citations_json TEXT,
    verification_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

-- Generated Analysis & Review Reports
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    report_type TEXT NOT NULL,  -- 'paper_analysis', 'comparison', 'literature_review', 'research_gaps'
    title TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

-- Persistent LLM Query Cache
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    response_text TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted Results Tables
CREATE TABLE IF NOT EXISTS results_tables (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    table_json TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);
