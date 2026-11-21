# ResearchLens AI: System Architecture

## 1. High-Level System Architecture

ResearchLens AI operates as a modular, five-layered system designed to process multi-page scientific papers, construct verifiable hybrid indices, and support grounded interactive inquiry and literature synthesis.

```mermaid
flowchart TD
    subgraph Layer1["1. Presentation Layer (Streamlit Multipage + Custom Design System)"]
        UI_Home["Landing & Settings (app.py)"]
        UI_Upload["Paper Upload & Library (1_Paper_Upload.py)"]
        UI_Analysis["Paper Analysis & Grounding (2_Paper_Analysis.py)"]
        UI_Search["Smart Hybrid Search (3_Smart_Search.py)"]
        UI_Chat["Research Chat & Citations (4_Research_Chat.py)"]
        UI_Comp["Paper Comparison (5_Paper_Comparison.py)"]
        UI_LitRev["Literature Review Draft (6_Literature_Review.py)"]
        UI_Gaps["Research Gaps (7_Research_Gaps.py)"]
        UI_Trends["Trends & Export Center (8_Trends_and_Export.py)"]
        UI_Theme["Design Tokens & Components (ui/theme.py, ui/components.py)"]
    end

    subgraph Layer2["2. Document Ingestion & Processing Layer"]
        PDF["pdf_processor.py (PyMuPDF Block Sorting + OCR Fallback)"]
        Clean["text_preprocessing.py (Dehyphenation, Header/Footer Stripping)"]
        SecDetect["section_detection.py (Font/Bold Heuristics + Expanded Regex Synonyms)"]
        Chunker["chunking.py (Section-Aware, 300-400 tokens, Page Preservation)"]
        Meta["metadata_extraction.py (DOI / Layout Heuristics + Crossref Enrichment)"]
        Tables["results_tables.py (PyMuPDF find_tables + Canonical Normalization + Semantics)"]
    end

    subgraph Layer3["3. NLP & Hybrid Retrieval Layer"]
        Keywords["keyword_extraction.py (KeyBERT + Academic Domain Stopwords + TF-IDF)"]
        EntityMatcher["research_extraction.py (spaCy Scientific Pattern Matcher)"]
        Embed["embeddings.py (BAAI/bge-small-en-v1.5, L2-Normalized)"]
        FAISS["semantic_search.py (FAISS IndexFlatIP)"]
        BM25["bm25_search.py (rank-bm25 Okapi)"]
        RRF["hybrid_search.py (Reciprocal Rank Fusion, k=60)"]
        Rerank["reranking.py (ms-marco-MiniLM-L-6-v2 Cross-Encoder)"]
    end

    subgraph Layer4["4. AI Intelligence Layer (LLMClient)"]
        LLM["core/llm (Gemini / Ollama / OpenAI-Compatible)"]
        Extract["research_extraction.py (Pydantic Schema + Fuzzy Verification)"]
        RAG["rag_pipeline.py (Query Rewrite + Evidence-Only Generation)"]
        Verify["citation_verifier.py (Claim Splitting + Judge Check)"]
        CompMod["paper_comparison.py (Faithful Matrix + Benchmark Guard)"]
        LitRevMod["literature_review.py (Thematic Clustering + Synthesized Draft)"]
        GapsMod["research_gap.py (Limitation Aggregation & Evidence Mapping)"]
        RefsMod["references.py (IEEE / APA / BibTeX Formatting)"]
    end

    subgraph Layer5["5. Relational & Index Storage Layer"]
        DB[("SQLite database/researchlens.db (Versioned Migrations)")]
        IndexDisk[("Persistent FAISS Indexes data/indexes/")]
        DocStore[("Uploaded PDF Repository data/uploaded_papers/")]
    end

    UI_Theme -.-> UI_Home
    UI_Theme -.-> UI_Upload
    UI_Theme -.-> UI_Analysis
    UI_Theme -.-> UI_Search
    UI_Theme -.-> UI_Chat
    UI_Theme -.-> UI_Comp
    UI_Theme -.-> UI_LitRev
    UI_Theme -.-> UI_Gaps
    UI_Theme -.-> UI_Trends

    UI_Upload --> PDF --> Clean --> SecDetect --> Chunker
    PDF --> Meta
    PDF --> Tables
    Chunker --> Keywords
    Chunker --> EntityMatcher
    Chunker --> Embed --> FAISS
    Chunker --> BM25
    Chunker --> DB
    Meta --> DB
    Keywords --> DB
    EntityMatcher --> DB
    Tables --> DB
    
    UI_Search --> RRF --> Rerank --> UI_Search
    FAISS --> RRF
    BM25 --> RRF
    
    UI_Chat --> RAG --> LLM
    RRF --> Rerank --> RAG
    RAG --> Verify --> UI_Chat
    
    UI_Analysis --> Extract --> LLM
    Extract --> Verify --> DB
    
    UI_Comp --> CompMod --> DB
    UI_LitRev --> LitRevMod --> LLM
    UI_Gaps --> GapsMod --> LLM
    UI_Trends --> DB
    UI_Trends --> RefsMod
```

---

## 2. Ingestion & Retrieval Data Flow

### 2.1 Ingestion Flow
1. **Upload & Deduplication:** File uploaded via `pages/1_Paper_Upload.py`. SHA-256 hash computed. If duplicate exists in the active collection, an alert is returned.
2. **Two-Column Reading Order:** PyMuPDF parses text blocks in geometric reading order. Pages are preserved 1-indexed.
3. **Noise Reduction:** Headers, footers, and page numbers repeating across $>60\%$ of pages are stripped. Hyphenation across line breaks is resolved.
4. **Section Demarcation:** Heading heuristics detect sections (`Abstract`, `Introduction`, `Methodology`, `Results`, `Limitations`, `Conclusion`, `References`) supporting roman numerals and multi-level numbering (`3.1 Proposed Framework`). Chunks are generated strictly within section boundaries.
5. **Table Normalization & Classification:** PyMuPDF table extraction converts all tables to the `CanonicalTable` schema with explicit `row_count` and `col_count`. Scientific results tables are classified based on numeric density and model/metric keyword matching.
6. **Dual Indexing:** Text chunks are indexed into disk-backed BM25 and embedded via `BAAI/bge-small-en-v1.5` into FAISS `IndexFlatIP`.
7. **Structured Extraction & Entity Matcher:** Core research components are extracted via `LLMClient` with Pydantic JSON schemas and complemented by a spaCy rule-based `Matcher` for scientific entities (baselines, datasets, metrics). Verbatim quotes are verified against source page text using fuzzy matching.

### 2.2 Query & Verification Flow
1. **Query Reformulation:** If part of a conversation, follow-up queries are rewritten to be standalone.
2. **Hybrid Fusion:** FAISS dense retrieval and BM25 sparse retrieval fetch top-$K$ candidates each, combined using Reciprocal Rank Fusion ($k=60$):
   $$RRF\_score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$
3. **Cross-Encoder Reranking:** Top 20 fusion candidates are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 5 most relevant passages.
4. **Evidence-Grounded Prompting:** Excerpts are formatted inside passive `<documents>` tags with anti-injection instructions.
5. **Automated Claim Verification:** Generated answer sentences are split and checked against cited chunks. Unsupported claims are visually flagged in the UI.

---

## 3. Database Architecture & Schema Versioning

SQLite database (`database/researchlens.db`) manages relational persistence with foreign-key constraints and cascading deletes:
- `schema_version`: Tracks migration history (`version`, `applied_at`).
- `collections`: Workspaces (`id`, `name`, `description`, `created_at`).
- `papers`: Uploaded documents (`id`, `collection_id`, `filename`, `sha256_hash`, `title`, `authors`, `year`, `page_count`, etc.).
- `chunks`: Section-aware text passages (`id`, `paper_id`, `chunk_index`, `section_name`, `page_start`, `page_end`, `content`, `token_count`).
- `extractions`: Extracted research components (`id`, `paper_id`, `field_name`, `field_value`, `page_number`, `evidence_quote`, `verified`, `user_edited`).
- `results_tables`: Canonical structured tables (`id`, `paper_id`, `page_number`, `table_index`, `headers_json`, `rows_json`, `row_count`, `col_count`, `caption`, `is_results_table`).
- `keywords`: Extracted keywords (`id`, `paper_id`, `keyword`, `score`, `method`).
- `chat_messages`: Multi-turn conversational history (`id`, `collection_id`, `role`, `content`, `citations_json`, `created_at`).
- `llm_cache`: Prompt-hash cache (`prompt_hash`, `provider`, `model`, `response_text`, `created_at`).
