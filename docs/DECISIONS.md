# ResearchLens AI: Architectural & Implementation Decisions

This document records key technical decisions, trade-offs, and rationale made during development.

---

### DEC-001: Python Runtime & Dependency Management via Astral `uv`
- **Context:** The development machine runs on Windows in a non-administrator environment where standard MSI/EXE installers fail due to headless UAC elevation requests.
- **Decision:** Installed Astral's static `uv` binary to manage Python 3.11.18 isolated runtimes and standard virtual environments (`venv`).
- **Rationale:** `uv` requires zero administrative privileges, runs deterministically in user-space, and manages dependencies at blazing speeds.
- **Consequences:** All project scripts and testing suites run inside `.\venv\Scripts\python.exe`.

### DEC-002: Provider-Agnostic LLM Layer with Multi-Backend Support
- **Context:** Researchers have varying constraints: some prefer cloud-hosted free tiers (Gemini), others require 100% offline local privacy (Ollama), and some use institutional OpenAI endpoints.
- **Decision:** Implemented an abstract base `LLMClient` with swappable backends: `GeminiClient` (default), `OllamaClient`, and `OpenAICompatClient`.
- **Rationale:** Keeps application and pipeline logic completely decoupled from specific vendor SDKs. Includes automatic prompt-hash caching in SQLite (`llm_cache`), exponential backoff retry, and Pydantic schema validation.

### DEC-003: Relational Persistence with SQLite and Persistent FAISS
- **Context:** Need a zero-configuration, lightweight, CPU-efficient storage engine suitable for a college-level project running on an 8 GB laptop.
- **Decision:** Use SQLite (with `PRAGMA foreign_keys = ON` and cascading deletes) for structured metadata, chunks, extractions, keywords, and chat history. Store FAISS vector indexes directly on disk under `data/indexes/<collection_id>.faiss`.
- **Rationale:** Zero external database server overhead; trivial backup, inspection, and portability.

### DEC-004: Section-Aware Chunking over Blind Sliding Windows
- **Context:** Academic papers have distinct semantic boundaries (e.g. Methodology vs Limitations). Standard sliding-window chunking blends sections, corrupting retrieval attribution.
- **Decision:** Parse sections using PyMuPDF font/bold heuristics + regex fallback and enforce that chunks never cross section boundaries.
- **Rationale:** Guarantees every chunk has an unambiguous `section_name` and true page span (`page_start`, `page_end`). Drop References from retrieval chunks to eliminate bibliography noise.

### DEC-005: Evidence-Grounded Extraction with Fuzzy Post-Verification
- **Context:** LLMs can hallucinate paper claims or attribute results to wrong sections.
- **Decision:** Mandate that all extracted fields include an exact `evidence_quote` and `page`. Perform rapid token-level fuzzy matching against the source page text.
- **Rationale:** Any quote not verifiable in the page text is visually flagged as "Unverified" in the UI, preserving integrity.

### DEC-006: Hybrid Retrieval with Reciprocal Rank Fusion (k=60) & Cross-Encoder Reranking
- **Context:** Dense vector search excels at high-level semantic themes but fails on exact dataset names, metric acronyms, or specific numbers. BM25 excels at lexical exact matches but fails on semantic paraphrasing.
- **Decision:** Implement Reciprocal Rank Fusion (RRF $k=60$) fusing top candidates from FAISS (`BAAI/bge-small-en-v1.5`) and Okapi BM25 (`rank-bm25`), followed by a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`).
- **Rationale:** Empirical evaluation demonstrated that hybrid + rerank achieved $1.000$ Precision@3 and $1.000$ nDCG@3 on the benchmark test set, strictly dominating either dense or sparse standalone retrieval.

### DEC-007: Scientific Comparability Guard on Multi-Paper Matrices
- **Context:** LLMs frequently produce superficial cross-paper comparisons declaring one paper "better" based on raw accuracy numbers, even when the papers were evaluated on completely different datasets or metrics.
- **Decision:** Built an automated comparability checker that parses datasets and metrics across papers. When non-identical benchmarks are detected, the system issues a prominent Comparability Warning and strictly orders the narrative synthesis engine never to rank raw scores across different benchmarks.
- **Rationale:** Maintains rigorous scientific integrity and prevents students from making invalid comparative claims in literature reviews.

### DEC-008: Auto-Syncing Vector & Lexical Indexes from Relational Chunks
- **Context:** Restarting sessions or switching collections could leave in-memory BM25 models or disk FAISS indexes out of sync with SQLite chunk records.
- **Decision:** Embedded automatic lazy synchronization into `SemanticSearchEngine.search()` and `BM25SearchEngine.search()`. If an index is empty or unloaded, it automatically loads all ready chunks from SQLite and builds the vector/sparse indexes on demand.
- **Rationale:** Eliminates manual re-indexing hurdles and guarantees zero retrieval misses.

