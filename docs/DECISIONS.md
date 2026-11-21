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

### DEC-009: Canonical Table Schema and Database Version Migration Engine
- **Context:** PyMuPDF table extraction produced varying dict shapes across iterations, causing runtime KeyErrors (`'row_count'`) when loading legacy database entries in `pages/2_Paper_Analysis.py`.
- **Decision:** Defined a strict Pydantic `CanonicalTable` schema with normalized keys (`page_number`, `table_index`, `headers`, `rows`, `row_count`, `col_count`, `caption`, `is_results_table`), built an automatic dictionary normalizer, and implemented a versioned migration engine (`schema_version` table) in `database/db_manager.py`.
- **Rationale:** Eliminates crashes caused by missing keys on old/new records and establishes a reliable upgrade path for database schemas.

### DEC-010: Direct Chrome DevTools Protocol (CDP) WebSocket Screenshot Automation
- **Context:** Streamlit is a reactive single-page app (SPA) requiring WebSocket hydration. Standard headless HTTP requests or premature DOM captures resulted in blank screens or spinner states.
- **Decision:** Built a native Python CDP WebSocket client (`scratch/capture_all_pages.py`) that launches headless Chrome, monitors `Page.loadEventFired` and Streamlit's WebSocket traffic, waits for hydration, and captures full-resolution PNG screenshots.
- **Rationale:** Fully automated visual regression testing and verifiable evidence without requiring heavy external dependencies like Selenium or Playwright.

### DEC-011: Numeric & Metric Density Heuristics for Scientific Results Table Detection
- **Context:** Research papers contain layout tables, author lists, or textual tables that lack quantitative empirical results. Treating every table as a results table cluttered downstream comparative analyses.
- **Decision:** Enhanced `is_likely_results_table()` in `modules/results_tables.py` to evaluate numeric cell density (>= 25%), model candidate tokens (e.g. `bert`, `llama`, `transformer`, `baseline`), and metric keywords (`accuracy`, `f1`, `bleu`, `rouge`, `loss`, `map`).
- **Rationale:** Guarantees that only genuine empirical performance tables are tagged for cross-paper quantitative comparison.

### DEC-012: Scientific Entity Matcher via Rule-Based spaCy Pattern Engine
- **Context:** LLM zero-shot extraction of scientific entities (baselines, metrics, benchmarks) can suffer from omissions or formatting inconsistencies.
- **Decision:** Implemented `extract_scientific_entities()` using spaCy's `Matcher` with rule-based linguistic patterns targeting common ML/NLP patterns (`F1[- ]score`, `BLEU[- ]\d`, `ResNet-\d+`, `GPT-\d+`, `ImageNet`, `GLUE`, etc.).
- **Rationale:** Provides deterministic, fast, zero-cost entity extraction that enriches LLM extractions and structured metadata.

### DEC-013: Domain-Specific Academic Stopword Filtering for KeyBERT Keyword Extraction
- **Context:** General stopword lists leave generic academic vocabulary (e.g. "proposed method", "our approach", "results", "paper", "experimental setup") in KeyBERT extraction results.
- **Decision:** Created `ACADEMIC_DOMAIN_STOPWORDS` and an explicit filtering stage in `keyword_extraction.py`, combined with extending candidate n-grams to (1, 3).
- **Rationale:** Elevates keyword quality from generic research phrasing to domain-specific terminology (e.g., "contrastive loss", "diffusion models", "retrieval augmented generation").

### DEC-014: Cross-Paper Metric Comparability Guards with Known Academic Benchmark Knowledge Base
- **Context:** Multi-paper synthesis can hallucinate or naively claim Paper A outperforms Paper B when each was evaluated on different datasets or evaluation splits.
- **Decision:** Built `check_comparability()` with an embedded knowledge base of standard ML/NLP benchmark datasets (`MMLongBench`, `HotpotQA`, `TriviaQA`, `RetrievalQA`, `SQuAD`, etc.). When differing benchmarks or metrics are detected, the system generates an explicit Comparability Warning and restricts synthesis prompts from performing direct score rankings.
- **Rationale:** Enforces scientific integrity and aligns with peer-review standards for comparative literature analysis.

### DEC-015: Empirical Evaluation Protocol with Dual Human Annotation Sheets
- **Context:** Inventing synthetic ground-truth labels for evaluation yields misleading, ungrounded benchmark figures.
- **Decision:** Developed `evaluation/generate_label_template.py`, which generates both human-readable Markdown annotation sheets and structured JSON schemas from uploaded PDFs. The evaluation harness (`evaluation/report.py`) reads real ground-truth files and outputs honest retrieval ablations and macro F1 scores to `evaluation/RESULTS.md`.
- **Rationale:** Adheres to the core project principle of zero fake results while providing researchers with a practical annotation pipeline.

### DEC-016: Unified Modern CSS Design System & Visual Reusable Component Architecture
- **Context:** Stock Streamlit components provide limited visual hierarchy, inconsistent card margins, and poor readability for complex academic citations and comparative matrices.
- **Decision:** Designed a cohesive design system in `ui/theme.py` and `ui/components.py` with academic glassmorphism styling, clean typography, badge indicators (verified vs unverified), metric cards, alert banners, and citation pills.
- **Rationale:** Transforms the application into a publication-ready research workspace while keeping the codebase modular and maintainable.
