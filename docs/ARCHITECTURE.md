# ResearchLens AI: System Architecture

## 1. High-Level System Architecture

ResearchLens AI operates as a modular, five-layered system designed to process multi-page scientific papers, construct verifiable hybrid indices, and support grounded interactive inquiry and literature synthesis.

```mermaid
flowchart TD
    subgraph Layer1["1. Presentation Layer (Streamlit Multipage)"]
        UI_Home["Landing & Settings (app.py)"]
        UI_Upload["Paper Upload & Library (1_Paper_Upload.py)"]
        UI_Analysis["Paper Analysis & Grounding (2_Paper_Analysis.py)"]
        UI_Search["Smart Hybrid Search (3_Smart_Search.py)"]
        UI_Chat["Research Chat & Citations (4_Research_Chat.py)"]
        UI_Comp["Paper Comparison (5_Paper_Comparison.py)"]
        UI_LitRev["Literature Review Draft (6_Literature_Review.py)"]
        UI_Gaps["Research Gaps (7_Research_Gaps.py)"]
        UI_Trends["Trends & Export Center (8_Trends_and_Export.py)"]
    end

    subgraph Layer2["2. Document Ingestion & Processing Layer"]
        PDF["pdf_processor.py (PyMuPDF Block Sorting + OCR Fallback)"]
        Clean["text_preprocessing.py (Dehyphenation, Header/Footer Stripping)"]
        SecDetect["section_detection.py (Font/Bold Heuristics + Regex)"]
        Chunker["chunking.py (Section-Aware, 300-400 tokens, Page Preservation)"]
        Meta["metadata_extraction.py (DOI / Layout Heuristics + Crossref Enrichment)"]
        Tables["results_tables.py (PyMuPDF find_tables + Normalization)"]
    end

    subgraph Layer3["3. NLP & Hybrid Retrieval Layer"]
        Keywords["keyword_extraction.py (KeyBERT + TF-IDF + spaCy)"]
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
        DB[("SQLite database/researchlens.db")]
        IndexDisk[("Persistent FAISS Indexes data/indexes/")]
        DocStore[("Uploaded PDF Repository data/uploaded_papers/")]
    end

    UI_Upload --> PDF --> Clean --> SecDetect --> Chunker
    PDF --> Meta
    PDF --> Tables
    Chunker --> Keywords
    Chunker --> Embed --> FAISS
    Chunker --> BM25
    Chunker --> DB
    Meta --> DB
    Keywords --> DB
    
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
4. **Section Demarcation:** Heading heuristics detect sections (`Abstract`, `Introduction`, `Methodology`, etc.). Chunks are generated strictly within section boundaries.
5. **Dual Indexing:** Text chunks are indexed into disk-backed BM25 and embedded via `BAAI/bge-small-en-v1.5` into FAISS `IndexFlatIP`.
6. **Structured Extraction:** Core research components are extracted via `LLMClient` with Pydantic JSON schemas, and verbatim quotes are verified against source page text using fuzzy matching.

### 2.2 Query & Verification Flow
1. **Query Reformulation:** If part of a conversation, follow-up queries are rewritten to be standalone.
2. **Hybrid Fusion:** FAISS dense retrieval and BM25 sparse retrieval fetch top-$K$ candidates each, combined using Reciprocal Rank Fusion ($k=60$):
   $$RRF\_score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$
3. **Cross-Encoder Reranking:** Top 20 fusion candidates are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 5 most relevant passages.
4. **Evidence-Grounded Prompting:** Excerpts are formatted inside passive `<documents>` tags with anti-injection instructions.
5. **Automated Claim Verification:** Generated answer sentences are split and checked against cited chunks. Unsupported claims are visually flagged in the UI.
