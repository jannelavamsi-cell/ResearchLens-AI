# ResearchLens AI: User Guide

Welcome to **ResearchLens AI**, an intelligent, evidence-grounded research-paper analysis and comparative literature review system.

---

## 1. Getting Started & Workspace Management

### 1.1 Collections (Workspaces)
ResearchLens AI organizes research papers into **Collections**. Each collection maintains its own:
- Uploaded papers and raw page texts
- Section-aware chunks and persistent FAISS vector index
- BM25 lexical search index
- Structured extractions, results tables, and keywords
- Interactive chat sessions and generated reports

**Creating a Collection:**
1. Open the application sidebar.
2. Expand the **➕ New Collection** panel.
3. Enter a descriptive collection name (e.g. *Medical NLP Transformers*) and an optional description.
4. Click **Create**. The workspace switches immediately to the new collection.

---

## 2. Paper Upload & Processing (`1_Paper_Upload.py`)

1. Navigate to **1_Paper_Upload** in the sidebar.
2. Drag and drop one or more research-paper PDF files into the upload box (up to 50 MB each).
3. The system executes:
   - **SHA-256 Deduplication:** Prevents duplicate paper uploads within the collection.
   - **Two-Column Aware Parsing:** Preserves true logical reading order across multi-column layouts.
   - **Header & Footer Stripping:** Removes repetitive running headers and page numbers.
   - **Dehyphenation:** Reconstructs hyphenated words split across line breaks.
   - **Section Detection:** Maps sections (`Abstract`, `Introduction`, `Methodology`, `Results`, `Limitations`, `Conclusion`, `References`).
   - **OCR Fallback:** Scans text density; if a page is scanned, runs OCR or displays a warning banner.
   - **Hybrid Indexing:** Embeds passages into FAISS and indexes them in BM25.
   - **Structured Extraction:** Extracts 9 core research fields with page citations.

---

## 3. Structured Paper Analysis (`2_Paper_Analysis.py`)

1. Select a processed paper from the dropdown.
2. Review the 9 extracted research components:
   - **Objective**
   - **Abstract (Verbatim)**
   - **Methodology**
   - **Datasets**
   - **Algorithms**
   - **Metrics**
   - **Results**
   - **Limitations**
   - **Future Work**
3. **Verification Badges:**
   - 🟢 **Verified Evidence Quote:** The quoted snippet was successfully verified against the source page text using fuzzy matching.
   - 🟡 **Unverified:** The quote could not be confirmed in the text (flagged for review).
4. **Researcher Edit Mode:** If an extracted value needs refinement, click **Edit Field**, modify the text or page number, and click **Save Override**. Overrides are saved permanently in the database.
5. **Results Tables:** View normalized benchmark results extracted from paper tables.

---

## 4. Smart Hybrid Search (`3_Smart_Search.py`)

1. Enter an academic query (e.g., *attention mechanism in EHR discharge summaries*).
2. Filter by paper, section, or year if desired.
3. Review retrieved passages:
   - Paper title, section badge, and 1-indexed page range.
   - Expand the **Score Breakdown** to view Dense score, BM25 rank, RRF score, and Cross-Encoder score.

---

## 5. Grounded Research Chat (`4_Research_Chat.py`)

1. Ask complex research questions across your collection.
2. **Scoping:** Choose to query a single paper, selected papers, or the entire collection.
3. **Verifiable Inline Citations:** Statements are cited inline (e.g. `[Paper 1, p. 3]`).
4. **Citation Drawers:** Click any citation pill to view the exact excerpt and page from the source document.
5. **Evidence Support Badge:** The automated verification engine evaluates claims against cited excerpts and shows an overall support percentage.
6. **Insufficient Evidence Behavior:** If the papers do not contain the answer, ResearchLens AI explicitly states what information is missing instead of hallucinating.

---

## 6. Multi-Paper Comparison (`5_Paper_Comparison.py`)

1. Select 2 to 5 papers to compare.
2. View the generated side-by-side comparison matrix across objectives, methods, datasets, metrics, results, and limitations.
3. **⚠️ Comparability Warning:** If papers evaluate on different datasets or metrics, a prominent alert warning is displayed. Direct ranking of raw scores across different datasets is explicitly forbidden.
4. Export the comparison table to CSV, Markdown, or DOCX.

---

## 7. Literature Review Synthesis (`6_Literature_Review.py`)

1. Select papers to include in the review.
2. Choose citation style (**IEEE** or **APA**) and review length.
3. ResearchLens AI clusters papers into themes and synthesizes:
   - Background Introduction
   - Thematic Synthesis
   - Methodological Comparison
   - Open Research Gaps
   - Conclusion & Formatted References
4. Review draft carries a mandatory academic disclaimer banner and automated claim-support check.

---

## 8. Research Gap Discovery (`7_Research_Gaps.py`)

1. Review synthesized research gap opportunities.
2. Each opportunity aggregates author-declared limitations across papers, showing:
   - Gap statement
   - Supporting papers and exact pages
   - Frequency count
   - Suggested methodological direction
   - Rationale

---

## 9. Trends & Export Center (`8_Trends_and_Export.py`)

1. **Plotly Visualizations:**
   - Keyword frequency distribution
   - Publication year timeline
   - Paper similarity & concept co-occurrence heatmap
2. **Multi-Format Export:**
   - Download complete literature reviews, comparison matrices, and analysis reports in **Markdown (.md)**, **Microsoft Word (.docx)**, and **PDF**.
   - Download **BibTeX (.bib)** bibliography for LaTeX papers.

---

## 10. Human Ground-Truth Labelling & Evaluation (`evaluation/generate_label_template.py`)

To evaluate the system against real research papers without hallucinating or inventing gold labels:

1. **Generate Annotation Templates:**
   Run the CLI template generator for all papers in the database:
   ```bash
   python -m evaluation.generate_label_template --all-db
   ```
   Or for a single PDF:
   ```bash
   python -m evaluation.generate_label_template --pdf path/to/paper.pdf
   ```
2. **Fill the Labelling Sheet:**
   The generator creates both a Markdown guide (`evaluation/testset/<paper_stem>_labelling_sheet.md`) and a structured JSON template (`evaluation/testset/<paper_stem>_template.json`).
   - Read the paper and enter the ground-truth values and verbatim quotes with physical page numbers.
   - Formulate 2–3 retrieval queries and 2–3 RAG questions (including 1 unanswerable question).
3. **Run the Benchmark:**
   Merge your annotated JSON into `evaluation/testset/gold_testset.json` and run:
   ```bash
   python -m evaluation.report
   ```
   This executes the retrieval ablation study (Dense vs BM25 vs Hybrid vs Reranker), extraction F1 benchmark, and RAG faithfulness verification, outputting real empirical scores to `evaluation/RESULTS.md`.
