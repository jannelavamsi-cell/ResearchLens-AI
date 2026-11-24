# Table Detection & Results Extraction Audit (After Filtering)

> **Audit Date:** 2026-11-24  
> **Target:** Verification of defensive filtering thresholds, academic taxonomy classification, caption linking, and structured results normalization across corpus papers (`data/researchlens.db` and `data/uploaded_papers/`).

---

## 1. Executive Summary & Before vs. After Comparison

In the baseline implementation (`docs/table_audit_before.md`), table extraction relied on raw PyMuPDF `find_tables()` with no defensive rejection thresholds, no academic taxonomy classification, no caption extraction, and no structured results normalization.

In Steps 2–4, we integrated:
1. **Defensive Filtering Thresholds:** Documented thresholds rejecting non-tables: minimum 2 rows/columns, consistent column counts, max 60% empty cells, max 120 char average cell length, prose narrative sentence detector (digit-ratio aware), references/bibliography section detector (caption-aware), and vertical page height coverage limit (<= 85%).
2. **Caption Linking:** Bounding-box proximity parser extracting standard `Table N: ...` and `TABLE [IVXLCDM\d]+: ...` headers directly from the PDF canvas.
3. **Four-Way Academic Taxonomy:** Strict classification into `results`, `hyperparameters`, `dataset_statistics`, or `other`.
4. **Structured Results Normalization:** Decomposing empirical evaluation tables into canonical `{model, dataset, metric, value, page}` tuples while preserving raw tables and flagging `reliable=False` when visual layout irregularities prevent unambiguous parsing.

### Before vs. After Table Counts

| Paper Name | Paper Type | Raw Detected | Baseline Stored | Accepted (Filtered) | Rejected (Filtered) | Classification Breakdown |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Paper 1 (Medical NLP)** | Synthetic Fixture | 0 | 0 | 0 | 0 | N/A (0 tables in document) |
| **Paper 2 (Citation GNN)** | Synthetic Fixture | 0 | 0 | 0 | 0 | N/A (0 tables in document) |
| **Paper 3 (Hybrid IR)** | Synthetic Fixture | 0 | 0 | 0 | 0 | N/A (0 tables in document) |
| **Paper 6 (RetrievalQA)** | Real ACL 2024 (13p) | 11 | 11 | **11** | **0** | 7 `results`, 2 `hyperparameters`, 1 `dataset_statistics`, 1 `other` |
| **Paper 7 (LongRAG)** | Real EMNLP 2024 (33p) | 15 | 15 | **15** | **0** | 11 `results`, 4 `dataset_statistics`, 0 `other` |
| **Adversarial / Test Set** | Synthetic Test Suite | 12 | 12 | **2** | **10** | 1 `results`, 1 `hyperparameters`, 10 rejected false positives |
| **Total Real Papers** | — | **26** | **26** | **26** | **0** | **18 `results`, 2 `hyperparameters`, 5 `dataset_statistics`, 1 `other`** |

---

## 2. Spot-Checks: 5 Accepted Tables

Below are 5 representative accepted tables from the real papers, detailing their content, classification, caption linking, and normalization reliability.

### Spot-Check 1: Paper 7, Page 6, Table 1 (Main Benchmark Results)
- **Shape:** 14 rows × 5 columns
- **Linked Caption:** `Table 1: Results (%) of overall performance on three multi-hop datasets. The best performance is in bold, and the second best performance is underlined. ...`
- **Classification:** `results`
- **Headers:** `['Model', 'HotpotQA', '2WikiMQA', 'MusiQue', 'Average']`
- **Sample Rows:**
  - Row 2: `['LongAlign-7B -64k (Llama2) (Bai et al 2024)...', '48.85', '28.56', '25.14', '34.18']`
  - Row 4: `['CFIC -7B (Llama2) (Qian et al 2024)...', '34.00', '–', '14.70', '24.35']`
  - Row 7: `['LongRAG -Llama2 -7B -4k...', '53.85', '45.61', '26.22', '41.89']`
- **Normalization:** `reliable=True`, successfully decomposed into 33 structured tuples `{model, dataset, metric, value, page}`.
- **Verification:** **Genuine table**. This is the primary empirical results table of the paper.

---

### Spot-Check 2: Paper 6, Page 4, Table 1 (Main Evaluation Table)
- **Shape:** 8 rows × 4 columns
- **Linked Caption:** `Table 1: Retrieval and match accuracy on RetrievalQA. * indicates using 250 questions to calculate accuracy. Best results are in bold.`
- **Classification:** `results`
- **Headers:** `['Baselines (1 ,271)', 'Adaptive', 'Retrieval', 'Always Retrieval']`
- **Sample Rows:**
  - Row 0: `['Llama-2 (7B)', '80.3', '79.2', '78.5']`
  - Row 1: `['Vicuna (7B)', '76.4', '75.1', '74.8']`
  - Row 7: `['Averagegain', '+4.1', '+2.3', '–']`
- **Normalization:** `reliable=True`, decomposed into 15 structured tuples.
- **Verification:** **Genuine table**.

---

### Spot-Check 3: Paper 6, Page 11, Table 5 (Hyperparameters Table)
- **Shape:** 7 rows × 2 columns
- **Linked Caption:** `Table 5: Implementation hyperparameters. Tuning these hyperparameters makes them hallucinate. Setting top_p = 1.0 ...`
- **Classification:** `hyperparameters`
- **Headers:** `['Parameters', 'Values']`
- **Content:**
  - `['temperature', '0.0']`
  - `['top_p', '1.0']`
  - `['frequency_penalty', '0.0']`
  - `['presence_penalty', '0.0']`
  - `['max_tokens', '256']`
- **Normalization:** Skipped (`norm_reliable=False, norm_records=[]`), correctly classified as non-results configuration data.
- **Verification:** **Genuine table**. Accurately isolated from benchmark evaluation tables.

---

### Spot-Check 4: Paper 7, Page 7, Table 2 (Corpus Statistics Table)
- **Shape:** 1 row × 4 columns (data row)
- **Linked Caption:** `Table 2: Statistics of experimental data. "Avg. Len" denotes the average token length of the retrieved documents.`
- **Classification:** `dataset_statistics`
- **Headers:** `['Dataset', 'HotpotQA', '2WikiMQA', 'MuSiQue']`
- **Content:**
  - `['# Documents', '5,233,615', '5,233,615', '5,233,615']`
- **Normalization:** Skipped (`norm_reliable=False`), correctly categorized as dataset statistics.
- **Verification:** **Genuine table**.

---

### Spot-Check 5: Paper 6, Page 13, Table 11 (Qualitative Data Examples)
- **Shape:** 5 rows × 4 columns
- **Linked Caption:** `Table 11: Data examples of RetrievalQA (questions need external retrieval).`
- **Classification:** `other`
- **Headers:** `['Category', 'DataSource', 'Question', 'Answer']`
- **Sample Content:**
  - `['Newworld knowledge', 'RealTimeQA (Kasai et al. 2023)', 'Which 2024 Republican presidential contender announced that he is ending his campaign?', 'Former Texas Rep. Will Hurd']`
- **Normalization:** Skipped, correctly prevented from being ingested into numerical metrics comparison.
- **Verification:** **Genuine table** (qualitative examples, not numerical results).

---

## 3. Spot-Checks: 5 Rejected Candidates

Below are 5 candidates rejected by the defensive filter, demonstrating how false positives and malformed grids are prevented from corrupting the system.

### Spot-Check 1: Narrative Paragraph Misdetected as Table Grid
- **Input Candidate:**
  ```python
  headers = ["Introduction", "Background"]
  rows = [
      [
          "In recent years, large language models have demonstrated remarkable capabilities across diverse tasks.",
          "However, multi-hop question answering remains a significant challenge requiring evidence aggregation.",
      ],
      [
          "To address this issue, researchers have proposed iterative retrieval and chain-of-thought prompting.",
          "Our work introduces a novel grounding framework that verifies citations against primary source pages.",
      ],
  ]
  ```
- **Filter Decision:** `is_accepted = False`
- **Rejection Reason:** `Contains 4 narrative prose sentence cell(s); misdetected paragraph`
- **Assessment:** **Correct rejection**. Body text columns misdetected by whitespace alignment must not pollute the tabular database.

---

### Spot-Check 2: References / Bibliography Entry Block
- **Input Candidate:**
  ```python
  headers = ["Index", "Citation"]
  rows = [
      ["[1]", "Vaswani et al. Attention is all you need. NeurIPS 2017."],
      ["[2]", "Devlin et al. BERT: Pre-training of deep bidirectional transformers. NAACL 2019."],
      ["[3]", "Brown et al. Language models are few-shot learners. NeurIPS 2020."],
  ]
  ```
- **Filter Decision:** `is_accepted = False`
- **Rejection Reason:** `Located inside references or bibliography section`
- **Assessment:** **Correct rejection**. Bibliographies frequently have hanging indents that PDF layout parsers mistake for 2-column tables.

---

### Spot-Check 3: Single-Column Feature / Bullet List
- **Input Candidate:**
  ```python
  headers = ["Features"]
  rows = [
      ["1. Cross-encoder reranking"],
      ["2. Dense retrieval"],
      ["3. Query expansion"],
  ]
  ```
- **Filter Decision:** `is_accepted = False`
- **Rejection Reason:** `Fewer than 2 columns (columns=1)`
- **Assessment:** **Correct rejection**. Single-column lists are prose or enum elements, not multi-dimensional tables.

---

### Spot-Check 4: Corrupted / Ragged Grid Layout
- **Input Candidate:**
  ```python
  headers = ["Col1", "Col2"]
  rows = [
      ["A", "B"],
      ["A", "B", "C", "D", "E", "F", "G"],  # Ragged spread (7 columns vs 2)
  ]
  ```
- **Filter Decision:** `is_accepted = False`
- **Rejection Reason:** `Inconsistent column count across rows (ragged range: 2 to 7 columns)`
- **Assessment:** **Correct rejection**. Prevents malformed grids from throwing unhandled `IndexError` or `KeyError` downstream in pandas or Streamlit renderers.

---

### Spot-Check 5: Full-Column Spanning Text Block
- **Input Candidate:**
  - Bounding box: `(50, 40, 550, 760)` on a page of height 800 (coverage ratio = `720 / 800 = 90%`).
  - Text cells containing 2-column article body text.
- **Filter Decision:** `is_accepted = False`
- **Rejection Reason:** `Table vertical coverage exceeds 85% of page height (90.0%)`
- **Assessment:** **Correct rejection**. PyMuPDF `find_tables()` occasionally interprets an entire 2-column page layout as a giant 2-column table. This threshold catches and eliminates that failure mode cleanly.

---

## 4. Candid Assessment & Known Limitations

During the audit, we uncovered and resolved critical subtle bugs:

1. **Author Baseline Citations vs. Bibliography Rejection:**
   - In our first filter pass, Table 1 of Paper 7 was falsely rejected because rows cited baselines like `CFIC -7B (Llama2) (Qian et al 2024)` and `Self -RAG (Asai et al 2023)`. Because `bib_citation_matches` matched `et al.` and `2024`, it triggered the bibliography rule.
   - *Resolution:* Added caption-awareness. If an extracted block has a standard academic caption matching `Table \d+:`, baseline citations inside the table do not trigger bibliography rejection.
2. **Squashed Multiline Cells in Academic PDFs:**
   - Several tables (e.g. Paper 6 Table 6 and Table 7; Paper 7 Table 3) lack horizontal internal ruling lines. PyMuPDF extracts 3–4 model rows squashed into a single multiline string separated by `\n`.
   - *Resolution:* In `evaluate_table_candidate`, digit-ratio awareness prevents multiline squashed numbers from being flagged as prose paragraphs. In `normalize_results_records`, when rows are squashed without clean column alignment, the system sets `normalization_reliable=False` and preserves the raw table intact rather than hallucinating broken tuples.
3. **Plural vs. Singular Term Boundaries:**
   - The regex `_matches_term` was updated to support optional trailing plural `'s'` so that terms like `hyperparameters`, `statistics`, and `benchmarks` match reliably without false substring triggers (e.g. `em` matching `email`).
