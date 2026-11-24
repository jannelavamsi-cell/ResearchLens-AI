# Table Detection & Results Extraction Audit (After Filtering)

> **Audit Date:** 2026-11-24  
> **Target:** Verification of defensive filtering thresholds, academic taxonomy classification, caption linking, verbatim cell extraction, and structured results normalization across corpus papers (`data/researchlens.db` and `data/uploaded_papers/`).

---

## 1. Executive Summary & Before vs. After Comparison

In the baseline implementation (`docs/table_audit_before.md`), table extraction relied on raw PyMuPDF `find_tables()` with no defensive rejection thresholds, no academic taxonomy classification, no caption extraction, and no structured results normalization. Furthermore, raw PyMuPDF table extraction suffered from split decimal points across cell newlines (e.g. `'803\n.'` instead of `'80.3'`).

In the updated pipeline, we implemented:
1. **Verbatim Intact Cell Extraction:** Using cell bounding boxes (`page.get_text("words", clip=Rect(cell_bbox))`) to ensure decimal numbers (`80.3`, `48.85`, `+14.9`) remain unbroken.
2. **Defensive Filtering Thresholds:** Documented thresholds rejecting non-tables: minimum 2 rows/columns, consistent column counts, max 60% empty cells, max 120 char average cell length, prose narrative sentence detector (digit-ratio aware), references/bibliography section detector (caption-aware), and vertical page height coverage limit (<= 85%).
3. **Caption Linking:** Bounding-box proximity parser extracting standard `Table N: ...` and `TABLE [IVXLCDM\d]+: ...` headers directly from the PDF canvas.
4. **Four-Way Academic Taxonomy:** Strict classification into `results`, `hyperparameters`, `dataset_statistics`, or `other`.
5. **Strict Structured Results Normalization:** Decomposing empirical evaluation tables into canonical `{model, dataset, metric, value, page}` tuples. Crucially, if a table contains collapsed sub-rows, squashed multiline models, or ambiguous multi-level headers, the normalizer strictly sets `reliable=False`, filters out squashed row tuples, and preserves the verbatim raw table as canonical truth.

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

## 2. Spot-Checks: 5 Accepted Tables (Verbatim Database Dumps)

The following spot-checks are verbatim command-line printouts extracted directly from `data/researchlens.db`. No values were typed by hand.

### Spot-Check 1: Paper 7 (LongRAG) - Page 6, Table 1 (Empirical Multi-Hop Results)
- **Page:** 6 | **Table Index:** 1
- **Linked Caption:** `Table 1: Results (%) of overall performance on three multi-hop datasets. The "Grey Areas" represent different categories of baselines or our system with different fine-tuning settings. “Bold Font” denotes the highest absolute value, while "Underlined Font" expresses the highest relative gain value compared to Vanilla RAG. Ours with (or`
- **Classification:** `results`
- **Shape:** 14 rows × 5 columns
- **Headers:** `["Model", "HotpotQA", "2WikiMQA", "MusiQue", "Average"]`
- **Normalization Reliable:** `False`
- **Normalization Note:** `Irregular row/column alignment or collapsed sub-rows prevent unambiguous decomposition; raw table preserved.`
- **Normalized Records Count:** 16

```text
QUERY: sqlite3 data/researchlens.db "SELECT table_json FROM results_tables WHERE paper_id = '16718843-97ea-4e24-8965-cee147facac3' AND page_number = 6 AND table_index = 1"
VERBATIM STORED ROWS:
Row 0: ["", "", "", "", ""]
Row 1: ["#", "Long-Context LLM Methods #", "", "", ""]
Row 2: ["LongAlign-7B-64k (Llama2) (Bai et al., 2024) LongLoRA-13B-32k (Llama2) (Chen et al., 2023b)", "48.85 47.45", "28.56 42.92", "25.14 29.46", "34.18 39.94"]
Row 3: ["", "# Advanced RAG Methods #", "", "", ""]
Row 4: ["CFIC-7B (Llama2) (Qian et al., 2024) CRAG (GPT-3.5-Turbo) (Yan et al., 2024) Self-RAG (GPT-3.5-Turbo) (Asai et al., 2023)", "34.00 52.04 50.51", "- 41.13 46.75", "14.70 25.34 24.62", "24.35 39.50 40.63"]
Row 5: ["", "# RAG-Base (Vanilla RAG) #", "", "", ""]
Row 6: ["Vicuna-v1.5-7B-16k (Zheng et al., 2023) Qwen-1.5-7B-32k (Bai et al., 2023a) Llama3-8B-8k (Touvron et al., 2023) ChatGLM3-6B-32k (Du et al., 2022) GPT-3.5-Turbo-16k GPT-3.5-Turbo Llama3-70B-8k GLM-4", "38.63 45.70 48.25 52.57 50.17 52.31 52.33 57.41", "27.92 34.69 43.47 42.56 45.32 43.44 50.23 52.91", "15.68 25.08 19.66 25.51 21.84 25.22 25.49 27.55", "27.41 35.16 37.13 40.21 39.11 40.32 42.68 45.96"]
Row 7: ["", "# Ours with SFT #", "", "", ""]
Row 8: ["LongRAG-Llama2-7B-4k LongRAG-Llama2-13B-4k LongRAG-Qwen-1.5-7B-32k LongRAG-Llama3-8B-8k LongRAG-Vicuna-v1.5-7B-16k", "53.85 57.05 52.91 (7.21↑) 52.39 (4.14↑) 55.55 (16.92↑)", "45.61 49.95 46.65 (11.96↑) 49.67 (6.20↑) 50.13 (22.21↑)", "26.22 33.63 31.85 (6.77↑) 31.70 (12.04↑) 28.29 (12.61↑)", "41.89 46.88 43.80 (8.65↑) 44.59 (7.46↑) 44.66 (17.25↑)"]
Row 9: ["LongRAG-ChatGLM3-6B-32k", "55.93 (3.36↑)", "54.85 (12.29↑)", "33.00 (7.49↑)", "47.93 (7.71↑)"]
Row 10: ["", "# Ours without SFT", "#", "", ""]
Row 11: ["LongRAG-GPT-3.5-Turbo", "56.17 (3.86↑)", "51.37 (7.93↑)", "32.83 (7.61↑)", "46.79 (6.47↑)"]
Row 12: ["LongRAG-GPT-3.5-Turbo-16k", "59.11 (8.94↑)", "51.25 (5.93↑)", "30.37 (8.53↑)", "46.91 (7.80↑)"]
Row 13: ["LongRAG-GLM-4", "62.11 (4.70↑)", "57.16 (4.25↑)", "38.40 (10.85↑)", "52.56 (6.60↑)"]

NORMALIZED RECORDS (EXTRACTED TUPLES):
  {"model": "LongRAG-ChatGLM3-6B-32k", "dataset": "HotpotQA", "metric": "Score (%)", "value": "55.93 (3.36↑)", "page": 6}
  {"model": "LongRAG-ChatGLM3-6B-32k", "dataset": "2WikiMQA", "metric": "Score (%)", "value": "54.85 (12.29↑)", "page": 6}
  {"model": "LongRAG-ChatGLM3-6B-32k", "dataset": "MusiQue", "metric": "Score (%)", "value": "33.00 (7.49↑)", "page": 6}
  {"model": "LongRAG-ChatGLM3-6B-32k", "dataset": "Benchmark Dataset", "metric": "Average", "value": "47.93 (7.71↑)", "page": 6}
  {"model": "LongRAG-GPT-3.5-Turbo", "dataset": "HotpotQA", "metric": "Score (%)", "value": "56.17 (3.86↑)", "page": 6}
  {"model": "LongRAG-GPT-3.5-Turbo", "dataset": "2WikiMQA", "metric": "Score (%)", "value": "51.37 (7.93↑)", "page": 6}
  {"model": "LongRAG-GPT-3.5-Turbo", "dataset": "MusiQue", "metric": "Score (%)", "value": "32.83 (7.61↑)", "page": 6}
  {"model": "LongRAG-GPT-3.5-Turbo", "dataset": "Benchmark Dataset", "metric": "Average", "value": "46.79 (6.47↑)", "page": 6}
  ... (8 more tuples)
```

---

### Spot-Check 2: Paper 6 (RetrievalQA) - Page 4, Table 1 (Main Evaluation Table)
- **Page:** 4 | **Table Index:** 1
- **Linked Caption:** `Table 1: Retrieval and match accuracy on RetrievalQA. * indicates using 250 examples for testing to reduce API costs. Best scores in Bold and second best in underline.`
- **Classification:** `results`
- **Shape:** 8 rows × 4 columns
- **Headers:** `["Baselines (1,271)", "Adaptive", "Retrieval", "Always Retrieval"]`
- **Normalization Reliable:** `False`
- **Normalization Note:** `Irregular row/column alignment or collapsed sub-rows prevent unambiguous decomposition; raw table preserved.`
- **Normalized Records Count:** 6

```text
QUERY: sqlite3 data/researchlens.db "SELECT table_json FROM results_tables WHERE paper_id = '46353944-90f2-435b-a40d-5822cec352e1' AND page_number = 4 AND table_index = 1"
VERBATIM STORED ROWS:
Row 0: ["Self-RAG (7B) t = 0.25 t = 0.5 t = 0.75", "Retrieval Match Calibration-based 100.0 31.9 23.0 10.6 0.0 6.0", "", "Match 31.9"]
Row 1: ["t = None Vanilla §3", "0.4", "6.0 Model-based", ""]
Row 2: ["TinyLlama (1.1B) Phi-2 (2.7B)", "39.1 94.1", "14.7 35.0", "28.2 36.4"]
Row 3: ["Llama-2 (7B)", "80.3", "26.1", "36.0"]
Row 4: ["GPT-3.5", "49.3", "20.8", "38.2"]
Row 5: ["GPT-4* Ours TA-ARE §4", "67.6", "37.6", "46.0"]
Row 6: ["TinyLlama (1.1B) Phi-2 (2.7B) Llama-2 (7B) GPT-3.5 GPT-4*", "54.1(+15.0) 95.5(+1.4) 86.0(+5.7) 86.3(+37.0) 83.2(+15.6)", "19.0(+4.3) 36.0(+1.0) 30.7(+4.6) 35.8(+15.0) 46.4(+8.8)", "28.2 36.4 36.0 38.2 46.0"]
Row 7: ["Average gain", "+14.9", "+6.7", "–"]

NORMALIZED RECORDS (EXTRACTED TUPLES):
  {"model": "Llama-2 (7B)", "dataset": "RetrievalQA", "metric": "Adaptive", "value": "80.3", "page": 4}
  {"model": "Llama-2 (7B)", "dataset": "RetrievalQA", "metric": "Retrieval", "value": "26.1", "page": 4}
  {"model": "Llama-2 (7B)", "dataset": "RetrievalQA", "metric": "Always Retrieval", "value": "36.0", "page": 4}
  {"model": "GPT-3.5", "dataset": "RetrievalQA", "metric": "Adaptive", "value": "49.3", "page": 4}
  {"model": "GPT-3.5", "dataset": "RetrievalQA", "metric": "Retrieval", "value": "20.8", "page": 4}
  {"model": "GPT-3.5", "dataset": "RetrievalQA", "metric": "Always Retrieval", "value": "38.2", "page": 4}
```

---

### Spot-Check 3: Paper 6 (RetrievalQA) - Page 11, Table 5 (Implementation Hyperparameters)
- **Page:** 11 | **Table Index:** 2
- **Linked Caption:** `Table 5: Implementation hyperparameters. rameters makes them hallucinate. Since these ques- tions only take a tiny portion of the entire dataset`
- **Classification:** `hyperparameters`
- **Shape:** 7 rows × 2 columns
- **Headers:** `["Parameters", "Values"]`
- **Normalization Reliable:** `False`
- **Normalization Note:** `Rejected or non-results table`
- **Normalized Records Count:** 0

```text
QUERY: sqlite3 data/researchlens.db "SELECT table_json FROM results_tables WHERE paper_id = '46353944-90f2-435b-a40d-5822cec352e1' AND page_number = 11 AND table_index = 2"
VERBATIM STORED ROWS:
Row 0: ["temperature", "0.0"]
Row 1: ["top_p", "1.0"]
Row 2: ["max_tokens", "100"]
Row 3: ["Retrieved docs", "top-5"]
Row 4: ["Threshold (Self-RAG)", "[None, 0.25, 0.5, 0.75]"]
Row 5: ["# demonstrations (TA-ARE)", "4"]
Row 6: ["Eval metric", "match/retrieval accuracy"]
```

---

### Spot-Check 4: Paper 7 (LongRAG) - Page 7, Table 2 (Dataset Corpus Statistics)
- **Page:** 7 | **Table Index:** 1
- **Linked Caption:** `Table 2: Statistics of experimental data. "Avg.`
- **Classification:** `dataset_statistics`
- **Shape:** 1 rows × 4 columns
- **Headers:** `["Dataset", "HotpotQA", "2WikiMQA", "MuSiQue"]`
- **Normalization Reliable:** `False`
- **Normalization Note:** `Rejected or non-results table`
- **Normalized Records Count:** 0

```text
QUERY: sqlite3 data/researchlens.db "SELECT table_json FROM results_tables WHERE paper_id = '16718843-97ea-4e24-8965-cee147facac3' AND page_number = 7 AND table_index = 1"
VERBATIM STORED ROWS:
Row 0: ["Num of Samples Avg. Length of p Num of p Avg. Length of P", "200 1092 1715 9151", "200 535 1464 4887", "200 1032 1877 11214"]
```

---

### Spot-Check 5: Paper 6 (RetrievalQA) - Page 13, Table 11 (Qualitative Data Examples)
- **Page:** 13 | **Table Index:** 2
- **Linked Caption:** `Table 11: Data examples of RetrievalQA (questions need external retrieval).`
- **Classification:** `other`
- **Shape:** 5 rows × 4 columns
- **Headers:** `["Category", "Data Source", "Question", "Answer"]`
- **Normalization Reliable:** `False`
- **Normalization Note:** `Rejected or non-results table`
- **Normalized Records Count:** 0

```text
QUERY: sqlite3 data/researchlens.db "SELECT table_json FROM results_tables WHERE paper_id = '46353944-90f2-435b-a40d-5822cec352e1' AND page_number = 13 AND table_index = 2"
VERBATIM STORED ROWS:
Row 0: ["New world", "RealTimeQA (Kasai et al., 2023)", "Which 2024 Republican presidential contender announced that he is ending his campaign?", "Former Texas Rep. Will Hurd"]
Row 1: ["knowledge", "FreshQA (Vu et al., 2023)", "What is the latest highest-grossing movie of the week at the Box office?", "Mean Girls"]
Row 2: ["", "ToolQA (Zhuang et al., 2023)", "What time did Grace attend Broadway Show on 2022/02/17?", "8:00 PM"]
Row 3: ["Long-tail knowledge", "PopQA (Mallen et al., 2023)", "What is Henry Feilden’s occupation?", "politician"]
Row 4: ["", "TriviaQA (Joshi et al., 2017)", "Which bird, that breeds in northern Europe in pine and beech forests, has a chestnut brown back, grey head, dark tail, buff breast and a striped black throat?", "fieldfare"]
```


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

## 4. Candid Assessment, Ground-Truth Comparison & Known Limitations

### 4.1 PyMuPDF Decimal Splitting vs. Clean Cell Extraction
- **Issue Discovered:** When calling `table.extract()` directly in PyMuPDF, decimal characters at line ends were split into separate lines (e.g. `'803\n.'` or `'4885\n.'`).
- **Fix:** Switched table extraction in `modules/results_tables.py` and `modules/pdf_processor.py` to extract cell words bounded by `page.get_text("words", clip=Rect(cell_bbox))`. This extracts decimals completely intact (`80.3`, `48.85`, `+14.9`).

### 4.2 Multi-Level Headers & Squashed Rows in Academic PDFs
- **Complex Table Structure:** Real academic benchmark tables (such as RetrievalQA Table 1 on page 4 and LongRAG Table 1 on page 6) frequently use hierarchical two-level headers (e.g. a spanning "Adaptive Retrieval" super-header above "Retrieval" and "Match" subheaders) and collapsed rows without horizontal rule lines.
- **Strict Normalization Reliability:**
  - Previously, `normalization_reliable=True` was erroneously reported on these tables because some rows parsed.
  - Now, the normalizer checks for squashed multi-model entries (`_is_squashed_model`) and multi-value cell entries (`_is_multi_value_cell`). When squashed rows are detected, `unreliable_detected=True` is raised.
  - The table strictly outputs `normalization_reliable=False` with note: `"Irregular row/column alignment or collapsed sub-rows prevent unambiguous decomposition; raw table preserved."`
  - Downstream applications and the UI must rely on the raw table (`rows_data`), which preserves the exact ground truth numbers from the PDF.

### 4.3 Provenance of Earlier Hallucinated Numbers
In an earlier drafting of `docs/table_audit_after.md`, three sample rows were manually typed rather than printed verbatim:
1. `Llama-2 (7B) 80.3 / 79.2 / 78.5`: `80.3` is genuine (Adaptive Retrieval -> Retrieval Acc). However, `79.2` and `78.5` do not appear anywhere in the paper; they were typed from memory/hallucinated during manual markdown drafting.
2. `Vicuna (7B) 76.4 / 75.1 / 74.8`: There is no Vicuna row in Table 1 on page 4. Vicuna appears in Table 4 (page 11) and in LongRAG, while `76.4` is the F1 score of GPT-4* in Table 10 (page 13). The row was hallucinated.
3. `Average gain +4.1 / +2.3`: The true table values are `+14.9 / +6.7 / –`. The claimed numbers were picked from discussion text on page 5.

All spot-checks in this document are now generated exclusively via verbatim SQLite dumps directly from `data/researchlens.db`.
