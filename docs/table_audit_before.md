# Table Detection Audit (Before Filtering)

> **Audit Date:** 2026-11-24  
> **Target:** Evaluation of raw PyMuPDF `find_tables()` detection on corpus papers in `data/researchlens.db` and `data/uploaded_papers/`.

---

## 1. Corpus Paper Classification

The database currently indexes 5 research papers:

| # | Paper Title | Filename | Page Count | Type | Status / Notes |
|---|---|---|:---:|:---:|---|
| 1 | Attention-Based Deep Learning for Medical Document Intelligence | `paper_1_medical_nlp.pdf` | 3 | **Synthetic Sample** | Short fixture paper; 0 tables detected. |
| 2 | Graph Neural Networks for Scientific Citation Link Prediction | `paper_2_citation_gnn.pdf` | 2 | **Synthetic Sample** | Short fixture paper; 0 tables detected. |
| 3 | Hybrid Information Retrieval for Scientific Document Search | `paper_3_hybrid_ir.pdf` | 2 | **Synthetic Sample** | Short fixture paper; 0 tables detected. |
| 4 | RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation for Short-form Open-Domain Question Answering | `2024.findings-acl.415-paper-6.pdf` | 13 | **Real Paper** | Findings of ACL 2024; 11 tables detected. |
| 5 | LongRAG: A Dual-Perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering | `2024.emnlp-main.1259-paper-7.pdf` | 33 | **Real Paper** | EMNLP 2024 Main; 15 tables detected. |

---

## 2. Table Inventory for Real Academic Papers

### 2.1 Paper 6: RetrievalQA (`2024.findings-acl.415-paper-6.pdf`, 13 Pages)

Total tables detected by raw `page.find_tables()`: **11**

| Table # | Page | Shape (R × C) | First Row / Headers | Detected Caption | Audit Classification |
|:---:|:---:|:---:|---|---|---|
| **T1** | 4 | 9 × 4 | `['Baselines (1 ,271)', 'Adaptive', 'Retrieval', 'Always Retrieval']` | *Table 1: Retrieval and match accuracy on RetrievalQA.* | **Real Results Table** |
| **T2** | 5 | 3 × 5 | `['', 'Time', 'Example', 'Avg. Retrieval', 'Avg. Match']` | *Table 2: Ablation study for current date and demonstration examples.* | **Real Results Table** |
| **T3** | 10 | 6 × 7 | `['Category', 'DataSource', '#Original', '#After Filtering', '# Avg. QTokens', '# Avg. AnsTokens', '# Avg. Doc Tokens']` | *Table 3: Data statistics of RetrievalQA.* | **Other Real Table** (Dataset Statistics) |
| **T4** | 11 | 4 × 3 | `['ModelName', 'ModelSize', 'Release']` | *Table 4: Model used in the experiments.* | **Other Real Table** (Model Overview) |
| **T5** | 11 | 8 × 2 | `['Parameters', 'Values']` | *Table 5: Implementation hyperparameters.* | **Other Real Table** (Hyperparameters) |
| **T6** | 11 | 2 × 3 | `['Model', 'Match', 'F1']` | *Table 6: Match and F1 scores of models on RetrievalQA without retrieval.* | **Real Results Table** *(formatting flaw: multiline squashed into row 1)* |
| **T7** | 11 | 2 × 2 | `['Model', 'Match']` | *Table 7: Match scores of models on 1,517 questions.* | **Real Results Table** *(formatting flaw: multiline squashed into row 1)* |
| **T8** | 11 | 4 × 2 | `['', 'AdaptiveRetrieval']` | *Table 8: Ablation: our TA-ARE without the current date.* | **Real Results Table** |
| **T9** | 11 | 3 × 2 | `['Baselines', 'AdaptiveRetrieval Retrieval Match']` | *Table 9: Ablation: our TA-ARE without demonstration examples.* | **Real Results Table** |
| **T10** | 13 | 10 × 8 | `['Baselines (2 ,785)', 'NoRetrieval', '', 'AdaptiveRetrieval', ...]` | *Table 10: Retrieval and match accuracy on RetrievalQA (overall).* | **Real Results Table** |
| **T11** | 13 | 6 × 4 | `['Category', 'DataSource', 'Question', 'Answer']` | *Table 11: Data examples of RetrievalQA (questions need external retrieval).* | **Other Real Table** (Qualitative Examples) |

**Paper 6 Breakdown:**
- **Real Results Tables:** 7 (T1, T2, T6, T7, T8, T9, T10)
- **Other Real Tables:** 4 (T3 dataset stats, T4 model metadata, T5 hyperparameters, T11 qualitative examples)
- **False Positives:** 0 (PyMuPDF did not misdetect body text or bibliography on this paper, but T6 and T7 suffered severe cell-squashing due to missing horizontal ruling lines).

---

### 2.2 Paper 7: LongRAG (`2024.emnlp-main.1259-paper-7.pdf`, 33 Pages)

Total tables detected by raw `page.find_tables()`: **15**

| Table # | Page | Shape (R × C) | First Row / Headers | Detected Caption | Audit Classification |
|:---:|:---:|:---:|---|---|---|
| **T1** | 6 | 15 × 5 | `['Model', 'HotpotQA', '2WikiMQA', 'MusiQue', 'Average']` | *Table 1: Results (%) of overall performance on three multi-hop datasets.* | **Real Results Table** |
| **T2** | 7 | 2 × 4 | `['Dataset', 'HotpotQA', '2WikiMQA', 'MuSiQue']` | *Table 2: Statistics of experimental data.* | **Other Real Table** (Dataset Statistics) |
| **T3** | 8 | 5 × 3 | `['R&B R&L Ext Fil E&F', 'R&B R&L Ext Fil E&F', 'R&B R&L Ext Fil E&F']` | *Table 3: Results (%) of the ablation study.* | **Real Results Table** |
| **T4** | 14 | 4 × 2 | `['Datasets', 'ReAct (GPT -3 .5 -Turbo)']` | *Table 4: Results of ReAct.* | **Real Results Table** |
| **T5** | 15 | 4 × 2 | `['Datasets', 'Self -RAG (GPT -3 .5 -Turbo)']` | *Table 5: Results of Self-RAG via the second strategy.* | **Real Results Table** |
| **T6** | 17 | 20 × 5 | `['Model', '', 'HotpotQA', None, '']` | *Table 6: Overall performance of our LongRAG on HotpotQA dataset.* | **Real Results Table** |
| **T7** | 18 | 20 × 3 | `['Model', '2WikiMultiHopQA', None]` | *Table 7: Overall performance of our LongRAG on 2WikiMultiHopQA dataset.* | **Real Results Table** |
| **T8** | 19 | 20 × 5 | `['Model', '', 'MusiQue', None, '']` | *Table 8: Overall performance of our LongRAG on MusiQue dataset.* | **Real Results Table** |
| **T9** | 20 | 5 × 4 | `['', '', 'HotpotQA', '']` | *Table 9: Analysis of the component transferability of E&F on HotpotQA.* | **Real Results Table** |
| **T10** | 20 | 5 × 4 | `['', '', '2WikiMultiHopQA', None]` | *Table 10: Analysis of the component transferability of E&F on 2WikiMultiHopQA.* | **Real Results Table** |
| **T11** | 20 | 5 × 4 | `['', '', 'MusiQue', '']` | *Table 11: Analysis of the component transferability of E&F on MusiQue.* | **Real Results Table** |
| **T12** | 21 | 6 × 5 | `['', '', 'HotpotQA', None, '']` | *Table 12: Values of the token length fed into the generator on HotpotQA.* | **Other Real Table** (Token Length Stats) |
| **T13** | 21 | 6 × 6 | `['', '', '2WikiMultiHopQA', None, None, '']` | *Table 13: Values of the token length fed into the generator on 2WikiMultiHopQA.* | **Other Real Table** (Token Length Stats) |
| **T14** | 21 | 6 × 6 | `['', '', 'MusiQue', None, '', '']` | *Table 14: Values of the token length fed into the generator on MusiQue.* | **Other Real Table** (Token Length Stats) |
| **T15** | 22 | 6 × 5 | `['Datasets', 'HotpotQA', '2WikiMultiHopQA', 'MusiQue', 'QASPER']` | *Table 15: Statistics of our fine-tuning instruction dataset LRGinstruction.* | **Other Real Table** (Dataset Statistics) |

**Paper 7 Breakdown:**
- **Real Results Tables:** 10 (T1, T3, T4, T5, T6, T7, T8, T9, T10, T11)
- **Other Real Tables:** 5 (T2 dataset stats, T12 token length stats, T13 token length stats, T14 token length stats, T15 fine-tuning stats)
- **False Positives:** 0

---

## 3. Aggregate Table Counts Before Filtering

| Paper Name | Total Detected | Real Results Tables | Other Real Tables | False Positives |
|---|:---:|:---:|:---:|:---:|
| **Paper 1 (Medical NLP)** | 0 | 0 | 0 | 0 |
| **Paper 2 (Citation GNN)** | 0 | 0 | 0 | 0 |
| **Paper 3 (Hybrid IR)** | 0 | 0 | 0 | 0 |
| **Paper 6 (RetrievalQA)** | 11 | 7 | 4 | 0 |
| **Paper 7 (LongRAG)** | 15 | 10 | 5 | 0 |
| **Total** | **26** | **17** | **9** | **0** |

---

## 4. Key Failure Modes & Vulnerabilities Identified in Baseline System

1. **No Explicit Table Taxonomy / Classification:**
   The baseline `is_likely_results_table()` function only returns a binary boolean flag (`True`/`False`). It cannot distinguish between empirical evaluation results, hyperparameters, dataset statistics, qualitative text examples, and arbitrary layout grids.
2. **Missing Caption Linking:**
   In the baseline DB records, `caption` is uniformly empty string (`""`) or `None`, even though rich `Table N: ...` captions exist directly above or below the bounding box in the PDF.
3. **Lack of Defensive Filtering Thresholds:**
   The current code lacks explicit filters for:
   - Paragraph text misdetected as tables (average cell length > 120 chars, full punctuation sentences).
   - Wildly ragged column counts (inconsistent grid structure).
   - Tables embedded inside the References / Bibliography section.
   - Tables covering almost the entirety of a page text column.
4. **No Structured Results Normalization:**
   Results tables are stored solely as raw grid arrays (`List[List[str]]`). There is no structured entity extraction converting results into canonical tuples: `{model, dataset, metric, value, page}`.
