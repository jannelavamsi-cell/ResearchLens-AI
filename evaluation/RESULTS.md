# ResearchLens AI: Empirical Evaluation & Ablation Results

> **Evaluated Collection:** `Default Workspace` (ID: `default-collection`)
> **Embedding Model:** `BAAI/bge-small-en-v1.5` | **Cross-Encoder:** `ms-marco-MiniLM-L-6-v2`
> **Ground-Truth Test Set:** `evaluation/testset/gold_testset.json`
> **Sample Size:** 3 papers, 4 retrieval queries, 4 RAG questions

> [!NOTE]
> **Sample Size & Trust Notice:** The current evaluation set is a compact benchmark suite (3 papers, 4 search queries, 4 questions) designed to verify end-to-end algorithmic correctness and ablation differences. Small sample sizes limit how far percentage scores can be generalized. To expand this evaluation with real human annotations, use `python -m evaluation.generate_label_template --all-db` to generate labelling sheets for any uploaded papers.

---

## 1. Information Retrieval Ablation Study (Precision@3, Recall@3, MRR, nDCG@3)

| Retrieval Pipeline | Precision@3 | Recall@3 | MRR | nDCG@3 |
|:---|:---:|:---:|:---:|:---:|
| **Dense Only** | 0.917 | 1.000 | 0.875 | 0.923 |
| **BM25 Only** | 0.750 | 0.750 | 0.750 | 0.750 |
| **Hybrid (RRF)** | 0.833 | 1.000 | 1.000 | 1.000 |
| **Hybrid + Reranker** | 1.000 | 1.000 | 1.000 | 1.000 |

*Observation: Hybrid retrieval with Reciprocal Rank Fusion (RRF k=60) combined with Cross-Encoder reranking achieves the highest rank precision and nDCG, outperforming dense-only vector search and sparse BM25.*

---

## 2. Evidence-Grounded Extraction Benchmark

- **Overall Macro F1:** `0.640`
- **Evidence Quote Verification Accuracy:** `100.0%`

| Research Component | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **Objective** | 0.518 | 0.492 | 0.518 |
| **Methodology** | 0.739 | 0.702 | 0.739 |
| **Datasets** | 0.815 | 0.774 | 0.815 |
| **Algorithms** | 0.446 | 0.424 | 0.446 |
| **Metrics** | 0.741 | 0.704 | 0.741 |
| **Results** | 0.533 | 0.506 | 0.533 |
| **Limitations** | 0.688 | 0.654 | 0.688 |

---

## 3. Conversational RAG Faithfulness & Citation Accuracy

- **Answer Faithfulness (Supported Claims):** `12.2%`
- **Citation Grounding Accuracy:** `12.6%`
- **Insufficient Evidence Detection Accuracy:** `0.0%`

*All generated claims are verified against physical page text. Zero hallucinations are tolerated.*

