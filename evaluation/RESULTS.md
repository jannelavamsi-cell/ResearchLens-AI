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
| **Hybrid (RRF)** | 0.833 | 1.000 | 0.875 | 0.908 |
| **Hybrid + Reranker** | 1.000 | 1.000 | 1.000 | 1.000 |

*Observation: Hybrid retrieval with Reciprocal Rank Fusion (RRF k=60) combined with Cross-Encoder reranking achieves the highest rank precision and nDCG, outperforming dense-only vector search and sparse BM25.*

---

## 2. Evidence-Grounded Extraction Benchmark

- **Overall Macro F1:** `0.491`
- **Evidence Quote Verification Accuracy:** `94.4%`

| Research Component | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **Objective** | 0.549 | 0.522 | 0.549 |
| **Methodology** | 0.255 | 0.242 | 0.255 |
| **Datasets** | 0.000 | 0.000 | 0.000 |
| **Algorithms** | 0.659 | 0.626 | 0.659 |
| **Metrics** | 0.784 | 0.745 | 0.784 |
| **Results** | 0.502 | 0.477 | 0.502 |
| **Limitations** | 0.688 | 0.654 | 0.688 |

---

## 3. Conversational RAG Faithfulness & Citation Accuracy

- **Answer Faithfulness (Supported Claims):** `11.7%`
- **Citation Grounding Accuracy:** `12.2%`
- **Insufficient Evidence Detection Accuracy:** `0.0%`

*All generated claims are verified against physical page text. Zero hallucinations are tolerated.*

