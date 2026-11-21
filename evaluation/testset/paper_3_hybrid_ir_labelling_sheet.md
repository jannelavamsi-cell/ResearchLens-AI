# ResearchLens AI: Human Ground-Truth Labelling Sheet

> **Instructions for Human Annotator:**
> 1. Read through the research paper `paper_3_hybrid_ir.pdf`.
> 2. Fill in the `gold_value`, `evidence_quote`, and `page` for each research component.
> 3. **Critical Rule:** Every `evidence_quote` MUST be an exact verbatim excerpt from the physical page specified in `page`. Do not paraphrase quotes.
> 4. If a field is not stated in the paper, write `"Not stated in paper"`.
> 5. Create 2–3 retrieval search queries and 2–3 RAG questions with expected citations.

---

## 1. Paper Overview
- **Filename:** `paper_3_hybrid_ir.pdf`
- **Title:** Hybrid Information Retrieval for Scientific Document Search
- **Authors:** Dr. Aris Thorne, Dr. Maya Patel, MIT CSAIL
- **Total Pages:** 2
- **Detected Sections:** `Abstract`, `Future Work`, `Introduction`, `Limitations`, `Methodology`, `Results`

---

## 2. Structured Extraction Ground Truth

Please extract the 8 core scientific components. Provide the exact verbatim snippet and 1-indexed physical page.

### 2.1 Objective
- **Description:** Primary research problem, hypothesis, or goal.
- **Gold Value:** [Enter concise summary here]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 1]

### 2.2 Methodology
- **Description:** Core scientific approach, system architecture, or theoretical framework.
- **Gold Value:** [Enter concise summary here]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 2]

### 2.3 Datasets
- **Description:** Benchmark datasets, corpora, or experimental data used.
- **Gold Value:** [Enter dataset names, e.g. MIMIC-III, ImageNet]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 3]

### 2.4 Algorithms
- **Description:** Specific machine learning models, algorithms, or baselines implemented.
- **Gold Value:** [Enter algorithm/model names]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 3]

### 2.5 Metrics
- **Description:** Quantitative evaluation metrics used to measure performance.
- **Gold Value:** [e.g. Macro F1, MRR@10, BLEU, Accuracy]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 4]

### 2.6 Results
- **Description:** Key quantitative findings, primary scores, or comparative outcomes.
- **Gold Value:** [e.g. 88.4% Macro F1, outperforming baseline by 3.2%]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 4]

### 2.7 Limitations
- **Description:** Explicit author-acknowledged constraints, assumptions, or failure cases.
- **Gold Value:** [Enter author-declared limitations]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 5]

### 2.8 Future Work
- **Description:** Directions for future investigation explicitly stated by the authors.
- **Gold Value:** [Enter planned future extensions]
- **Evidence Quote:** "[Enter verbatim text from paper here]"
- **Page Number:** [e.g. 5]

---

## 3. Information Retrieval Benchmark Queries

Write 2 to 4 realistic search queries that a researcher looking for this paper would type.

### Query 1
- **Query Text:** [e.g. What dataset is used for clinical diagnostic code evaluation?]
- **Relevant Keywords:** [keyword1, keyword2, keyword3]
- **Expected Target Paper:** `paper_3_hybrid_ir.pdf`

### Query 2
- **Query Text:** [Enter search query]
- **Relevant Keywords:** [keyword1, keyword2]
- **Expected Target Paper:** `paper_3_hybrid_ir.pdf`

---

## 4. RAG Question Answering Benchmark

Write 2 to 3 factual questions answered by this paper, plus 1 negative question not answered by this paper.

### Question 1 (Factual)
- **Question:** [Enter complex research question]
- **Expected Key Facts:** [Fact 1, Fact 2]
- **Expected Citations:** [`[paper_3_hybrid_ir, p. X]`]

### Question 2 (Factual)
- **Question:** [Enter second research question]
- **Expected Key Facts:** [Fact 1, Fact 2]
- **Expected Citations:** [`[paper_3_hybrid_ir, p. Y]`]

### Question 3 (Negative / Insufficient Evidence Test)
- **Question:** [Enter question about a topic completely absent from this paper]
- **Expected Response:** "The uploaded papers do not contain sufficient evidence..."
- **Is Insufficient:** true
