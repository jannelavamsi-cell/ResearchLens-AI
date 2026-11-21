"""
ResearchLens AI: Human Annotation Template Generator.
Generates structured Markdown labelling sheets and JSON templates for human researchers
to ground truth real research papers without hallucinating or inventing gold labels.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager  # noqa: E402
from modules.pdf_processor import PDFProcessor  # noqa: E402
from modules.section_detection import detect_sections_from_pages  # noqa: E402

logger = logging.getLogger("researchlens.eval.template_gen")


def generate_paper_markdown_sheet(
    filename: str,
    title: str,
    authors: str,
    num_pages: int,
    detected_sections: List[str],
    output_dir: str = "evaluation/testset",
) -> str:
    """Generates a structured Markdown labelling sheet for human annotation."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(filename).stem
    out_file = os.path.join(output_dir, f"{stem}_labelling_sheet.md")

    sections_str = ", ".join(f"`{s}`" for s in detected_sections) if detected_sections else "None detected"

    md_content = f"""# ResearchLens AI: Human Ground-Truth Labelling Sheet

> **Instructions for Human Annotator:**
> 1. Read through the research paper `{filename}`.
> 2. Fill in the `gold_value`, `evidence_quote`, and `page` for each research component.
> 3. **Critical Rule:** Every `evidence_quote` MUST be an exact verbatim excerpt from the physical page specified in `page`. Do not paraphrase quotes.
> 4. If a field is not stated in the paper, write `"Not stated in paper"`.
> 5. Create 2–3 retrieval search queries and 2–3 RAG questions with expected citations.

---

## 1. Paper Overview
- **Filename:** `{filename}`
- **Title:** {title or "Unknown Title"}
- **Authors:** {authors or "Unknown Authors"}
- **Total Pages:** {num_pages}
- **Detected Sections:** {sections_str}

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
- **Expected Target Paper:** `{filename}`

### Query 2
- **Query Text:** [Enter search query]
- **Relevant Keywords:** [keyword1, keyword2]
- **Expected Target Paper:** `{filename}`

---

## 4. RAG Question Answering Benchmark

Write 2 to 3 factual questions answered by this paper, plus 1 negative question not answered by this paper.

### Question 1 (Factual)
- **Question:** [Enter complex research question]
- **Expected Key Facts:** [Fact 1, Fact 2]
- **Expected Citations:** [`[{stem}, p. X]`]

### Question 2 (Factual)
- **Question:** [Enter second research question]
- **Expected Key Facts:** [Fact 1, Fact 2]
- **Expected Citations:** [`[{stem}, p. Y]`]

### Question 3 (Negative / Insufficient Evidence Test)
- **Question:** [Enter question about a topic completely absent from this paper]
- **Expected Response:** "The uploaded papers do not contain sufficient evidence..."
- **Is Insufficient:** true
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return out_file


def generate_paper_json_template(
    filename: str,
    output_dir: str = "evaluation/testset",
) -> str:
    """Generates a blank JSON schema file for the paper ready for manual filling."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(filename).stem
    out_file = os.path.join(output_dir, f"{stem}_template.json")

    template_data: Dict[str, Any] = {
        "filename": filename,
        "gold_fields": {
            "objective": "FILL_ME",
            "methodology": "FILL_ME",
            "datasets": "FILL_ME",
            "algorithms": "FILL_ME",
            "metrics": "FILL_ME",
            "results": "FILL_ME",
            "limitations": "FILL_ME",
            "future_work": "FILL_ME",
        },
        "evidence_quotes": {
            "objective": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "methodology": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "datasets": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "algorithms": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "metrics": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "results": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "limitations": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
            "future_work": {"quote": "FILL_VERBATIM_QUOTE", "page": 1},
        },
        "retrieval_queries": [
            {
                "query": "FILL_SEARCH_QUERY_HERE",
                "relevant_keywords": ["keyword1", "keyword2"],
                "expected_paper": filename,
            }
        ],
        "rag_questions": [
            {
                "question": "FILL_RAG_QUESTION_HERE",
                "expected_answer_contains": ["key_fact_1", "key_fact_2"],
                "expected_citations": [f"{filename}, p. 1"],
                "is_insufficient": False,
            }
        ],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(template_data, f, indent=2)

    return out_file


def generate_templates_for_paper_path(
    pdf_path: str,
    output_dir: str = "evaluation/testset",
) -> Dict[str, str]:
    """Inspects a PDF file directly and generates Markdown and JSON templates."""
    processor = PDFProcessor()
    pages, _tables, doc_meta = processor.process_pdf(pdf_path)
    pages_tuples = [(p.page_number, p.text) for p in pages]
    sections = detect_sections_from_pages(pages_tuples)
    section_names = list({s.section_name for s in sections})

    filename = os.path.basename(pdf_path)
    md_path = generate_paper_markdown_sheet(
        filename=filename,
        title=doc_meta.get("title") or Path(filename).stem.replace("_", " ").title(),
        authors=doc_meta.get("authors") or "Unknown",
        num_pages=len(pages),
        detected_sections=section_names,
        output_dir=output_dir,
    )
    json_path = generate_paper_json_template(filename=filename, output_dir=output_dir)

    print(f"Generated Markdown sheet: {md_path}")
    print(f"Generated JSON template:  {json_path}")
    return {"markdown": md_path, "json": json_path}


def generate_templates_for_database(
    db: Optional[DatabaseManager] = None,
    output_dir: str = "evaluation/testset",
) -> List[Dict[str, str]]:
    """Generates templates for all papers currently recorded in the database."""
    if db is None:
        db = DatabaseManager()

    results = []
    with db.get_connection() as conn:
        papers = conn.execute("SELECT id, filename, title, authors, n_pages FROM papers").fetchall()

    for p in papers:
        # Get detected sections from chunks
        chunks = db.get_chunks(p["id"])
        section_names = sorted(list({c["section_name"] for c in chunks if c.get("section_name")}))

        md_path = generate_paper_markdown_sheet(
            filename=p["filename"],
            title=p["title"] or p["filename"],
            authors=p["authors"] or "Unknown",
            num_pages=p["n_pages"] or 1,
            detected_sections=section_names,
            output_dir=output_dir,
        )
        json_path = generate_paper_json_template(filename=p["filename"], output_dir=output_dir)
        results.append({"filename": p["filename"], "markdown": md_path, "json": json_path})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human annotation labelling templates for papers.")
    parser.add_argument("--pdf", type=str, help="Path to a single research paper PDF.")
    parser.add_argument("--all-db", action="store_true", help="Generate templates for all papers in SQLite DB.")
    parser.add_argument("--out-dir", type=str, default="evaluation/testset", help="Output directory for templates.")

    args = parser.parse_args()

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file not found at {args.pdf}")
            return
        generate_templates_for_paper_path(args.pdf, output_dir=args.out_dir)
    elif args.all_db:
        res = generate_templates_for_database(output_dir=args.out_dir)
        print(f"Successfully generated labelling sheets for {len(res)} papers in database.")
    else:
        # Default: generate for DB papers
        res = generate_templates_for_database(output_dir=args.out_dir)
        print(f"Generated labelling sheets for {len(res)} papers. Use --pdf <path> for specific PDFs.")


if __name__ == "__main__":
    main()
