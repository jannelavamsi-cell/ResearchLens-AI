"""
Paper comparison module for ResearchLens AI.
Constructs multi-paper comparative matrices (2 to 5 papers) across all 9 research components,
analyzes experimental benchmark comparability, issues prominent comparability warnings,
and synthesizes grounded comparative narratives without invalid cross-dataset rankings.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from core.llm import get_llm_client
from core.llm.prompts import COMPARISON_NARRATIVE_PROMPT
from core.schemas import ComparabilityWarning, ComparisonCell
from database.db_manager import DatabaseManager
from modules.research_extraction import (
    CORE_FIELDS,
    extract_paper_components,
)

logger = logging.getLogger("researchlens.comparison")


def build_comparison_matrix(
    paper_ids: List[str],
    db: DatabaseManager,
) -> Dict[str, Any]:
    """
    Constructs a cross-paper comparative grid for 2 to 5 research papers.
    Ensures extractions exist for each paper and formats cells with page numbers.
    """
    if len(paper_ids) < 2:
        raise ValueError("Comparison requires at least 2 papers.")
    if len(paper_ids) > 5:
        raise ValueError("Comparison supports at most 5 papers simultaneously.")

    papers_meta = []
    paper_extractions: Dict[str, Dict[str, Any]] = {}

    for pid in paper_ids:
        paper = db.get_paper(pid)
        if not paper:
            continue
        papers_meta.append({
            "id": pid,
            "title": paper["title"] or paper["filename"],
            "authors": paper.get("authors") or "Unknown",
            "year": paper.get("year") or "N/A",
            "venue": paper.get("venue") or "N/A",
            "n_pages": paper["n_pages"],
        })

        exts = extract_paper_components(pid, db)
        paper_extractions[pid] = exts

    # Construct matrix rows
    matrix_rows: Dict[str, List[ComparisonCell]] = {}

    for field_name in CORE_FIELDS:
        cells = []
        for p_meta in papers_meta:
            pid = p_meta["id"]
            field_obj = paper_extractions.get(pid, {}).get(field_name)

            if field_obj:
                cell = ComparisonCell(
                    paper_id=pid,
                    field_name=field_name,
                    value=field_obj.value,
                    evidence_quote=field_obj.evidence_quote,
                    page=field_obj.page,
                    verified=field_obj.verified,
                )
            else:
                cell = ComparisonCell(
                    paper_id=pid,
                    field_name=field_name,
                    value="Not stated in paper",
                    evidence_quote="",
                    page=0,
                    verified=True,
                )
            cells.append(cell)
        matrix_rows[field_name] = cells

    return {
        "papers": papers_meta,
        "rows": matrix_rows,
    }


def check_comparability(matrix: Dict[str, Any]) -> ComparabilityWarning:
    """
    Analyzes datasets and metrics across the compared papers.
    Detects if papers evaluate on non-identical benchmarks or metrics,
    issuing a strict warning against raw numerical ranking.
    """
    rows = matrix["rows"]

    datasets_cells = rows.get("datasets", [])
    metrics_cells = rows.get("metrics", [])

    reasons = []

    # 1. Compare datasets
    dataset_values = [c.value.lower().strip() for c in datasets_cells if c.value != "Not stated in paper"]
    # Check pairwise similarity or string divergence
    distinct_datasets = set()
    for d in dataset_values:
        # Simplify key dataset names
        for token in ["mimic", "cora", "citeseer", "pubmed", "ms-marco", "imagenet", "squad", "glue"]:
            if token in d:
                distinct_datasets.add(token.upper())

    if len(distinct_datasets) > 1:
        reasons.append(
            f"Different benchmark datasets deployed: {', '.join(sorted(distinct_datasets))}."
        )
    elif len(dataset_values) > 1 and len(set(dataset_values)) > 1:
        # Generic difference detected
        reasons.append(
            "Distinct dataset definitions or experimental splits used across papers."
        )

    # 2. Compare metrics
    metric_values = [c.value.lower().strip() for c in metrics_cells if c.value != "Not stated in paper"]
    distinct_metrics = set()
    for m in metric_values:
        for token in ["accuracy", "f1", "macro f1", "micro f1", "precision", "recall", "mrr", "ndcg", "bleu", "rouge", "auc"]:
            if token in m:
                distinct_metrics.add(token.title())

    if len(distinct_metrics) > 1:
        reasons.append(
            f"Non-identical evaluation metrics measured: {', '.join(sorted(distinct_metrics))}."
        )

    if reasons:
        msg = (
            "⚠️ Comparability Warning: The compared papers evaluate on distinct datasets or metric definitions. "
            "Direct raw numerical comparison or ranking of performance metrics is scientifically invalid and misleading."
        )
        return ComparabilityWarning(
            has_warning=True,
            reasons=reasons,
            message=msg,
        )

    return ComparabilityWarning(
        has_warning=False,
        reasons=[],
        message="Papers appear to evaluate on comparable benchmarks and metrics.",
    )


def generate_comparison_narrative(
    matrix: Dict[str, Any],
    warning: ComparabilityWarning,
) -> str:
    """
    Synthesizes a grounded cross-paper comparative narrative using the LLM.
    Explicitly incorporates comparability caveats and enforces page-level citations.
    """
    papers = matrix["papers"]
    rows = matrix["rows"]

    # Build prompt context table
    table_lines = []
    paper_titles = [f"Paper {i+1}: {p['title']} ({p['year']})" for i, p in enumerate(papers)]
    table_lines.append("Papers being compared:\n" + "\n".join(paper_titles) + "\n")

    for field_name, cells in rows.items():
        table_lines.append(f"### {field_name.upper()}")
        for idx, cell in enumerate(cells):
            p = papers[idx]
            cit = f"[p. {cell.page}]" if cell.page > 0 else ""
            table_lines.append(f"- **{p['title']}**: {cell.value} {cit}")
        table_lines.append("")

    context_str = "\n".join(table_lines)

    warning_directive = ""
    if warning.has_warning:
        warning_directive = (
            f"MANDATORY WARNING: {warning.message}\n"
            f"Detailed reasons: {'; '.join(warning.reasons)}\n"
            f"You MUST explicitly include this comparability caveat in your analysis and "
            f"refrain from claiming any paper has superior raw accuracy."
        )

    prompt = (
        f"Compare the following research papers across their objectives, methodologies, algorithms, results, and limitations.\n\n"
        f"{warning_directive}\n\n"
        f"{context_str}\n\n"
        f"Write a structured 3-part comparison:\n"
        f"1. Methodological Approaches & Algorithmic Trade-offs\n"
        f"2. Experimental Benchmarks & Results Synthesis (with comparability caveats)\n"
        f"3. Author-Declared Limitations & Future Directions\n"
        f"Cite page numbers [Title, p. X] for all specific claims."
    )

    try:
        client = get_llm_client()
        narrative = client.generate(
            prompt=prompt,
            system_prompt=COMPARISON_NARRATIVE_PROMPT,
            temperature=0.1,
            max_tokens=900,
        )
        if narrative and len(narrative.strip()) > 50:
            return narrative.strip()
    except Exception as err:
        logger.warning(f"LLM comparison narrative unavailable ({err}); constructing heuristic synthesis.")

    # Grounded heuristic synthesis fallback
    lines = []
    lines.append("### 1. Methodological Approaches & Algorithmic Trade-offs")
    for idx, p in enumerate(papers):
        meth_cell = rows["methodology"][idx]
        algo_cell = rows["algorithms"][idx]
        lines.append(
            f"- **{p['title']}** ({p['year']}): Utilizes {algo_cell.value} [p. {algo_cell.page}]. "
            f"Approach: {meth_cell.value} [p. {meth_cell.page}]."
        )

    lines.append("\n### 2. Experimental Benchmarks & Results Synthesis")
    if warning.has_warning:
        lines.append(f"> **Comparability Notice:** {warning.message}")
        for r in warning.reasons:
            lines.append(f"> - {r}")
        lines.append("")

    for idx, p in enumerate(papers):
        d_cell = rows["datasets"][idx]
        m_cell = rows["metrics"][idx]
        r_cell = rows["results"][idx]
        lines.append(
            f"- **{p['title']}**: Evaluated on {d_cell.value} [p. {d_cell.page}] "
            f"measuring {m_cell.value} [p. {m_cell.page}]. Finding: {r_cell.value} [p. {r_cell.page}]."
        )

    lines.append("\n### 3. Author-Declared Limitations & Future Directions")
    for idx, p in enumerate(papers):
        lim_cell = rows["limitations"][idx]
        fut_cell = rows["future_work"][idx]
        lines.append(
            f"- **{p['title']}**: Limitation: {lim_cell.value} [p. {lim_cell.page}]. "
            f"Future work: {fut_cell.value} [p. {fut_cell.page}]."
        )

    return "\n".join(lines)


def matrix_to_dataframe(matrix: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts comparison matrix into a presentation pandas DataFrame.
    Rows are research fields; columns are paper titles with citation pills in cells.
    """
    papers = matrix["papers"]
    rows = matrix["rows"]

    col_names = [f"{p['title']} ({p['year']})" for p in papers]
    data = {}

    for field_name in CORE_FIELDS:
        display_name = field_name.replace("_", " ").title()
        cells = rows.get(field_name, [])
        col_values = []
        for c in cells:
            cit = f" [p. {c.page}]" if c.page > 0 else ""
            col_values.append(f"{c.value}{cit}")
        data[display_name] = col_values

    df = pd.DataFrame(data, index=col_names).T
    return df


def matrix_to_markdown(
    matrix: Dict[str, Any],
    warning: ComparabilityWarning,
    narrative: str,
) -> str:
    """
    Renders complete comparative report into GitHub Flavored Markdown.
    """
    papers = matrix["papers"]
    df = matrix_to_dataframe(matrix)

    md_lines = []
    md_lines.append("# ResearchLens AI: Comparative Paper Analysis\n")
    md_lines.append(f"**Compared Papers ({len(papers)}):**")
    for p in papers:
        md_lines.append(f"- **{p['title']}** ({p['authors']}, {p['year']})")
    md_lines.append("\n---\n")

    if warning.has_warning:
        md_lines.append(f"> ⚠️ **Comparability Warning**  \n> {warning.message}\n>")
        for r in warning.reasons:
            md_lines.append(f"> - {r}")
        md_lines.append("\n---\n")

    md_lines.append("## Cross-Paper Comparison Matrix\n")
    md_lines.append(df.to_markdown())
    md_lines.append("\n\n---\n")

    md_lines.append("## Comparative Synthesis Narrative\n")
    md_lines.append(narrative)
    md_lines.append("\n")

    return "\n".join(md_lines)
