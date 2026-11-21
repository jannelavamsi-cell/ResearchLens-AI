"""
Test script generating a comparative matrix across real research papers,
checking comparability warnings and narrative synthesis.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from database.db_manager import DatabaseManager
from modules.paper_comparison import (
    build_comparison_matrix,
    check_comparability,
    generate_comparison_narrative,
    matrix_to_dataframe,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def run_comparison_test():
    settings = Settings()
    db = DatabaseManager(settings.database_path)

    with db.get_connection() as conn:
        papers = conn.execute(
            "SELECT id, filename, title FROM papers WHERE filename LIKE '%.pdf' AND filename NOT LIKE 'paper_%' ORDER BY created_at LIMIT 5"
        ).fetchall()

    if len(papers) < 2:
        print("Not enough real papers found in DB.")
        return

    paper_ids = [p["id"] for p in papers]
    print("\n=======================================================")
    print(f"GENERATING CROSS-PAPER COMPARATIVE MATRIX FOR {len(paper_ids)} REAL PAPERS")
    print("=======================================================\n")
    for i, p in enumerate(papers, 1):
        print(f"  [{i}] {p['title']} ({p['filename']})")

    # 1. Build comparison matrix
    matrix = build_comparison_matrix(paper_ids, db)
    df = matrix_to_dataframe(matrix)

    print("\n--- COMPARATIVE MATRIX (FIELDS x PAPERS) ---")
    print(df.to_string())

    # 2. Check comparability warning
    warning = check_comparability(matrix)
    print("\n--- COMPARABILITY CHECK RESULT ---")
    print(f"Has Warning: {warning.has_warning}")
    print(f"Message:     {warning.message}")
    print("Reasons:")
    for r in warning.reasons:
        print(f"  • {r}")

    # 3. Generate grounded comparison narrative
    print("\n--- SYNTHESIZING GROUNDED COMPARATIVE NARRATIVE ---")
    narrative = generate_comparison_narrative(matrix, warning)
    print(narrative[:600] + "..." if len(narrative) > 600 else narrative)

    # Save to disk
    out_file = Path("evaluation/real_papers_comparison.md")
    md_content = "# Real Papers Cross-Comparison Report\n\n"
    if warning.has_warning:
        md_content += f"> **{warning.message}**\n\n"
        for r in warning.reasons:
            md_content += f"- {r}\n"
        md_content += "\n---\n\n"

    md_content += "## 1. Comparison Matrix\n\n"
    md_content += df.to_markdown() + "\n\n"
    md_content += "## 2. Comparative Synthesis Narrative\n\n"
    md_content += narrative + "\n"

    out_file.write_text(md_content, encoding="utf-8")
    print(f"\n[DONE] Comparative analysis saved to {out_file}\n")


if __name__ == "__main__":
    run_comparison_test()
