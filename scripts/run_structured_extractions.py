"""
Run structured extraction across all 5 real research papers using Gemini LLM,
verify quotes against page text, and print structured output report.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from database.db_manager import DatabaseManager
from modules.research_extraction import extract_paper_components

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("run_extractions")


def main():
    settings = Settings()
    db = DatabaseManager(settings.database_path)

    with db.get_connection() as conn:
        papers = conn.execute(
            "SELECT id, filename, title FROM papers WHERE collection_id = 'default-collection' ORDER BY created_at"
        ).fetchall()

    print("\n=======================================================")
    print(f"RUNNING STRUCTURED EXTRACTION ON {len(papers)} REAL PAPERS")
    print(f"LLM Provider: {settings.llm_provider} | Model: {settings.gemini_model}")
    print("=======================================================\n")

    all_results = {}

    for idx, p in enumerate(papers, 1):
        paper_id = p["id"]
        filename = p["filename"]
        title = p["title"]
        print("\n-------------------------------------------------------")
        print(f"[{idx}/{len(papers)}] {filename}")
        print(f"Title: {title}")
        print("-------------------------------------------------------")

        extractions = extract_paper_components(paper_id, db, force_refresh=True)

        paper_report = {}
        for f_name, field in extractions.items():
            status_symbol = "🟢 Verified" if field.verified else "🟡 Unverified"
            if field.value == "Not stated in paper":
                status_symbol = "⚪ Not Stated"

            print(f"  • {f_name.upper()}:")
            print(f"      Value: {field.value[:100]}..." if len(field.value) > 100 else f"      Value: {field.value}")
            print(f"      Page: p. {field.page} | Status: {status_symbol} | Confidence: {field.confidence:.2f}")
            if field.evidence_quote:
                quote_preview = field.evidence_quote[:80] + "..." if len(field.evidence_quote) > 80 else field.evidence_quote
                print(f"      Quote: \"{quote_preview}\"")

            paper_report[f_name] = {
                "value": field.value,
                "page": field.page,
                "evidence_quote": field.evidence_quote,
                "verified": field.verified,
                "confidence": field.confidence,
                "source_section": field.source_section,
            }

        all_results[filename] = {
            "title": title,
            "paper_id": paper_id,
            "fields": paper_report,
        }

    out_file = Path("evaluation/real_papers_extractions.json")
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[DONE] Extractions stored to database and saved to {out_file}\n")


if __name__ == "__main__":
    main()
