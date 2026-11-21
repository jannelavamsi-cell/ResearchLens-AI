"""
ResearchLens AI - Dedicated Keyword Extraction & Topic Exploration Hub.
Provides per-paper KeyBERT keywords with scores, collection-wide keyword analytics,
topic co-occurrence, and semantic related-paper suggestions.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.config import settings
from database.db_manager import DatabaseManager
from modules.keyword_extraction import extract_paper_keywords
from ui.components import empty_state, keyword_chips, page_header, render_footer, safe_page_render
from ui.theme import inject_custom_css

db = DatabaseManager(settings.database_path)
inject_custom_css()


@safe_page_render
def render_keyword_hub() -> None:
    page_header(
        title="Keyword Intelligence & Topic Hub",
        subtitle="Explore KeyBERT and academic domain keywords across individual papers and whole collections.",
        icon="🏷️",
        badge="NLP Extraction",
    )

    current_col_id = st.session_state.get("active_collection_id", "")
    papers = db.list_papers(current_col_id) if current_col_id else []

    if not papers:
        empty_state(
            title="No Papers in Workspace",
            message="Upload research PDFs to automatically extract KeyBERT keywords and analyze topic distributions.",
            icon="📚",
            action_label="🚀 Go to PDF Upload",
            action_page="pages/1_Paper_Upload.py",
        )
        render_footer()
        return

    tab_paper, tab_collection, tab_related = st.tabs([
        "📄 Per-Paper Keywords",
        "🌐 Collection-Wide Keyword Table",
        "🔗 Semantic Paper Suggestions",
    ])

    with tab_paper:
        paper_map = {
            f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('n_pages', 0)}p)": p.get("id")
            for p in papers
        }
        selected_label = st.selectbox("Select Target Paper", list(paper_map.keys()), key="kw_paper_select")
        target_pid = paper_map[selected_label]

        kw_list = db.get_keywords(target_pid)

        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            st.markdown("##### Extracted Scientific Keywords & Scores")
            if kw_list:
                keyword_chips(kw_list, max_display=25)
            else:
                st.info("No keywords currently indexed for this paper.")

        with col_k2:
            st.markdown("##### Maintenance")
            if st.button("🔄 Re-extract Keywords", type="primary", use_container_width=True):
                with st.status("Re-extracting keywords...", expanded=True) as status:
                    st.write("Fetching paper text and running KeyBERT with academic stopword filtering...")
                    new_kw = extract_paper_keywords(target_pid, db=db, top_n=15)
                    st.write(f"Extracted {len(new_kw)} fresh keywords!")
                    status.update(label="Keywords updated successfully!", state="complete")
                st.rerun()

    with tab_collection:
        st.subheader("Collection-Wide Scientific Keyword Matrix")
        all_kw = db.get_collection_keywords(current_col_id)
        if all_kw:
            kw_counts: Dict[str, float] = {}
            kw_paper_counts: Dict[str, int] = {}
            for k in all_kw:
                word = str(k.get("keyword") or "").lower().strip()
                score = float(k.get("score", 1.0))
                if word:
                    kw_counts[word] = kw_counts.get(word, 0.0) + score
                    kw_paper_counts[word] = kw_paper_counts.get(word, 0) + 1

            rows = [
                {
                    "Keyword / Topic": word,
                    "Papers Mentioning": kw_paper_counts[word],
                    "Aggregated Relevance Score": round(kw_counts[word], 3),
                }
                for word in kw_counts
            ]
            rows.sort(key=lambda x: x["Aggregated Relevance Score"], reverse=True)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No keywords detected across this collection yet.")

    with tab_related:
        st.subheader("Shared Topic Clusters & Related Papers")
        if len(papers) < 2:
            st.info("Upload at least 2 papers in this collection to discover cross-paper semantic relationships.")
        else:
            # Compute Jaccard overlap on keywords between papers
            paper_kws: Dict[str, set] = {}
            paper_titles: Dict[str, str] = {}
            for p in papers:
                pid = p["id"]
                paper_titles[pid] = p.get("title") or p.get("filename", "Untitled")
                kws = {str(k.get("keyword", "")).lower().strip() for k in db.get_keywords(pid)}
                paper_kws[pid] = kws

            pids = list(paper_kws.keys())
            found_relations: List[Dict[str, Any]] = []
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    p1, p2 = pids[i], pids[j]
                    shared = paper_kws[p1].intersection(paper_kws[p2])
                    if shared:
                        found_relations.append({
                            "Paper A": paper_titles[p1],
                            "Paper B": paper_titles[p2],
                            "Shared Topics": ", ".join(sorted(shared)),
                            "Shared Count": len(shared),
                        })

            if found_relations:
                found_relations.sort(key=lambda x: x["Shared Count"], reverse=True)
                st.dataframe(pd.DataFrame(found_relations), use_container_width=True, hide_index=True)
            else:
                st.info("Papers in this collection explore distinct specialized sub-topics with no overlapping keywords yet.")

    render_footer()


render_keyword_hub()
