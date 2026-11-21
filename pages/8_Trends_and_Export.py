"""
ResearchLens AI - Research Trends & Multi-Format Export Center.
Provides interactive visual analytics (keyword distributions, publication timelines, similarity heatmaps)
and a unified export engine compiling documents into Markdown, DOCX, PDF, and BibTeX.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.embeddings import get_embedding_model
from modules.export import (
    export_bibtex_file,
    export_to_docx,
    export_to_pdf,
)
from modules.literature_review import review_to_markdown, synthesize_literature_review
from modules.paper_comparison import (
    build_comparison_matrix,
    check_comparability,
    generate_comparison_narrative,
    matrix_to_dataframe,
    matrix_to_markdown,
)
from modules.research_gap import discover_research_gaps
from ui.components import (
    empty_state,
    page_header,
    safe_page_render,
    stat_card,
)
from ui.theme import get_plotly_template, inject_custom_css

logger = logging.getLogger("researchlens.ui.trends")
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="Trends & Export | ResearchLens AI",
    page_icon="📈",
    layout="wide",
)
inject_custom_css()


@safe_page_render("Trends & Export Center")
def main():
    collections = db.list_collections()
    if not collections:
        empty_state("No Collections Found", "Create a workspace collection from the library first.", icon="📚")
        return

    col_map = {c["name"]: c["id"] for c in collections}
    if "active_collection_id" not in st.session_state or st.session_state["active_collection_id"] not in col_map.values():
        st.session_state["active_collection_id"] = collections[0]["id"]
        st.session_state["active_collection_name"] = collections[0]["name"]

    active_col_id = st.session_state["active_collection_id"]
    active_col_name = st.session_state["active_collection_name"]

    papers = db.list_papers(active_col_id)

    page_header(
        title="Research Trends & Document Export",
        subtitle=f"Corpus analytics, thematic distributions, and multi-format document compilation for {active_col_name}",
        badge_text="Visual Analytics + Export Hub",
    )

    if not papers:
        empty_state(
            title="No Research Papers Available",
            message=f"The active workspace '{active_col_name}' contains no papers. Upload papers to view corpus analytics and export reports.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="📊",
        )
        return

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📈 Analytics & Export")
        st.caption(f"Workspace: **{active_col_name}** ({len(papers)} papers)")

    # -----------------------------------------------------------------------
    # Top KPI Bar
    # -----------------------------------------------------------------------
    kw_records = db.get_collection_keywords(active_col_id)
    years = [p.get("year") for p in papers if p.get("year")]
    year_range_str = f"{min(years)} - {max(years)}" if years else "N/A"

    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    with kpi_c1:
        stat_card("Papers in Corpus", str(len(papers)), delta="Indexed in collection", icon="📚")
    with kpi_c2:
        stat_card("Extracted Key Terms", str(len(kw_records)), delta="Domain terminology", icon="🏷️")
    with kpi_c3:
        stat_card("Publication Span", year_range_str, delta=f"{len(years)} dated papers", icon="📅")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Main Tabs: Analytics vs Unified Export
    # -----------------------------------------------------------------------
    tab_analytics, tab_export = st.tabs([
        "📊 Corpus Intelligence & Visual Analytics",
        "📥 Unified Document Export Center",
    ])

    plotly_tpl = get_plotly_template()

    # -----------------------------------------------------------------------
    # Tab 1: Visual Analytics
    # -----------------------------------------------------------------------
    with tab_analytics:
        col_chart1, col_chart2 = st.columns(2)

        # 1. Top Research Topics & Keywords
        with col_chart1:
            st.markdown("##### 🏷️ Top Research Topics & Keyword Frequency")
            if kw_records:
                kw_counts: Dict[str, float] = {}
                for k in kw_records:
                    term = str(k.get("keyword") or "").lower().strip()
                    if term:
                        kw_counts[term] = kw_counts.get(term, 0) + float(k.get("score", 1.0))
                sorted_kw = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:12]
                df_kw = pd.DataFrame(sorted_kw, columns=["Keyword / Topic", "Aggregated Score"])
                fig_kw = px.bar(
                    df_kw,
                    x="Aggregated Score",
                    y="Keyword / Topic",
                    orientation="h",
                    color="Aggregated Score",
                    color_continuous_scale="Purples",
                    template=plotly_tpl,
                )
                fig_kw.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No keywords extracted yet. Process papers on the Paper Analysis page.")

        # 2. Publication Timeline
        with col_chart2:
            st.markdown("##### 📅 Publication Year Distribution")
            if years:
                df_years = pd.DataFrame(years, columns=["Year"]).value_counts().reset_index()
                df_years.columns = ["Year", "Paper Count"]
                df_years = df_years.sort_values(by="Year")
                fig_years = px.bar(
                    df_years,
                    x="Year",
                    y="Paper Count",
                    text="Paper Count",
                    color_discrete_sequence=["#4F46E5"],
                    template=plotly_tpl,
                )
                fig_years.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_years, use_container_width=True)
            else:
                st.info("Publication years not detected in current document metadata.")

        st.markdown("---")

        # 3. Cross-Paper Semantic Similarity Heatmap
        st.markdown("##### 🧬 Cross-Paper Semantic Similarity Heatmap")
        st.caption("Cosine similarity of paper embeddings reflecting conceptual alignment and thematic overlap.")

        if len(papers) >= 2:
            corpus = [f"{p.get('title') or p.get('filename', 'Paper')}. {p.get('authors', '')}" for p in papers]
            titles = [p.get("title") or p.get("filename", "Paper") for p in papers]
            short_titles = [t[:24] + ("..." if len(t) > 24 else "") for t in titles]

            try:
                model = get_embedding_model()
                embeddings = model.encode(corpus, normalize_embeddings=True)
                sim_matrix = np.dot(embeddings, embeddings.T)
                sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

                fig_hm = go.Figure(
                    data=go.Heatmap(
                        z=sim_matrix,
                        x=short_titles,
                        y=short_titles,
                        colorscale="Blues",
                        zmin=0.2,
                        zmax=1.0,
                        text=np.round(sim_matrix, 2),
                        texttemplate="%{text}",
                    )
                )
                fig_hm.update_layout(height=460, margin=dict(l=40, r=40, t=20, b=40), template=plotly_tpl)
                st.plotly_chart(fig_hm, use_container_width=True)
            except Exception as err:
                st.warning(f"Could not compute semantic similarity matrix: {err}")
        else:
            st.info("Upload at least 2 papers to view the similarity heatmap.")

    # -----------------------------------------------------------------------
    # Tab 2: Unified Export Center
    # -----------------------------------------------------------------------
    with tab_export:
        st.markdown("### 📥 Publication-Ready Document Compilation")
        st.caption("Select an academic artifact and compile it directly into Markdown, Microsoft Word (.docx), Adobe PDF, or BibTeX.")

        # Export Grid: 4 Cards
        card_col1, card_col2 = st.columns(2)

        # 1. Literature Review Export Card
        with card_col1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.3rem;">📖</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: var(--rl-primary);">Literature Review Draft</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--rl-text-secondary); margin-bottom: 0.75rem;">
                        Thematic synthesis, methodological comparison, open research gaps, and formatted references.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                lr_format = st.selectbox("Format", ["Markdown (.md)", "Microsoft Word (.docx)", "Adobe PDF (.pdf)"], key="export_lr_fmt")
                if st.button("Compile Literature Review", key="btn_compile_lr", type="primary", use_container_width=True):
                    with st.spinner("Synthesizing and compiling literature review..."):
                        draft = synthesize_literature_review(collection_id=active_col_id, db=db)
                        if "Markdown" in lr_format:
                            md_bytes = review_to_markdown(draft).encode("utf-8")
                            st.download_button("📥 Download .md", data=md_bytes, file_name=f"literature_review_{active_col_name[:8]}.md", mime="text/markdown", use_container_width=True)
                        elif "Word" in lr_format:
                            secs = [
                                {"title": "1. Introduction", "content": draft.introduction},
                                {"title": "2. Methodological Comparison", "content": draft.methodological_comparison},
                                {"title": "3. Research Gaps & Open Challenges", "content": draft.research_gaps},
                                {"title": "4. Conclusion", "content": draft.conclusion},
                            ]
                            buf = export_to_docx(title=draft.title, sections=secs, references=draft.references, disclaimer=draft.disclaimer)
                            st.download_button("📥 Download .docx", data=buf.getvalue(), file_name=f"literature_review_{active_col_name[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
                        elif "PDF" in lr_format:
                            secs = [
                                {"title": "1. Introduction", "content": draft.introduction},
                                {"title": "2. Methodological Comparison", "content": draft.methodological_comparison},
                                {"title": "3. Research Gaps & Open Challenges", "content": draft.research_gaps},
                                {"title": "4. Conclusion", "content": draft.conclusion},
                            ]
                            buf = export_to_pdf(title=draft.title, sections=secs, references=draft.references, disclaimer=draft.disclaimer)
                            st.download_button("📥 Download .pdf", data=buf.getvalue(), file_name=f"literature_review_{active_col_name[:8]}.pdf", mime="application/pdf", use_container_width=True)
                        st.toast("Literature review compiled successfully!", icon="✅")

        # 2. Paper Comparison Matrix Export Card
        with card_col2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.3rem;">⚖️</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: var(--rl-primary);">Cross-Paper Comparison Matrix</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--rl-text-secondary); margin-bottom: 0.75rem;">
                        Side-by-side component matrix, comparability verification, and comparative synthesis narrative.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cmp_format = st.selectbox("Format", ["Markdown (.md)", "Microsoft Word (.docx)", "Adobe PDF (.pdf)", "CSV Table (.csv)"], key="export_cmp_fmt")
                if st.button("Compile Comparison Report", key="btn_compile_cmp", type="primary", use_container_width=True):
                    with st.spinner("Building comparison report..."):
                        pids = [p["id"] for p in papers[:5] if p.get("id")]
                        matrix = build_comparison_matrix(pids, db)
                        warning = check_comparability(matrix)
                        narrative = generate_comparison_narrative(matrix, warning)

                        if "Markdown" in cmp_format:
                            md_bytes = matrix_to_markdown(matrix, warning, narrative).encode("utf-8")
                            st.download_button("📥 Download .md", data=md_bytes, file_name=f"paper_comparison_{active_col_name[:8]}.md", mime="text/markdown", use_container_width=True)
                        elif "CSV" in cmp_format:
                            df = matrix_to_dataframe(matrix)
                            st.download_button("📥 Download .csv", data=df.to_csv().encode("utf-8"), file_name=f"paper_comparison_{active_col_name[:8]}.csv", mime="text/csv", use_container_width=True)
                        elif "Word" in cmp_format:
                            df = matrix_to_dataframe(matrix)
                            table_rows = [["Component"] + list(df.columns)]
                            for idx_val, row in df.iterrows():
                                table_rows.append([str(idx_val)] + [str(v) for v in row.values])
                            secs = [{"title": "Comparative Synthesis Narrative", "content": narrative}]
                            if warning.has_warning:
                                secs.insert(0, {"title": "Comparability Warning", "content": warning.message})
                            buf = export_to_docx(title="Cross-Paper Comparative Analysis", sections=secs, table_data=table_rows, disclaimer="Direct ranking across distinct experimental benchmarks is scientifically invalid.")
                            st.download_button("📥 Download .docx", data=buf.getvalue(), file_name=f"paper_comparison_{active_col_name[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
                        elif "PDF" in cmp_format:
                            secs = [{"title": "Comparative Synthesis Narrative", "content": narrative}]
                            if warning.has_warning:
                                secs.insert(0, {"title": "Comparability Warning", "content": warning.message})
                            buf = export_to_pdf(title="Cross-Paper Comparative Analysis", sections=secs, disclaimer="Direct ranking across distinct experimental benchmarks is scientifically invalid.")
                            st.download_button("📥 Download .pdf", data=buf.getvalue(), file_name=f"paper_comparison_{active_col_name[:8]}.pdf", mime="application/pdf", use_container_width=True)
                        st.toast("Comparison report compiled successfully!", icon="✅")

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        card_col3, card_col4 = st.columns(2)

        # 3. Research Gaps Export Card
        with card_col3:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.3rem;">🚀</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: var(--rl-primary);">Research Gaps & Proposals</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--rl-text-secondary); margin-bottom: 0.75rem;">
                        Synthesized limitation catalog, challenge recurrence frequencies, and thesis proposal abstracts.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                gap_format = st.selectbox("Format", ["Markdown (.md)", "Microsoft Word (.docx)", "Adobe PDF (.pdf)"], key="export_gap_fmt")
                if st.button("Compile Gaps Report", key="btn_compile_gap", type="primary", use_container_width=True):
                    with st.spinner("Compiling research gaps catalog..."):
                        gap_items = discover_research_gaps(active_col_id, db=db)
                        secs = []
                        for idx_g, g in enumerate(gap_items, 1):
                            sec_text = (
                                f"Frequency: Observed in {g.frequency_count} paper(s).\n\n"
                                f"Actionable Direction: {g.suggested_direction}\n\n"
                                f"Rationale: {g.rationale}\n\n"
                                f"Supporting Papers:\n"
                            )
                            for sp in g.supporting_papers:
                                sp_title = sp.get("paper_title", "Unknown Paper")
                                sp_page = sp.get("page", "?")
                                sp_quote = sp.get("limitation_quote", "")
                                sec_text += f"- {sp_title} (p. {sp_page}): \"{sp_quote}\"\n"
                            secs.append({"title": f"Gap #{idx_g}: {g.gap_statement}", "content": sec_text})

                        if "Markdown" in gap_format:
                            md_lines = ["# Grounded Research Gaps Analysis\n"]
                            for s in secs:
                                md_lines.append(f"## {s['title']}\n{s['content']}\n---\n")
                            st.download_button("📥 Download .md", data="\n".join(md_lines).encode("utf-8"), file_name=f"research_gaps_{active_col_name[:8]}.md", mime="text/markdown", use_container_width=True)
                        elif "Word" in gap_format:
                            buf = export_to_docx(title="Grounded Research Gaps & Opportunities Report", sections=secs)
                            st.download_button("📥 Download .docx", data=buf.getvalue(), file_name=f"research_gaps_{active_col_name[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
                        elif "PDF" in gap_format:
                            buf = export_to_pdf(title="Grounded Research Gaps & Opportunities Report", sections=secs)
                            st.download_button("📥 Download .pdf", data=buf.getvalue(), file_name=f"research_gaps_{active_col_name[:8]}.pdf", mime="application/pdf", use_container_width=True)
                        st.toast("Research gaps report compiled successfully!", icon="✅")

        # 4. Collection BibTeX Export Card
        with card_col4:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.3rem;">📚</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: var(--rl-primary);">Complete Collection BibTeX</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--rl-text-secondary); margin-bottom: 0.75rem;">
                        Standardized BibTeX citations for all papers currently in the active workspace.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.selectbox("Format", ["BibTeX (.bib)"], disabled=True, key="export_bib_fmt")
                if st.button("Generate BibTeX File", key="btn_compile_bib", type="primary", use_container_width=True):
                    bib_content = export_bibtex_file(papers)
                    st.download_button(
                        label="📥 Download .bib",
                        data=bib_content.encode("utf-8"),
                        file_name=f"collection_{active_col_name[:8]}.bib",
                        mime="text/plain",
                        use_container_width=True,
                    )
                    st.toast("BibTeX file generated successfully!", icon="✅")


if __name__ == "__main__":
    main()
