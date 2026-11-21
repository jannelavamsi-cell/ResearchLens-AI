"""
Streamlit Page 8: Research Trends & Multi-Format Export.
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

from database.db_manager import DatabaseManager
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

logger = logging.getLogger("researchlens.ui.trends")

st.set_page_config(
    page_title="Trends & Export | ResearchLens AI",
    page_icon="📈",
    layout="wide",
)

db = DatabaseManager()

st.sidebar.title("📈 Trends & Export")

collections = db.list_collections()
col_options = {c.get("name", "Default"): c.get("id", "") for c in collections}

if not collections:
    st.warning("No collections found.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Workspace Collection", list(col_options.keys()))
active_col_id = col_options[selected_col_name]

papers = db.list_papers(active_col_id)
if not papers:
    st.info("No papers available in this collection.")
    st.markdown("👉 Head over to **Paper Upload** in the sidebar to upload research papers or load sample PDFs.")
    st.stop()

st.title("📈 Research Trends & Unified Export")
st.caption(
    "Corpus-wide keyword frequency, temporal distributions, embedding similarity heatmaps, "
    "and publication-ready document compilation."
)

tab_analytics, tab_export = st.tabs([
    "📊 Corpus Intelligence & Visual Analytics",
    "📥 Unified Document Export Center",
])

with tab_analytics:
    st.subheader("Corpus-Wide Research Intelligence")

    col_chart1, col_chart2 = st.columns(2)

    # 1. Keyword Frequency Analysis
    with col_chart1:
        st.markdown("##### 🏷️ Top Research Topics & Keywords")
        kw_records = db.get_collection_keywords(active_col_id)
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
                color_continuous_scale="Blues",
            )
            fig_kw.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Keywords have not yet been extracted for papers in this collection.")

    # 2. Publication Timeline
    with col_chart2:
        st.markdown("##### 📅 Publication Year Distribution")
        years = [p.get("year") for p in papers if p.get("year")]
        if years:
            df_years = pd.DataFrame(years, columns=["Year"]).value_counts().reset_index()
            df_years.columns = ["Year", "Paper Count"]
            df_years = df_years.sort_values(by="Year")
            fig_years = px.bar(
                df_years,
                x="Year",
                y="Paper Count",
                text="Paper Count",
                color_discrete_sequence=["#3182CE"],
            )
            fig_years.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_years, use_container_width=True)
        else:
            st.info("Publication years not detected in current document metadata.")

    st.markdown("---")

    # 3. Paper Semantic Similarity Heatmap
    st.markdown("##### 🧬 Cross-Paper Semantic Similarity Heatmap")
    st.caption("Cosine similarity of paper embeddings reflecting thematic convergence.")

    if len(papers) >= 2:
        corpus = [f"{p.get('title') or p.get('filename', 'Paper')}. {p.get('authors', '')}" for p in papers]
        titles = [p.get("title") or p.get("filename", "Paper") for p in papers]
        # Shorten titles for axes
        short_titles = [t[:28] + ("..." if len(t) > 28 else "") for t in titles]

        try:
            model = get_embedding_model()
            embeddings = model.encode(corpus, normalize_embeddings=True)
            # Pairwise cosine similarity matrix
            sim_matrix = np.dot(embeddings, embeddings.T)
            sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=sim_matrix,
                    x=short_titles,
                    y=short_titles,
                    colorscale="YlGnBu",
                    zmin=0.2,
                    zmax=1.0,
                    text=np.round(sim_matrix, 2),
                    texttemplate="%{text}",
                )
            )
            fig_hm.update_layout(height=450, margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as err:
            st.warning(f"Could not compute semantic similarity matrix: {err}")
    else:
        st.info("Upload at least 2 papers to view the similarity heatmap.")

with tab_export:
    st.subheader("Unified Document Compilation & Export")
    st.caption("Generate publication-ready academic artifacts in Markdown, Word (.docx), PDF, or BibTeX.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        target_doc = st.selectbox(
            "Select Artifact to Export",
            [
                "Literature Review Draft",
                "Cross-Paper Comparison Matrix",
                "Research Gaps & Opportunities Report",
                "Complete Collection BibTeX File",
            ],
        )

    with exp_col2:
        if target_doc == "Complete Collection BibTeX File":
            export_format = st.selectbox("Export Format", ["BibTeX (.bib)"])
        else:
            export_format = st.selectbox("Export Format", ["Markdown (.md)", "Microsoft Word (.docx)", "Adobe PDF (.pdf)"])

    st.markdown("---")

    if st.button("🚀 Compile & Export Artifact", type="primary"):
        with st.spinner("Compiling document artifact..."):
            if target_doc == "Complete Collection BibTeX File":
                bib_content = export_bibtex_file(papers)
                st.download_button(
                    label="📥 Download BibTeX Bibliography (.bib)",
                    data=bib_content.encode("utf-8"),
                    file_name="collection_bibliography.bib",
                    mime="text/plain",
                )
                st.success("BibTeX bibliography ready for download!")

            elif target_doc == "Literature Review Draft":
                draft = synthesize_literature_review(collection_id=active_col_id, db=db)
                doc_title = draft.title

                if "Markdown" in export_format:
                    md_text = review_to_markdown(draft)
                    st.download_button(
                        label="📥 Download Literature Review (.md)",
                        data=md_text.encode("utf-8"),
                        file_name="literature_review.md",
                        mime="text/markdown",
                    )
                elif "Word" in export_format:
                    sections = [
                        {"title": "1. Introduction", "content": draft.introduction},
                        {"title": "2. Methodological Comparison", "content": draft.methodological_comparison},
                        {"title": "3. Research Gaps & Open Challenges", "content": draft.research_gaps},
                        {"title": "4. Conclusion", "content": draft.conclusion},
                    ]
                    docx_buf = export_to_docx(
                        title=doc_title,
                        sections=sections,
                        references=draft.references,
                        disclaimer=draft.disclaimer,
                    )
                    st.download_button(
                        label="📥 Download Literature Review (.docx)",
                        data=docx_buf.getvalue(),
                        file_name="literature_review.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                elif "PDF" in export_format:
                    sections = [
                        {"title": "1. Introduction", "content": draft.introduction},
                        {"title": "2. Methodological Comparison", "content": draft.methodological_comparison},
                        {"title": "3. Research Gaps & Open Challenges", "content": draft.research_gaps},
                        {"title": "4. Conclusion", "content": draft.conclusion},
                    ]
                    pdf_buf = export_to_pdf(
                        title=doc_title,
                        sections=sections,
                        references=draft.references,
                        disclaimer=draft.disclaimer,
                    )
                    st.download_button(
                        label="📥 Download Literature Review (.pdf)",
                        data=pdf_buf.getvalue(),
                        file_name="literature_review.pdf",
                        mime="application/pdf",
                    )
                st.success(f"Successfully compiled {target_doc} in {export_format}!")

            elif target_doc == "Cross-Paper Comparison Matrix":
                pids = [p["id"] for p in papers[:5] if p.get("id")]
                matrix = build_comparison_matrix(pids, db)
                warning = check_comparability(matrix)
                narrative = generate_comparison_narrative(matrix, warning)

                if "Markdown" in export_format:
                    md_text = matrix_to_markdown(matrix, warning, narrative)
                    st.download_button(
                        label="📥 Download Comparison Report (.md)",
                        data=md_text.encode("utf-8"),
                        file_name="paper_comparison.md",
                        mime="text/markdown",
                    )
                elif "Word" in export_format:
                    df = matrix_to_dataframe(matrix)
                    table_rows = [["Component"] + list(df.columns)]
                    for idx, row in df.iterrows():
                        table_rows.append([str(idx)] + [str(v) for v in row.values])

                    sections = [
                        {"title": "Comparative Synthesis Narrative", "content": narrative},
                    ]
                    if warning.has_warning:
                        sections.insert(0, {"title": "Comparability Warning", "content": warning.message})

                    docx_buf = export_to_docx(
                        title="Cross-Paper Comparative Analysis",
                        sections=sections,
                        table_data=table_rows,
                        disclaimer="Direct ranking across distinct experimental benchmarks is scientifically invalid.",
                    )
                    st.download_button(
                        label="📥 Download Comparison Report (.docx)",
                        data=docx_buf.getvalue(),
                        file_name="paper_comparison.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                elif "PDF" in export_format:
                    sections = [
                        {"title": "Comparative Synthesis Narrative", "content": narrative},
                    ]
                    if warning.has_warning:
                        sections.insert(0, {"title": "Comparability Warning", "content": warning.message})

                    pdf_buf = export_to_pdf(
                        title="Cross-Paper Comparative Analysis",
                        sections=sections,
                        disclaimer="Direct ranking across distinct experimental benchmarks is scientifically invalid.",
                    )
                    st.download_button(
                        label="📥 Download Comparison Report (.pdf)",
                        data=pdf_buf.getvalue(),
                        file_name="paper_comparison.pdf",
                        mime="application/pdf",
                    )
                st.success(f"Successfully compiled {target_doc} in {export_format}!")

            elif target_doc == "Research Gaps & Opportunities Report":
                gap_items = discover_research_gaps(active_col_id, db=db)
                sections = []
                for idx, g in enumerate(gap_items, 1):
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
                    sections.append({"title": f"Gap #{idx}: {g.gap_statement}", "content": sec_text})

                if "Markdown" in export_format:
                    md_lines = ["# Grounded Research Gaps Analysis\n"]
                    for s in sections:
                        s_title = s.get("title", "Section")
                        s_content = s.get("content", "")
                        md_lines.append(f"## {s_title}\n{s_content}\n---\n")
                    st.download_button(
                        label="📥 Download Gaps Report (.md)",
                        data="\n".join(md_lines).encode("utf-8"),
                        file_name="research_gaps_report.md",
                        mime="text/markdown",
                    )
                elif "Word" in export_format:
                    docx_buf = export_to_docx(
                        title="Grounded Research Gaps & Opportunities Report",
                        sections=sections,
                    )
                    st.download_button(
                        label="📥 Download Gaps Report (.docx)",
                        data=docx_buf.getvalue(),
                        file_name="research_gaps_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                elif "PDF" in export_format:
                    pdf_buf = export_to_pdf(
                        title="Grounded Research Gaps & Opportunities Report",
                        sections=sections,
                    )
                    st.download_button(
                        label="📥 Download Gaps Report (.pdf)",
                        data=pdf_buf.getvalue(),
                        file_name="research_gaps_report.pdf",
                        mime="application/pdf",
                    )
                st.success(f"Successfully compiled {target_doc} in {export_format}!")
