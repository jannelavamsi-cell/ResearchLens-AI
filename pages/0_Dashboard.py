"""
ResearchLens AI - Workspace Dashboard & Executive Intelligence Hub.
Provides live corpus metrics, workflow step progression, recent activity,
and quick action routing across all research modules.
"""

import streamlit as st

from core.config import settings
from core.llm import get_llm_client
from database.db_manager import DBManager
from ui.components import render_footer, safe_page_render, stat_card
from ui.theme import inject_custom_css

db = DBManager(settings.database_path)
inject_custom_css()


@safe_page_render
def render_dashboard() -> None:
    current_col_id = st.session_state.get("active_collection_id", "")
    current_col_name = st.session_state.get("active_collection_name", "Default Workspace")

    # Fetch live database statistics for the active collection
    stats = db.get_collection_stats(current_col_id) if current_col_id else {
        "papers_count": 0, "pages_count": 0, "ready_papers": 0,
        "chunks_count": 0, "keywords_count": 0, "extractions_count": 0,
        "verified_extractions": 0, "unverified_extractions": 0,
        "tables_count": 0, "chat_questions": 0,
        "recent_papers": [], "recent_questions": [],
    }

    # 1. Hero Banner
    st.markdown(
        f"""
        <div class="rl-hero-banner">
            <div class="rl-hero-title">🔬 Welcome to ResearchLens AI</div>
            <p class="rl-hero-subtitle">
                Intelligent, evidence-grounded research paper analysis and comparative literature review platform.
                Active workspace: <b>{current_col_name}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. 5-Step Workflow Progression Strip
    papers_count = stats["papers_count"]
    extractions_count = stats["extractions_count"]
    chunks_count = stats["chunks_count"]

    step1_done = papers_count > 0
    step2_done = extractions_count > 0
    step3_done = chunks_count > 0
    step4_done = papers_count >= 2
    step5_done = extractions_count > 0 and papers_count >= 1

    st.markdown(
        f"""
        <div class="rl-workflow-strip">
            <div class="rl-workflow-step {'rl-workflow-step-done' if step1_done else ''}">
                <span>{'✅' if step1_done else '1️⃣'}</span>
                <span>1. Upload Papers</span>
            </div>
            <div class="rl-workflow-step {'rl-workflow-step-done' if step2_done else ''}">
                <span>{'✅' if step2_done else '2️⃣'}</span>
                <span>2. Structured Extraction</span>
            </div>
            <div class="rl-workflow-step {'rl-workflow-step-done' if step3_done else ''}">
                <span>{'✅' if step3_done else '3️⃣'}</span>
                <span>3. Semantic Indexing</span>
            </div>
            <div class="rl-workflow-step {'rl-workflow-step-done' if step4_done else ''}">
                <span>{'✅' if step4_done else '4️⃣'}</span>
                <span>4. Cross Comparison</span>
            </div>
            <div class="rl-workflow-step {'rl-workflow-step-done' if step5_done else ''}">
                <span>{'✅' if step5_done else '5️⃣'}</span>
                <span>5. Literature Review</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Live Corpus Stat Cards
    st.markdown("### 📊 Active Workspace Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(
            label="Research Papers",
            value=stats["papers_count"],
            sublabel=f"{stats['ready_papers']} fully processed",
            icon="📄",
        )
    with c2:
        stat_card(
            label="Total Pages",
            value=stats["pages_count"],
            sublabel=f"{stats['chunks_count']} section chunks",
            icon="📑",
        )
    with c3:
        stat_card(
            label="Extracted Claims",
            value=stats["extractions_count"],
            sublabel=f"🟢 {stats['verified_extractions']} verified • 🟡 {stats['unverified_extractions']} unverified",
            icon="🔍",
        )
    with c4:
        llm = get_llm_client()
        h = llm.health_check()
        h_model = h.get("model", "Active")
        stat_card(
            label="AI Intelligence",
            value=settings.llm_provider.upper(),
            sublabel=f"Model: {h_model}",
            icon="⚡",
        )

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        stat_card(
            label="Indexed Keywords",
            value=stats["keywords_count"],
            sublabel="KeyBERT & domain terms",
            icon="🏷️",
        )
    with c6:
        stat_card(
            label="Results Tables",
            value=stats["tables_count"],
            sublabel="Quantitative benchmark data",
            icon="📊",
        )
    with c7:
        stat_card(
            label="RAG Inquiries",
            value=stats["chat_questions"],
            sublabel="Grounded Q&A questions asked",
            icon="💬",
        )
    with c8:
        stat_card(
            label="Retrieval Fusion",
            value="Hybrid RRF",
            sublabel="FAISS + BM25 + Reranker",
            icon="🎯",
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 4. Recent Activity & Quick Navigation
    col_activity, col_quick = st.columns([1, 1])

    with col_activity:
        st.markdown("### 🕒 Recent Activity")
        with st.container(border=True):
            st.markdown("##### 📄 Recently Uploaded Papers")
            if stats["recent_papers"]:
                for p in stats["recent_papers"]:
                    p_title = p.get("title") or p.get("filename", "Untitled")
                    p_pages = p.get("n_pages", 0)
                    st.markdown(f"- **{p_title}** ({p_pages} pages) — *{p.get('status', 'ready').capitalize()}*")
            else:
                st.caption("No papers uploaded in this collection yet.")

            st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)
            st.markdown("##### 💬 Recent Chat Inquiries")
            if stats["recent_questions"]:
                for q in stats["recent_questions"]:
                    q_text = q.get("content", "")
                    if len(q_text) > 65:
                        q_text = q_text[:62] + "..."
                    st.markdown(f"- *\"{q_text}\"*")
            else:
                st.caption("No research chat questions asked yet.")

    with col_quick:
        st.markdown("### ⚡ Quick Workspace Actions")
        with st.container(border=True):
            st.markdown("##### Add Research Materials")
            st.write("Upload multi-page research PDFs to automatically parse sections, extract components, and build hybrid search indices.")
            if st.button("🚀 Go to PDF Upload", type="primary", use_container_width=True):
                st.switch_page("pages/1_Paper_Upload.py")

            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("🔍 Explore Structured Analysis", use_container_width=True):
                st.switch_page("pages/2_Paper_Analysis.py")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 5. All 8 Platform Module Action Cards with st.switch_page
    st.markdown("### 📑 Platform Capabilities & Tooling")

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)

    with row1_c1:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">📄 PDF Upload</div>
                    <div class="rl-module-desc">Two-column layout parsing, OCR fallback, section demarcation, and SHA-256 deduplication.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Upload", key="mod_upload", use_container_width=True):
            st.switch_page("pages/1_Paper_Upload.py")

    with row1_c2:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">🔍 Paper Analysis</div>
                    <div class="rl-module-desc">Extract 9 structured scientific components with verified quotes and page citations.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_analysis = papers_count == 0
        if st.button(
            "Open Analysis" if not dis_analysis else "Needs Upload",
            key="mod_analysis",
            disabled=dis_analysis,
            use_container_width=True,
        ):
            st.switch_page("pages/2_Paper_Analysis.py")

    with row1_c3:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">🏷️ Keyword Hub</div>
                    <div class="rl-module-desc">KeyBERT extraction with domain academic stopwords, TF-IDF ranking, and keyword clouds.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_kw = papers_count == 0
        if st.button(
            "Open Keywords" if not dis_kw else "Needs Upload",
            key="mod_keywords",
            disabled=dis_kw,
            use_container_width=True,
        ):
            st.switch_page("pages/2b_Keyword_Extraction.py")

    with row1_c4:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">⚡ Smart Search</div>
                    <div class="rl-module-desc">Hybrid retrieval fusing FAISS dense vectors and BM25 with Reciprocal Rank Fusion & Cross-Encoder.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_search = chunks_count == 0
        if st.button(
            "Open Search" if not dis_search else "Needs Indexing",
            key="mod_search",
            disabled=dis_search,
            use_container_width=True,
        ):
            st.switch_page("pages/3_Smart_Search.py")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)

    with row2_c1:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">💬 AI Assistant</div>
                    <div class="rl-module-desc">Conversational RAG answering with inline citations [Paper N, p. X] and claim verification.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_chat = chunks_count == 0
        if st.button(
            "Open Assistant" if not dis_chat else "Needs Indexing",
            key="mod_chat",
            disabled=dis_chat,
            use_container_width=True,
        ):
            st.switch_page("pages/4_Research_Chat.py")

    with row2_c2:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">⚖️ Paper Comparison</div>
                    <div class="rl-module-desc">Side-by-side matrices across 2-5 papers with benchmark comparability warning guards.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_comp = papers_count < 2
        if st.button(
            "Open Comparison" if not dis_comp else "Needs 2+ Papers",
            key="mod_comp",
            disabled=dis_comp,
            use_container_width=True,
        ):
            st.switch_page("pages/5_Paper_Comparison.py")

    with row2_c3:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">📖 Literature Review</div>
                    <div class="rl-module-desc">Synthesize thematic reviews with IEEE, APA, or BibTeX references and academic notice banner.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_review = papers_count == 0
        if st.button(
            "Open Review" if not dis_review else "Needs Papers",
            key="mod_review",
            disabled=dis_review,
            use_container_width=True,
        ):
            st.switch_page("pages/6_Literature_Review.py")

    with row2_c4:
        st.markdown(
            """
            <div class="rl-module-card">
                <div>
                    <div class="rl-module-title">🚀 Research Gaps</div>
                    <div class="rl-module-desc">Cluster author-declared limitations across papers to discover open thesis & research challenges.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dis_gaps = papers_count == 0
        if st.button(
            "Open Gaps" if not dis_gaps else "Needs Papers",
            key="mod_gaps",
            disabled=dis_gaps,
            use_container_width=True,
        ):
            st.switch_page("pages/7_Research_Gaps.py")

    render_footer()


render_dashboard()
