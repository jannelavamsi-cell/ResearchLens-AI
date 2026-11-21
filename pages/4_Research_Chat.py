"""
Streamlit Page 4: Research Chat & Grounded QA.
Conversational research question-answering with verifiable inline citations [Paper N, p. X],
claim verification drawer, unsupported sentence highlighting, and multi-turn query rewriting.
"""

import logging

import streamlit as st

from database.db_manager import DatabaseManager
from modules.rag_pipeline import answer_research_question

logger = logging.getLogger("researchlens.ui.chat")

st.set_page_config(
    page_title="Research Chat | ResearchLens AI",
    page_icon="💬",
    layout="wide",
)

db = DatabaseManager()

# Sidebar: Collections & Sessions
st.sidebar.title("💬 Research Chat")

collections = db.list_collections()
col_options = {c.get("name", "Default"): c.get("id", "") for c in collections}

if not collections:
    st.warning("No collections found. Please initialize a collection on the Home page.")
    st.stop()

selected_col_name = st.sidebar.selectbox("Workspace Collection", list(col_options.keys()))
active_col_id = col_options[selected_col_name]

papers = db.list_papers(active_col_id)
if not papers:
    st.info("No papers available in this collection.")
    st.markdown("👉 Head over to **Paper Upload** in the sidebar to upload research papers or load sample PDFs.")
    st.stop()

# Scope Filter
st.sidebar.subheader("🎯 Retrieval Scope")
scope_mode = st.sidebar.radio("Scope", ["All Papers in Collection", "Specific Papers Only"])
selected_paper_ids = None

if scope_mode == "Specific Papers Only":
    paper_map = {
        f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('n_pages', 0)}p)": p.get("id")
        for p in papers
    }
    selected_labels = st.sidebar.multiselect("Select Target Papers", list(paper_map.keys()), default=list(paper_map.keys())[:2])
    selected_paper_ids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]
    if not selected_paper_ids:
        st.sidebar.warning("Please select at least one paper.")

top_k = st.sidebar.slider("Evidence Passages (Top-K)", min_value=2, max_value=8, value=4)

# Chat Sessions Management
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Chat Sessions")

sessions = db.list_chat_sessions(active_col_id)
if not sessions:
    new_sess_id = db.create_chat_session(active_col_id, "Initial Research Chat")
    sessions = db.list_chat_sessions(active_col_id)

sess_options = {
    s.get("title", f"Session {idx+1}"): s.get("id", "")
    for idx, s in enumerate(sessions)
}
current_sess_title = list(sess_options.keys())[0]

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in sess_options.values():
    st.session_state.active_session_id = sess_options[current_sess_title]

# Find index of active session
current_index = 0
for idx, sid in enumerate(sess_options.values()):
    if sid == st.session_state.active_session_id:
        current_index = idx
        break

selected_sess_title = st.sidebar.selectbox("Active Session", list(sess_options.keys()), index=current_index)
st.session_state.active_session_id = sess_options[selected_sess_title]

new_chat_col1, new_chat_col2 = st.sidebar.columns(2)
with new_chat_col1:
    if st.button("➕ New Chat", use_container_width=True):
        new_id = db.create_chat_session(active_col_id, f"Chat #{len(sessions) + 1}")
        st.session_state.active_session_id = new_id
        st.rerun()

with new_chat_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        # Create fresh session
        fresh_id = db.create_chat_session(active_col_id, f"Chat #{len(sessions) + 1}")
        st.session_state.active_session_id = fresh_id
        st.rerun()

# Main Chat View
st.title("💬 Grounded Research Chat")
st.caption(
    "Ask comparative or in-depth technical questions. Every answer includes verifiable "
    "citations `[Paper N, p. X]` and automated claim verification."
)

# Load existing chat messages
chat_history_records = db.get_chat_messages(st.session_state.active_session_id)

# Render Chat History
for msg in chat_history_records:
    msg_role = msg.get("role", "user")
    msg_content = msg.get("content", "")
    with st.chat_message(msg_role):
        st.markdown(msg_content)

        if msg_role == "assistant":
            ver = msg.get("verification")
            cits = msg.get("citations")

            if ver:
                score = ver.get("overall_support_score", 1.0)
                score_pct = int(score * 100)
                badge_color = "🟢" if score_pct >= 75 else ("🟡" if score_pct >= 40 else "🔴")
                st.caption(f"{badge_color} **Evidence Support: {score_pct}%**")

                unsupported = ver.get("unsupported_claims", [])
                if unsupported:
                    with st.container(border=True):
                        st.markdown("⚠️ **Caution: Potential Unverified Claims**")
                        for u in unsupported:
                            st.caption(f"• *\"{u}\"*")

            if cits:
                with st.expander(f"📚 View {len(cits)} Verifiable Sources & Citations"):
                    for c_idx, cit in enumerate(cits, 1):
                        paper_name = cit.get("paper_title", "Unknown Paper")
                        page_num = cit.get("page_number", "?")
                        sec_name = cit.get("section_name", "General")
                        snippet_txt = cit.get("snippet", "")
                        st.markdown(f"**Source #{c_idx}: {paper_name}** | Page {page_num} | `{sec_name}`")
                        st.info(f'"{snippet_txt}"')

# Chat Input Box
user_query = st.chat_input("Ask a question about algorithms, results, datasets, or comparisons...")

if user_query:
    # Display user query immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Save user message to database
    db.insert_chat_message(
        session_id=st.session_state.active_session_id,
        role="user",
        content=user_query,
    )

    # Format history for multi-turn query rewriting
    history_for_rewrite = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in chat_history_records[-4:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence passages and synthesizing grounded answer..."):
            rag_result = answer_research_question(
                query=user_query,
                collection_id=active_col_id,
                paper_ids=selected_paper_ids,
                chat_history=history_for_rewrite,
                top_k=top_k,
                db=db,
            )

            st.markdown(rag_result.answer)

            # Verification badge
            score_pct = int(rag_result.overall_support_score * 100)
            badge_color = "🟢" if score_pct >= 75 else ("🟡" if score_pct >= 40 else "🔴")
            st.caption(f"{badge_color} **Evidence Support: {score_pct}%**")

            # Check unsupported claims
            unsupported_texts = [
                v.claim_text for v in rag_result.verifications
                if v.support_level == "unsupported"
            ]
            if unsupported_texts:
                with st.container(border=True):
                    st.markdown("⚠️ **Caution: Potential Unverified Claims**")
                    for u in unsupported_texts:
                        st.caption(f"• *\"{u}\"*")

            # Citations drawer
            if rag_result.citations:
                with st.expander(f"📚 View {len(rag_result.citations)} Verifiable Sources & Citations"):
                    for c_idx, cit in enumerate(rag_result.citations, 1):
                        st.markdown(f"**Source #{c_idx}: {cit.paper_title}** | Page {cit.page_number} | `{cit.section_name}`")
                        st.info(f'"{cit.snippet}"')

            # Save assistant message to database
            db.insert_chat_message(
                session_id=st.session_state.active_session_id,
                role="assistant",
                content=rag_result.answer,
                citations=[c.model_dump() for c in rag_result.citations],
                verification={
                    "overall_support_score": rag_result.overall_support_score,
                    "unsupported_claims": unsupported_texts,
                    "claims": [v.model_dump() for v in rag_result.verifications],
                },
            )
