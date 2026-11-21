"""
ResearchLens AI - Research Chat & Grounded QA Assistant.
Conversational research question-answering with verifiable inline citations [Paper N, p. X],
claim verification drawer, unsupported sentence highlighting, and multi-turn query rewriting.
"""

import html
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from core.config import settings
from database.db_manager import DBManager
from modules.rag_pipeline import answer_research_question
from ui.components import (
    banner,
    citation_pill,
    empty_state,
    page_header,
    safe_page_render,
    status_badge,
)
from ui.theme import inject_custom_css

logger = logging.getLogger("researchlens.ui.chat")
db = DBManager(settings.database_path)

st.set_page_config(
    page_title="Research Chat Assistant | ResearchLens AI",
    page_icon="💬",
    layout="wide",
)
inject_custom_css()


@st.dialog("Clear Chat Session")
def confirm_clear_session(session_id: str, col_id: str):
    """Confirmation modal dialog before clearing a chat session."""
    st.warning("Are you sure you want to clear this chat session? All message history will be removed.")
    col_del, col_cancel = st.columns(2)
    with col_del:
        if st.button("Yes, Clear History", type="primary", use_container_width=True):
            fresh_id = db.create_chat_session(col_id, f"Research Chat {datetime.now().strftime('%H:%M')}")
            st.session_state["active_session_id"] = fresh_id
            st.toast("Chat session reset.", icon="🗑️")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def format_transcript_markdown(session_title: str, collection_name: str, messages: List[Dict[str, Any]]) -> str:
    """Formats the conversation history into an academic-grade Markdown export."""
    lines = [
        "# ResearchLens AI - Conversation Transcript",
        f"**Session:** {session_title}  ",
        f"**Workspace Collection:** {collection_name}  ",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Messages:** {len(messages)}  ",
        "\n---\n",
    ]

    for idx, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "").strip()
        lines.append(f"### {idx}. {role}\n")
        lines.append(f"{content}\n")

        ver = msg.get("verification")
        if ver and isinstance(ver, dict):
            score = ver.get("overall_support_score", 1.0)
            lines.append(f"> **Evidence Grounding Score:** {int(score * 100)}%\n")
            unsupported = ver.get("unsupported_claims", [])
            if unsupported:
                lines.append("> ⚠️ **Flagged Unsupported Statements:**")
                for u in unsupported:
                    lines.append(f"> - *\"{u}\"*")
                lines.append("")

        cits = msg.get("citations")
        if cits and isinstance(cits, list):
            lines.append("#### Sources & Citations:\n")
            for c in cits:
                ptitle = c.get("paper_title", "Unknown")
                pnum = c.get("page_number", "?")
                sname = c.get("section_name", "General")
                snip = c.get("snippet", "").strip()
                lines.append(f"- **{ptitle}** (p. {pnum}, `{sname}`): *\"{snip}\"*")
            lines.append("")

        lines.append("\n---\n")

    return "\n".join(lines)


@safe_page_render("Research Chat Assistant")
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

    # -----------------------------------------------------------------------
    # Sidebar: Scope, Sessions, Export
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 💬 Assistant Settings")
        st.caption(f"Workspace: **{active_col_name}** ({len(papers)} papers)")

        if not papers:
            st.warning("No papers in this workspace. Upload papers to enable grounded chat.")
            return

        st.markdown("#### 🎯 Retrieval Scope")
        scope_mode = st.radio("Target Scope", ["All Workspace Papers", "Specific Papers Only"], label_visibility="collapsed")
        selected_paper_ids: Optional[List[str]] = None

        if scope_mode == "Specific Papers Only":
            paper_map = {
                f"{p.get('title') or p.get('filename', 'Unknown')} ({p.get('n_pages', 0)}p)": p.get("id")
                for p in papers
            }
            default_selection = list(paper_map.keys())[: min(2, len(paper_map))]
            selected_labels = st.multiselect("Select Target Papers", list(paper_map.keys()), default=default_selection)
            selected_paper_ids = [paper_map[lbl] for lbl in selected_labels if lbl in paper_map]
            if not selected_paper_ids:
                st.warning("Select at least one paper for retrieval.")

        top_k = st.slider("Evidence Passages (Top-K)", min_value=2, max_value=8, value=4, step=1)

        st.markdown("---")
        st.markdown("#### 📁 Conversation Sessions")

        sessions = db.list_chat_sessions(active_col_id)
        if not sessions:
            new_id = db.create_chat_session(active_col_id, "Research Chat Session")
            sessions = db.list_chat_sessions(active_col_id)
            st.session_state["active_session_id"] = new_id

        sess_options = {s.get("title", f"Session {idx+1}"): s.get("id", "") for idx, s in enumerate(sessions)}
        current_sess_id = st.session_state.get("active_session_id")
        if current_sess_id not in sess_options.values():
            current_sess_id = list(sess_options.values())[0]
            st.session_state["active_session_id"] = current_sess_id

        # Determine index
        current_idx = 0
        for idx, sid in enumerate(sess_options.values()):
            if sid == current_sess_id:
                current_idx = idx
                break

        selected_title = st.selectbox("Active Session", list(sess_options.keys()), index=current_idx)
        active_sess_id = sess_options[selected_title]
        st.session_state["active_session_id"] = active_sess_id

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("➕ New Chat", use_container_width=True, type="secondary"):
                new_id = db.create_chat_session(active_col_id, f"Research Chat #{len(sessions) + 1}")
                st.session_state["active_session_id"] = new_id
                st.rerun()

        with btn_c2:
            if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
                confirm_clear_session(active_sess_id, active_col_id)

    # -----------------------------------------------------------------------
    # Main Header
    # -----------------------------------------------------------------------
    page_header(
        title="Grounded Research Assistant",
        subtitle=f"Directly queries {len(papers)} indexed papers in {active_col_name} with inline citations & automated claim verification",
        badge_text="Grounded RAG",
    )

    if not papers:
        empty_state(
            title="No Research Papers Available",
            message="Please upload and process research PDFs before querying the assistant.",
            action_label="Go to Paper Upload",
            action_page="pages/1_Paper_Upload.py",
            icon="📄",
        )
        return

    # Check for incoming question from Search or other modules
    incoming_prompt = st.session_state.pop("pending_chat_prompt", None)

    # Load conversation records
    chat_history_records = db.get_chat_messages(active_sess_id)

    # Transcript export button in top action row
    act_col1, act_col2 = st.columns([4, 1.2])
    with act_col2:
        if chat_history_records:
            transcript_md = format_transcript_markdown(selected_title, active_col_name, chat_history_records)
            st.download_button(
                label="📥 Export Chat",
                data=transcript_md,
                file_name=f"research_transcript_{active_sess_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # -----------------------------------------------------------------------
    # Empty State & Starter Prompts
    # -----------------------------------------------------------------------
    triggered_starter: Optional[str] = None
    if not chat_history_records and not incoming_prompt:
        st.markdown(
            """
            <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: var(--rl-text-primary);">💡 Explore This Research Collection</h4>
                <p style="margin: 0; font-size: 0.9rem; color: var(--rl-text-secondary);">
                    Ask comparative questions, investigate specific algorithms, or verify reported empirical metrics. Every statement is validated against source chunks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Suggested Starter Inquiries:")
        st_c1, st_c2 = st.columns(2)
        with st_c1:
            if st.button("📊 Compare methodologies & architectures across all papers", use_container_width=True):
                triggered_starter = "Compare the primary methodologies, core architectures, and algorithmic novelties across the papers in this collection."
            if st.button("📈 What datasets and benchmark results are reported?", use_container_width=True):
                triggered_starter = "What datasets, evaluation metrics, and quantitative benchmark results are reported across these papers?"

        with st_c2:
            if st.button("⚠️ What are the critical limitations and failure modes?", use_container_width=True):
                triggered_starter = "Summarize the primary limitations, failure cases, and assumptions noted by the authors in their discussion or conclusion."
            if st.button("🔮 What open challenges and future directions are proposed?", use_container_width=True):
                triggered_starter = "What open research directions and future extensions do the authors propose?"

    # -----------------------------------------------------------------------
    # Render Conversation History
    # -----------------------------------------------------------------------
    for msg in chat_history_records:
        msg_role = msg.get("role", "user")
        msg_content = msg.get("content", "")
        with st.chat_message(msg_role):
            st.markdown(msg_content)

            if msg_role == "assistant":
                ver = msg.get("verification")
                cits = msg.get("citations")

                # Verification score badge
                if ver and isinstance(ver, dict):
                    score = ver.get("overall_support_score", 1.0)
                    score_pct = int(score * 100)
                    if score_pct >= 75:
                        badge_html = status_badge("ready", f"Evidence Support: {score_pct}%")
                    elif score_pct >= 40:
                        badge_html = status_badge("warning", f"Evidence Support: {score_pct}%")
                    else:
                        badge_html = status_badge("danger", f"Evidence Support: {score_pct}%")

                    st.markdown(
                        f"<div style='margin-top: 0.4rem; margin-bottom: 0.4rem;'>{badge_html}</div>",
                        unsafe_allow_html=True,
                    )

                    unsupported = ver.get("unsupported_claims", [])
                    if unsupported:
                        banner(
                            alert_type="warning",
                            title="Potential Unverified Statements Detected",
                            message="The following claims in the response could not be strictly grounded in the retrieved passages:",
                            details=[f'"{u}"' for u in unsupported],
                        )

                # Citations drawer
                if cits and isinstance(cits, list):
                    with st.expander(f"📚 View {len(cits)} Grounded Citations & Excerpts"):
                        for c_idx, cit in enumerate(cits, 1):
                            p_title = cit.get("paper_title", "Unknown Paper")
                            p_num = cit.get("page_number", "?")
                            s_name = cit.get("section_name", "General")
                            snip_txt = cit.get("snippet", "")
                            pill = citation_pill(p_num)

                            st.markdown(
                                f"""
                                <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span style="font-weight: 600; font-size: 0.9rem; color: var(--rl-text-primary);">Source #{c_idx}: {html.escape(p_title)}</span>
                                        <div>{pill} <span style="font-size: 0.75rem; color: var(--rl-text-muted); margin-left: 4px;">Section: {html.escape(s_name)}</span></div>
                                    </div>
                                    <div class="rl-evidence-quote" style="margin: 0; font-size: 0.88rem;">
                                        "{html.escape(snip_txt)}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # -----------------------------------------------------------------------
    # Handle New Input (User input, Starter button, or Incoming prompt)
    # -----------------------------------------------------------------------
    user_query = st.chat_input("Ask a question about methods, results, datasets, or paper comparisons...")

    active_prompt_to_run = user_query or triggered_starter or incoming_prompt

    if active_prompt_to_run:
        # Display user question
        with st.chat_message("user"):
            st.markdown(active_prompt_to_run)

        # Store user message in DB
        db.insert_chat_message(
            session_id=active_sess_id,
            role="user",
            content=active_prompt_to_run,
        )

        # Multi-turn history formatting
        history_for_rewrite = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in chat_history_records[-4:]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence passages and synthesizing grounded answer..."):
                rag_result = answer_research_question(
                    query=active_prompt_to_run,
                    collection_id=active_col_id,
                    paper_ids=selected_paper_ids,
                    chat_history=history_for_rewrite,
                    top_k=top_k,
                    db=db,
                )

                st.markdown(rag_result.answer)

                # Support badge
                score_pct = int(rag_result.overall_support_score * 100)
                if score_pct >= 75:
                    badge_html = status_badge("ready", f"Evidence Support: {score_pct}%")
                elif score_pct >= 40:
                    badge_html = status_badge("warning", f"Evidence Support: {score_pct}%")
                else:
                    badge_html = status_badge("danger", f"Evidence Support: {score_pct}%")

                st.markdown(
                    f"<div style='margin-top: 0.4rem; margin-bottom: 0.4rem;'>{badge_html}</div>",
                    unsafe_allow_html=True,
                )

                # Unsupported claims warning
                unsupported_texts = [
                    v.claim_text for v in rag_result.verifications
                    if v.support_level == "unsupported"
                ]
                if unsupported_texts:
                    banner(
                        alert_type="warning",
                        title="Potential Unverified Statements Detected",
                        message="The following claims could not be strictly grounded in the retrieved passages:",
                        details=[f'"{u}"' for u in unsupported_texts],
                    )

                # Citations expander
                if rag_result.citations:
                    with st.expander(f"📚 View {len(rag_result.citations)} Grounded Citations & Excerpts"):
                        for c_idx, cit in enumerate(rag_result.citations, 1):
                            pill = citation_pill(cit.page_number)
                            st.markdown(
                                f"""
                                <div style="background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span style="font-weight: 600; font-size: 0.9rem; color: var(--rl-text-primary);">Source #{c_idx}: {html.escape(cit.paper_title)}</span>
                                        <div>{pill} <span style="font-size: 0.75rem; color: var(--rl-text-muted); margin-left: 4px;">Section: {html.escape(cit.section_name)}</span></div>
                                    </div>
                                    <div class="rl-evidence-quote" style="margin: 0; font-size: 0.88rem;">
                                        "{html.escape(cit.snippet)}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Persist assistant response in DB
                db.insert_chat_message(
                    session_id=active_sess_id,
                    role="assistant",
                    content=rag_result.answer,
                    citations=[c.model_dump() for c in rag_result.citations],
                    verification={
                        "overall_support_score": rag_result.overall_support_score,
                        "unsupported_claims": unsupported_texts,
                        "claims": [v.model_dump() for v in rag_result.verifications],
                    },
                )

        # Trigger re-render to update session history state cleanly
        st.rerun()


if __name__ == "__main__":
    main()
