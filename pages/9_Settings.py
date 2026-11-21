"""
ResearchLens AI - Settings & System Configuration Page.
Provides LLM provider inspection, real health check diagnostics, retrieval tuning,
theme guidance, and system version details.
"""

import streamlit as st

from core.config import settings
from core.llm import get_llm_client
from ui.components import banner, page_header, render_footer, safe_page_render
from ui.theme import inject_custom_css

inject_custom_css()


@safe_page_render
def render_settings() -> None:
    page_header(
        title="Settings & System Diagnostics",
        subtitle="Manage LLM runtime providers, inspect hardware configuration, and review retrieval parameters.",
        icon="⚙️",
        badge="System Hub",
    )

    tab_provider, tab_retrieval, tab_about = st.tabs([
        "🤖 LLM & Intelligence Provider",
        "🔍 Retrieval & Index Settings",
        "ℹ️ About & Architecture",
    ])

    with tab_provider:
        st.subheader("Configured LLM Provider")
        st.markdown(
            "ResearchLens AI features a provider-agnostic LLM interface supporting Google Gemini API, "
            "local Ollama (for 100% offline data privacy), and OpenAI-compatible endpoints."
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.text_input("Active Provider", value=settings.llm_provider.upper(), disabled=True)
            if settings.llm_provider == "gemini":
                st.text_input("Gemini Model", value=settings.gemini_model, disabled=True)
                key_masked = f"{settings.gemini_api_key[:6]}...{settings.gemini_api_key[-4:]}" if len(settings.gemini_api_key) > 10 else "Not set"
                st.text_input("API Key Status", value=key_masked, disabled=True)
            elif settings.llm_provider == "ollama":
                st.text_input("Ollama Base URL", value=settings.ollama_base_url, disabled=True)
                st.text_input("Ollama Model", value=settings.ollama_model, disabled=True)
            else:
                st.text_input("OpenAI Model", value=settings.openai_model, disabled=True)

        with col_p2:
            st.markdown("##### Real-Time Provider Health Check")
            st.write("Executes an actual diagnostic ping to the configured model backend.")
            if st.button("🔌 Run Live Health Check", type="primary", use_container_width=True):
                with st.spinner("Connecting to model backend..."):
                    llm = get_llm_client()
                    health = llm.health_check()
                    status = health.get("status", "error")
                    msg = health.get("message", "")
                    model = health.get("model", "Unknown")

                    if status == "healthy":
                        banner("success", "LLM Provider Connected", f"Successfully reached {settings.llm_provider.upper()} backend with model `{model}`.")
                    elif status == "warning":
                        banner("warning", "Provider Warning", msg)
                    elif status == "unconfigured":
                        banner("warning", "Provider Unconfigured", "Missing API key in `.env`. Please update your configuration.")
                    else:
                        banner("danger", "Connection Error", f"Failed to connect: {msg}", details=["Verify server is running if using Ollama.", "Check network connectivity or API key quota."])

    with tab_retrieval:
        st.subheader("Hybrid Retrieval & Grounding Hyperparameters")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("##### Dense & Sparse Indexing")
            st.text_input("Embedding Model", value=settings.embedding_model_name, disabled=True)
            st.text_input("Embedding Device", value=settings.embedding_device.upper(), disabled=True)
            st.text_input("Reranker Model", value=settings.reranker_model_name if settings.enable_reranker else "Disabled", disabled=True)

        with col_r2:
            st.markdown("##### Passages & Fusions")
            st.number_input("Target Chunk Tokens", value=settings.chunk_size_tokens, disabled=True)
            st.number_input("Chunk Overlap Tokens", value=settings.chunk_overlap_tokens, disabled=True)
            st.number_input("Reciprocal Rank Fusion k", value=60, disabled=True)

        st.caption("Parameters are loaded from `.env` and `core/config.py`.")

    with tab_about:
        st.subheader("About ResearchLens AI")
        st.markdown(
            """
            **ResearchLens AI** is an intelligent, evidence-grounded research paper analysis and comparative literature review platform.

            - **Version:** `1.1.0-ai-saas`
            - **Architecture:** Section-Aware PyMuPDF Ingestion • Hybrid RRF Search (FAISS + BM25) • Cross-Encoder Reranking • Grounded Verification Judge
            - **Design Philosophy:** Grounded, verifiable outputs. Every claim is attributed to physical pages with token-level fuzzy quote confirmation.
            """
        )

    render_footer()


render_settings()
