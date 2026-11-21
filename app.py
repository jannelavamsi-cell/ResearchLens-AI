"""
ResearchLens AI - Intelligent Research-Paper Analysis & Comparative Literature Review System.
Main Streamlit application entry point and structured navigation shell.
"""

import streamlit as st

from core.config import settings
from core.llm import get_llm_client
from core.logging_config import setup_logging
from database.db_manager import DBManager
from ui.theme import inject_custom_css

# Initialize system logging and relational database
setup_logging()
db = DBManager(settings.database_path)

# Page configuration
st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ---------------------------------------------------------------------------
# Workspace & Collection Session State Initialization
# ---------------------------------------------------------------------------

collections = db.list_collections()
if not collections:
    db.create_collection("Default Workspace", "Initial research collection")
    collections = db.list_collections()

if "active_collection_id" not in st.session_state:
    st.session_state["active_collection_id"] = collections[0].get("id", "")
    st.session_state["active_collection_name"] = collections[0].get("name", "Default Workspace")

# ---------------------------------------------------------------------------
# Multi-Page Navigation Configuration with Grouped Hierarchy
# ---------------------------------------------------------------------------

pages = {
    "Overview": [
        st.Page("pages/0_Dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True, url_path="dashboard"),
    ],
    "Library": [
        st.Page("pages/1_Paper_Upload.py", title="PDF Upload", icon=":material/upload_file:", url_path="upload"),
    ],
    "Analyze": [
        st.Page("pages/2_Paper_Analysis.py", title="Paper Analysis", icon=":material/analytics:", url_path="analysis"),
        st.Page("pages/2b_Keyword_Extraction.py", title="Keyword Extraction", icon=":material/label:", url_path="keywords"),
        st.Page("pages/3_Smart_Search.py", title="Semantic Search", icon=":material/search:", url_path="search"),
    ],
    "Intelligence": [
        st.Page("pages/4_Research_Chat.py", title="AI Assistant", icon=":material/smart_toy:", url_path="assistant"),
        st.Page("pages/5_Paper_Comparison.py", title="Paper Comparison", icon=":material/compare_arrows:", url_path="comparison"),
    ],
    "Write": [
        st.Page("pages/6_Literature_Review.py", title="Literature Review", icon=":material/auto_stories:", url_path="literature_review"),
        st.Page("pages/7_Research_Gaps.py", title="Research Gap Analysis", icon=":material/lightbulb:", url_path="research_gaps"),
    ],
    "More": [
        st.Page("pages/8_Trends_and_Export.py", title="Trends & Export", icon=":material/insights:", url_path="trends"),
        st.Page("pages/9_Settings.py", title="Settings", icon=":material/settings:", url_path="settings"),
    ],
}

pg = st.navigation(pages)

# ---------------------------------------------------------------------------
# Sidebar Top: Brand & Workspace Selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.25rem 0 0.85rem 0;">
            <div style="font-size: 1.35rem; font-weight: 800; color: var(--rl-primary); display: flex; align-items: center; gap: 0.5rem; letter-spacing: -0.02em;">
                <span>🔬</span> <span>ResearchLens AI</span>
            </div>
            <div style="font-size: 0.78rem; color: var(--rl-text-muted); font-weight: 500; margin-top: 0.15rem;">
                Evidence-Grounded Research Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Collection Selector
    col_names = [c.get("name", "Default Workspace") for c in collections]
    current_col_name = st.session_state.get("active_collection_name", col_names[0])
    selected_idx = col_names.index(current_col_name) if current_col_name in col_names else 0

    selected_col = st.selectbox(
        "Active Workspace",
        col_names,
        index=selected_idx,
        key="app_active_col_select",
    )

    # Sync selection to session state
    for c in collections:
        if c.get("name") == selected_col:
            st.session_state["active_collection_id"] = c.get("id", "")
            st.session_state["active_collection_name"] = c.get("name", selected_col)
            break

    # Quick Collection Creator Expander
    with st.expander("➕ New Workspace", expanded=False):
        new_name = st.text_input("Workspace Name", key="nav_new_col_name")
        new_desc = st.text_input("Description", key="nav_new_col_desc")
        if st.button("Create Workspace", use_container_width=True, key="nav_btn_new_col"):
            if new_name.strip():
                new_id = db.create_collection(new_name.strip(), new_desc.strip())
                st.session_state["active_collection_id"] = new_id
                st.session_state["active_collection_name"] = new_name.strip()
                st.success(f"Created '{new_name}'")
                st.rerun()

# ---------------------------------------------------------------------------
# Run Routed Page
# ---------------------------------------------------------------------------

pg.run()

# ---------------------------------------------------------------------------
# Sidebar Bottom: System Health & Version
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    llm = get_llm_client()
    health = llm.health_check()
    h_status = health.get("status", "error")
    h_model = health.get("model", "Active")

    if h_status == "healthy":
        st.markdown(f"🟢 **{settings.llm_provider.upper()} Connected**  \n`{h_model}`")
    elif h_status in ("warning", "unconfigured"):
        st.markdown(f"🟡 **{settings.llm_provider.upper()} Notice**  \n`{health.get('message', 'Checking...')}`")
    else:
        st.markdown(f"🔴 **{settings.llm_provider.upper()} Offline**  \n`{health.get('message', 'Error')}`")

    st.caption("ResearchLens AI v1.1 • AI-SaaS Edition")
