"""
Design system tokens and CSS injection engine for ResearchLens AI.
Provides clean academic styling, color tokens, typography scales,
card containers, citation pills, and verification badges.
"""

from typing import Dict

import streamlit as st

# Color Palette Tokens
COLORS: Dict[str, str] = {
    "primary": "#4f46e5",         # Indigo 600
    "primary_hover": "#4338ca",   # Indigo 700
    "primary_light": "#eef2ff",   # Indigo 50
    "secondary": "#0ea5e9",       # Sky 500
    "text_dark": "#0f172a",       # Slate 900
    "text_muted": "#64748b",      # Slate 500
    "bg_card": "#ffffff",         # Pure White
    "bg_subtle": "#f8fafc",       # Slate 50
    "border": "#e2e8f0",          # Slate 200
    "border_highlight": "#cbd5e1",# Slate 300
    "success": "#059669",         # Emerald 600
    "success_bg": "#ecfdf5",      # Emerald 50
    "warning": "#d97706",         # Amber 600
    "warning_bg": "#fffbeb",      # Amber 50
    "danger": "#e11d48",          # Rose 600
    "danger_bg": "#fff1f2",       # Rose 50
    "info": "#2563eb",            # Blue 600
    "info_bg": "#eff6ff",         # Blue 50
}

CUSTOM_CSS = """
<style>
/* ==========================================================================
   ResearchLens AI Modern Academic Design System
   ========================================================================== */

/* Main App Container & Typography */
.main .block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    font-weight: 700;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.15rem;
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 1.5rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

h3 {
    font-size: 1.2rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Reusable UI Cards */
.rl-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease-in-out;
}

.rl-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.07);
}

.rl-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.5rem;
}

.rl-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
}

/* Metric KPI Containers */
.rl-metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: left;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}

.rl-metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.rl-metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.rl-metric-sublabel {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Verification Badges */
.rl-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1;
}

.rl-badge-verified {
    background-color: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.rl-badge-unverified {
    background-color: #fff1f2;
    color: #e11d48;
    border: 1px solid #fecdd3;
}

.rl-badge-neutral {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

/* Physical Citation Pill */
.rl-citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Evidence Quote Blockquote */
.rl-evidence-quote {
    background-color: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-style: italic;
    color: #334155;
    font-size: 0.9rem;
    margin: 0.5rem 0;
    line-height: 1.5;
}

/* Alert Boxes */
.rl-alert {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border: 1px solid transparent;
}

.rl-alert-warning {
    background-color: #fffbeb;
    border-color: #fde68a;
    color: #92400e;
}

.rl-alert-info {
    background-color: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af;
}

.rl-alert-danger {
    background-color: #fff1f2;
    border-color: #fecdd3;
    color: #9f1239;
}

.rl-alert-success {
    background-color: #ecfdf5;
    border-color: #a7f3d0;
    color: #065f46;
}

.rl-alert-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Buttons */
button[kind="primary"] {
    background-color: #4f46e5 !important;
    border-color: #4f46e5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease-in-out !important;
}

button[kind="primary"]:hover {
    background-color: #4338ca !important;
    border-color: #4338ca !important;
    transform: translateY(-1px);
}

button[kind="secondary"] {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Tabs */
button[role="tab"] {
    font-weight: 600 !important;
    border-radius: 6px !important;
}

button[role="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
}
</style>
"""


def inject_custom_css() -> None:
    """Injects design system CSS into the current Streamlit app page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
