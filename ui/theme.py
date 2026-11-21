"""
Modern AI-SaaS Design System Tokens and CSS Engine for ResearchLens AI.
Provides clean academic SaaS styling, CSS custom variables for Light/Dark modes,
responsive media queries, typography scales, buttons, badges, and shared Plotly templates.
"""

from typing import Dict

import plotly.graph_objects as go
import streamlit as st

# Color Palette Tokens
COLORS: Dict[str, str] = {
    "primary": "#4f46e5",         # Indigo 600
    "primary_hover": "#4338ca",   # Indigo 700
    "primary_light": "#eef2ff",   # Indigo 50
    "accent": "#06b6d4",          # Cyan 500
    "accent_hover": "#0891b2",    # Cyan 600
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
   ResearchLens AI: Modern AI-SaaS Design System & CSS Variables
   ========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --rl-bg: #f8fafc;
    --rl-surface: #ffffff;
    --rl-surface-hover: #f1f5f9;
    --rl-border: #e2e8f0;
    --rl-border-hover: #cbd5e1;
    --rl-text-primary: #0f172a;
    --rl-text-secondary: #475569;
    --rl-text-muted: #94a3b8;
    --rl-primary: #4f46e5;
    --rl-primary-hover: #4338ca;
    --rl-primary-subtle: #eef2ff;
    --rl-accent: #06b6d4;
    --rl-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --rl-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.07);
    --rl-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
}

[data-theme="dark"], .stApp[data-theme="dark"] {
    --rl-bg: #0b0f19;
    --rl-surface: #111827;
    --rl-surface-hover: #1f2937;
    --rl-border: #1f2937;
    --rl-border-hover: #374151;
    --rl-text-primary: #f9fafb;
    --rl-text-secondary: #cbd5e1;
    --rl-text-muted: #64748b;
    --rl-primary: #6366f1;
    --rl-primary-hover: #4f46e5;
    --rl-primary-subtle: rgba(99, 102, 241, 0.15);
    --rl-accent: #22d3ee;
    --rl-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.5);
    --rl-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    --rl-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6);
}

/* Base App Body & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--rl-text-primary);
}

.main .block-container {
    max-width: 1360px;
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    color: var(--rl-text-primary);
    font-weight: 700;
    letter-spacing: -0.025em;
}

h1 { font-size: 2.1rem; margin-bottom: 0.35rem; }
h2 { font-size: 1.5rem; margin-top: 1.25rem; margin-bottom: 0.5rem; }
h3 { font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.4rem; }

/* Streamlit Chrome Suppression */
#MainMenu, footer, [data-testid="stToolbar"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Reusable SaaS Card */
.rl-card {
    background-color: var(--rl-surface);
    border: 1px solid var(--rl-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--rl-shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.rl-card:hover {
    border-color: var(--rl-border-hover);
    box-shadow: var(--rl-shadow-md);
    transform: translateY(-1.5px);
}

.rl-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--rl-border);
    padding-bottom: 0.5rem;
}

.rl-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--rl-text-primary);
    margin: 0;
}

/* Hero Section */
.rl-hero-banner {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(6, 182, 212, 0.08) 100%);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 16px;
    padding: 2rem 2.25rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.rl-hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--rl-text-primary);
    margin: 0 0 0.5rem 0;
}

.rl-hero-subtitle {
    font-size: 1.1rem;
    color: var(--rl-text-secondary);
    line-height: 1.6;
    max-width: 860px;
    margin: 0;
}

/* Metric KPI Containers */
.rl-metric-box {
    background: var(--rl-surface);
    border: 1px solid var(--rl-border);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    text-align: left;
    box-shadow: var(--rl-shadow-sm);
    transition: all 0.2s ease;
}

.rl-metric-box:hover {
    border-color: var(--rl-primary);
    box-shadow: var(--rl-shadow-md);
    transform: translateY(-2px);
}

.rl-metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--rl-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}

.rl-metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--rl-text-primary);
    line-height: 1.2;
}

.rl-metric-sublabel {
    font-size: 0.8rem;
    color: var(--rl-text-muted);
    margin-top: 0.35rem;
}

/* Module Action Card */
.rl-module-card {
    background: var(--rl-surface);
    border: 1px solid var(--rl-border);
    border-radius: 14px;
    padding: 1.4rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 190px;
    box-shadow: var(--rl-shadow-sm);
    transition: all 0.2s ease-in-out;
}

.rl-module-card:hover {
    border-color: var(--rl-primary);
    box-shadow: var(--rl-shadow-md);
    transform: translateY(-2px);
}

.rl-module-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--rl-text-primary);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.rl-module-desc {
    font-size: 0.88rem;
    color: var(--rl-text-secondary);
    line-height: 1.5;
    margin-bottom: 1rem;
}

/* Workflow Step Strip */
.rl-workflow-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--rl-surface);
    border: 1px solid var(--rl-border);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    gap: 0.75rem;
}

.rl-workflow-step {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--rl-text-muted);
    border: 1px dashed var(--rl-border);
}

.rl-workflow-step-done {
    color: #059669;
    background: rgba(5, 150, 105, 0.08);
    border: 1px solid rgba(5, 150, 105, 0.25);
}

.rl-workflow-step-active {
    color: var(--rl-primary);
    background: var(--rl-primary-subtle);
    border: 1px solid rgba(79, 70, 229, 0.35);
}

/* Badges */
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

.rl-badge-ready, .rl-badge-verified, .rl-badge-success {
    background-color: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.rl-badge-unverified, .rl-badge-warning {
    background-color: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}

.rl-badge-danger, .rl-badge-failed {
    background-color: #fff1f2;
    color: #e11d48;
    border: 1px solid #fecdd3;
}

.rl-badge-neutral, .rl-badge-processing {
    background-color: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
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
    font-family: 'JetBrains Mono', monospace;
}

/* Keyword Chips */
.rl-keyword-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--rl-surface);
    border: 1px solid var(--rl-border);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--rl-text-secondary);
    margin: 0.2rem;
    transition: all 0.15s ease;
}

.rl-keyword-chip:hover {
    border-color: var(--rl-primary);
    color: var(--rl-primary);
    background: var(--rl-primary-subtle);
}

/* Evidence Quote Blockquote */
.rl-evidence-quote {
    background-color: var(--rl-surface-hover);
    border-left: 3px solid var(--rl-primary);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-style: italic;
    color: var(--rl-text-secondary);
    font-size: 0.9rem;
    margin: 0.5rem 0;
    line-height: 1.5;
}

/* Alert Boxes & Banners */
.rl-alert {
    border-radius: 12px;
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

/* Empty State Container */
.rl-empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background: var(--rl-surface);
    border: 2px dashed var(--rl-border);
    border-radius: 16px;
    margin: 1.5rem 0;
}

.rl-empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.rl-empty-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--rl-text-primary);
    margin-bottom: 0.5rem;
}

.rl-empty-desc {
    font-size: 0.92rem;
    color: var(--rl-text-muted);
    max-width: 480px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.5;
}

/* Buttons Styling */
button[kind="primary"] {
    background-color: var(--rl-primary) !important;
    border-color: var(--rl-primary) !important;
    color: #ffffff !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.08) !important;
    transition: all 0.15s ease-in-out !important;
}

button[kind="primary"]:hover {
    background-color: var(--rl-primary-hover) !important;
    border-color: var(--rl-primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--rl-shadow-md) !important;
}

button[kind="secondary"] {
    background-color: var(--rl-surface) !important;
    border: 1px solid var(--rl-border) !important;
    color: var(--rl-text-primary) !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
}

button[kind="secondary"]:hover {
    border-color: var(--rl-border-hover) !important;
    background-color: var(--rl-surface-hover) !important;
}

/* Danger Button custom helper */
.rl-btn-danger button {
    background-color: #e11d48 !important;
    border-color: #e11d48 !important;
    color: #ffffff !important;
}

.rl-btn-danger button:hover {
    background-color: #be123c !important;
}

/* Responsive Table Container */
.rl-table-container {
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--rl-border);
    border-radius: 10px;
    margin: 1rem 0;
}

/* Responsive Media Queries */
@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .rl-hero-title {
        font-size: 1.8rem;
    }
    .rl-workflow-strip {
        flex-direction: column;
        align-items: stretch;
    }
}

@media (max-width: 480px) {
    .main .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
    .rl-hero-title {
        font-size: 1.5rem;
    }
}
</style>
"""


def inject_custom_css() -> None:
    """Injects modern AI-SaaS design system CSS into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_plotly_template() -> go.layout.Template:
    """Returns a unified Plotly layout template matching the SaaS color palette."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"],
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1",
        ),
    )
    return template
