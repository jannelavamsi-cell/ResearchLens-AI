"""
Reusable presentation components for ResearchLens AI.
Provides unified headers, metric cards, citation pills, verification badges,
alert banners, and component cards.
"""

from typing import Any, List, Optional

import streamlit as st


def render_header(title: str, subtitle: str, icon: str = "🔬") -> None:
    """Renders a unified academic page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.25rem;">
                <span>{icon}</span> <span>{title}</span>
            </h1>
            <p style="color: #64748b; font-size: 1.05rem; margin: 0; line-height: 1.5;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, sublabel: str = "") -> None:
    """Renders a styled metric card."""
    sublabel_html = f'<div class="rl-metric-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f"""
        <div class="rl-metric-box">
            <div class="rl-metric-label">{label}</div>
            <div class="rl-metric-value">{value}</div>
            {sublabel_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citation_pill(page: int, paper_num: Optional[int] = None) -> str:
    """Returns HTML for an evidence citation pill."""
    text = f"p. {page}" if paper_num is None else f"Paper {paper_num}, p. {page}"
    return f'<span class="rl-citation-pill">📍 {text}</span>'


def render_verification_badge(verified: bool, label: Optional[str] = None) -> str:
    """Returns HTML for a quote verification status badge."""
    if verified:
        text = label or "Verified Grounding"
        return f'<span class="rl-badge rl-badge-verified">✓ {text}</span>'
    text = label or "Unverified Quote"
    return f'<span class="rl-badge rl-badge-unverified">⚠ {text}</span>'


def render_alert_box(
    title: str,
    message: str,
    alert_type: str = "warning",
    details: Optional[List[str]] = None,
) -> None:
    """Renders a styled notice / alert banner."""
    type_class = f"rl-alert-{alert_type}"
    details_html = ""
    if details:
        items = "".join(f"<li style='margin-bottom: 0.25rem;'>{d}</li>" for d in details)
        details_html = f"<ul style='margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.25rem;'>{items}</ul>"

    icon_map = {
        "warning": "⚠️",
        "info": "ℹ️",
        "danger": "🚨",
        "success": "✅",
    }
    icon = icon_map.get(alert_type, "ℹ️")

    st.markdown(
        f"""
        <div class="rl-alert {type_class}">
            <div class="rl-alert-title">{icon} {title}</div>
            <div style="font-size: 0.92rem; line-height: 1.5;">{message}</div>
            {details_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_component_card(
    field_name: str,
    value: str,
    evidence_quote: str = "",
    page: int = 1,
    verified: bool = True,
    confidence: float = 1.0,
) -> None:
    """Renders a grounded structured component with citation and quote box."""
    badge = render_verification_badge(verified)
    pill = render_citation_pill(page) if page > 0 else ""

    conf_str = f"{int(confidence * 100)}%"
    title_formatted = field_name.replace("_", " ").title()

    quote_html = ""
    if evidence_quote and evidence_quote.strip() and evidence_quote != "Not stated in paper":
        quote_html = f"""
        <div style="margin-top: 0.75rem;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Evidence Excerpt:</div>
            <div class="rl-evidence-quote">"{evidence_quote}"</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="rl-card">
            <div class="rl-card-header">
                <span class="rl-card-title">{title_formatted}</span>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    {pill}
                    {badge}
                    <span style="font-size: 0.75rem; color: #94a3b8;">Conf: {conf_str}</span>
                </div>
            </div>
            <div style="color: #1e293b; font-size: 0.95rem; line-height: 1.6;">
                {value}
            </div>
            {quote_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Renders persistent application footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem 0;">
            ResearchLens AI • Evidence-Grounded Scientific Document Analysis System • Built with Streamlit & PyMuPDF
        </div>
        """,
        unsafe_allow_html=True,
    )
