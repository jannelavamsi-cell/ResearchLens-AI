"""
Reusable presentation components for ResearchLens AI.
Provides unified headers, stat cards, module action cards, paper cards, citation pills,
status badges, evidence panels, keyword chips, empty states, alert banners, and safe-page wrappers.
"""

import functools
import logging
import textwrap
from typing import Any, Callable, List, Optional

import streamlit as st

logger = logging.getLogger("researchlens.ui.components")


# ---------------------------------------------------------------------------
# 1. Page Header & Hero Components
# ---------------------------------------------------------------------------

def page_header(
    title: str,
    subtitle: str,
    icon: str = "🔬",
    badge: Optional[str] = None,
) -> None:
    """Renders a unified academic SaaS page header."""
    badge_html = f'<span class="rl-badge rl-badge-neutral">{badge}</span>' if badge else ""
    header_html = textwrap.dedent(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <h1 style="display: flex; align-items: center; gap: 0.65rem; margin: 0;">
                    <span>{icon}</span> <span>{title}</span>
                </h1>
                {badge_html}
            </div>
            <p style="color: var(--rl-text-secondary); font-size: 1rem; margin-top: 0.35rem; line-height: 1.5;">{subtitle}</p>
        </div>
        """
    ).strip()
    st.markdown(header_html, unsafe_allow_html=True)


# Alias for backward compatibility
render_header = page_header


# ---------------------------------------------------------------------------
# 2. Metric & KPI Cards
# ---------------------------------------------------------------------------

def stat_card(
    label: str,
    value: Any,
    sublabel: str = "",
    icon: str = "",
    delta: Optional[str] = None,
) -> None:
    """Renders a styled SaaS metric card."""
    icon_html = f'<span style="font-size: 1.25rem; margin-right: 0.35rem;">{icon}</span>' if icon else ""
    sublabel_html = f'<div class="rl-metric-sublabel">{sublabel}</div>' if sublabel else ""
    delta_html = f'<span style="font-size: 0.75rem; color: #059669; font-weight: 600; margin-left: 0.5rem;">{delta}</span>' if delta else ""

    card_html = (
        '<div class="rl-metric-box">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<div class="rl-metric-label">{label}</div>{icon_html}'
        '</div>'
        '<div style="display: flex; align-items: baseline;">'
        f'<div class="rl-metric-value">{value}</div>{delta_html}'
        '</div>'
        f'{sublabel_html}'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# Alias for backward compatibility
render_metric_card = stat_card


# ---------------------------------------------------------------------------
# 3. Badges & Citation Pills
# ---------------------------------------------------------------------------

def citation_pill(page: int, paper_ref: Optional[Any] = None) -> str:
    """Returns HTML for an evidence citation pill."""
    if paper_ref is not None:
        text = f"Paper {paper_ref}, p. {page}" if isinstance(paper_ref, int) else f"{paper_ref}, p. {page}"
    else:
        text = f"p. {page}"
    return f'<span class="rl-citation-pill">📍 {text}</span>'


render_citation_pill = citation_pill


def status_badge(status_type: str, label: Optional[str] = None) -> str:
    """Returns HTML for a status badge."""
    st_lower = status_type.lower()
    if st_lower in ("ready", "verified", "success", "done"):
        cls = "rl-badge-ready"
        icon = "✓"
        text = label or ("Verified" if "verif" in st_lower else "Ready")
    elif st_lower in ("unverified", "warning", "pending"):
        cls = "rl-badge-warning"
        icon = "⚠"
        text = label or ("Unverified" if "unverif" in st_lower else "Warning")
    elif st_lower in ("danger", "failed", "error"):
        cls = "rl-badge-danger"
        icon = "✕"
        text = label or "Failed"
    else:
        cls = "rl-badge-neutral"
        icon = "ℹ"
        text = label or status_type.capitalize()

    return f'<span class="rl-badge {cls}">{icon} {text}</span>'


def render_verification_badge(verified: bool, label: Optional[str] = None) -> str:
    """Returns HTML for quote verification badge (backward compatibility)."""
    return status_badge("verified" if verified else "unverified", label)


# ---------------------------------------------------------------------------
# 4. Banner / Alert Messages
# ---------------------------------------------------------------------------

def banner(
    alert_type: str,
    title: str,
    message: str,
    details: Optional[List[str]] = None,
) -> None:
    """Renders a styled notice / alert banner."""
    type_class = f"rl-alert-{alert_type.lower()}"
    details_html = ""
    if details:
        items = "".join(f"<li style='margin-bottom: 0.25rem;'>{d}</li>" for d in details)
        details_html = f"<ul style='margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.25rem;'>{items}</ul>"

    icon_map = {
        "warning": "⚠️",
        "info": "ℹ️",
        "danger": "🚨",
        "error": "🚨",
        "success": "✅",
    }
    icon = icon_map.get(alert_type.lower(), "ℹ️")

    b_html = textwrap.dedent(
        f"""
        <div class="rl-alert {type_class}">
            <div class="rl-alert-title">{icon} {title}</div>
            <div style="font-size: 0.92rem; line-height: 1.5;">{message}</div>
            {details_html}
        </div>
        """
    ).strip()
    st.markdown(b_html, unsafe_allow_html=True)


render_alert_box = banner


# ---------------------------------------------------------------------------
# 5. Evidence Quote & Component Cards
# ---------------------------------------------------------------------------

def evidence_panel(quote: str, page: int = 1, paper_title: str = "") -> None:
    """Renders a styled expandable source passage evidence box."""
    if not quote or quote.strip() == "Not stated in paper":
        return

    header_text = f"Source Passage (Page {page})" if page > 0 else "Source Passage"
    if paper_title:
        header_text = f"{paper_title} — {header_text}"

    panel_html = textwrap.dedent(
        f"""
        <div style="margin-top: 0.5rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.75rem; font-weight: 600; color: var(--rl-text-muted); text-transform: uppercase;">
                {header_text}:
            </div>
            <div class="rl-evidence-quote">"{quote.strip()}"</div>
        </div>
        """
    ).strip()
    st.markdown(panel_html, unsafe_allow_html=True)


def render_component_card(
    field_name: str,
    value: str,
    evidence_quote: str = "",
    page: int = 1,
    verified: bool = True,
    confidence: float = 1.0,
) -> None:
    """Renders a grounded structured component with citation and quote box."""
    badge = status_badge("verified" if verified else "unverified")
    pill = citation_pill(page) if page > 0 else ""
    conf_str = f"{int(confidence * 100)}%"
    title_formatted = field_name.replace("_", " ").title()

    quote_html = ""
    if evidence_quote and evidence_quote.strip() and evidence_quote != "Not stated in paper":
        quote_html = f"""
        <div style="margin-top: 0.75rem;">
            <div style="font-size: 0.75rem; font-weight: 600; color: var(--rl-text-muted); text-transform: uppercase;">Evidence Excerpt:</div>
            <div class="rl-evidence-quote">"{evidence_quote.strip()}"</div>
        </div>
        """

    c_html = textwrap.dedent(
        f"""
        <div class="rl-card">
            <div class="rl-card-header">
                <span class="rl-card-title">{title_formatted}</span>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    {pill}
                    {badge}
                    <span style="font-size: 0.75rem; color: var(--rl-text-muted);">Conf: {conf_str}</span>
                </div>
            </div>
            <div style="color: var(--rl-text-primary); font-size: 0.95rem; line-height: 1.6;">
                {value}
            </div>
            {quote_html}
        </div>
        """
    ).strip()
    st.markdown(c_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. Keyword Chips & Empty States
# ---------------------------------------------------------------------------

def keyword_chips(keywords: List[Any], max_display: int = 15) -> None:
    """Renders an interactive list of keyword chips."""
    if not keywords:
        st.caption("No keywords extracted.")
        return

    chips_html = []
    for k in keywords[:max_display]:
        if isinstance(k, dict):
            word = k.get("keyword", "")
            score = k.get("score")
            text = f"{word} ({float(score):.2f})" if score is not None else word
        else:
            text = str(k)
        if text.strip():
            chips_html.append(f'<span class="rl-keyword-chip">🏷️ {text.strip()}</span>')

    st.markdown(f'<div style="line-height: 2.2;">{" ".join(chips_html)}</div>', unsafe_allow_html=True)


def empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_page: Optional[str] = None,
) -> None:
    """Renders a friendly SaaS empty state with optional navigation action."""
    e_html = textwrap.dedent(
        f"""
        <div class="rl-empty-state">
            <div class="rl-empty-icon">{icon}</div>
            <div class="rl-empty-title">{title}</div>
            <div class="rl-empty-desc">{message}</div>
        </div>
        """
    ).strip()
    st.markdown(e_html, unsafe_allow_html=True)
    if action_label and action_page:
        col_l, col_btn, col_r = st.columns([2, 1, 2])
        with col_btn:
            if st.button(action_label, key=f"empty_btn_{action_page}", type="primary", use_container_width=True):
                st.switch_page(action_page)


# ---------------------------------------------------------------------------
# 7. Safe Render Error Boundary
# ---------------------------------------------------------------------------

def safe_page_render(page_name_or_func: Optional[Any] = None) -> Any:
    """Decorator to catch unhandled errors on a page and render a friendly SaaS error banner.
    Supports both @safe_page_render and @safe_page_render("Page Name").
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as err:
                page_label = page_name_or_func if isinstance(page_name_or_func, str) else getattr(func, "__name__", "Page")
                logger.error(f"Error rendering page {page_label}: {err}", exc_info=True)
                banner(
                    alert_type="danger",
                    title=f"Something went wrong rendering {page_label}",
                    message=f"An unexpected error occurred: {str(err)}",
                    details=[
                        "Ensure your database and models are loaded properly.",
                        "Click 'Retry Page' below to reload the view.",
                    ],
                )
                btn_key = f"retry_{getattr(func, '__name__', 'page')}"
                if st.button("🔄 Retry Page", key=btn_key, type="primary"):
                    st.rerun()
        return wrapper

    if callable(page_name_or_func):
        return decorator(page_name_or_func)
    return decorator


# ---------------------------------------------------------------------------
# 8. Persistent Footer
# ---------------------------------------------------------------------------

def render_footer() -> None:
    """Renders persistent application footer."""
    st.markdown("---")
    f_html = textwrap.dedent(
        """
        <div style="text-align: center; color: var(--rl-text-muted); font-size: 0.82rem; padding: 1rem 0;">
            <b>ResearchLens AI</b> • Evidence-Grounded Scientific Document Intelligence Platform • Version 1.1
        </div>
        """
    ).strip()
    st.markdown(f_html, unsafe_allow_html=True)
