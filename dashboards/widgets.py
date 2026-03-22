"""Small Streamlit rendering helpers for the Stream Alpha M6 dashboard."""

from __future__ import annotations

import streamlit as st


def render_health_card(*, title: str, state: str, detail: str, healthy: bool) -> None:
    """Render one compact health card."""
    container = st.container(border=True)
    container.caption(title)
    container.metric("State", state)
    if healthy:
        container.success(detail)
    else:
        container.error(detail)


def render_table(title: str, rows: list[dict]) -> None:
    """Render one table or a clear empty state."""
    st.subheader(title)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info(f"No data available for {title.lower()}.")


_BANNER_THEME = {
    "success": {"background": "#0f3d2e", "border": "#34d399", "text": "#ecfdf5"},
    "warning": {"background": "#4a3200", "border": "#fbbf24", "text": "#fffbeb"},
    "error": {"background": "#4c0519", "border": "#fb7185", "text": "#fff1f2"},
}


def render_operator_banner(banner: dict[str, str | None]) -> None:
    """Render the persistent operator banner with strong safety posture styling."""
    theme = _BANNER_THEME.get(str(banner.get("severity")), _BANNER_THEME["warning"])
    mode = banner.get("mode", "-")
    posture = banner.get("safety_posture", "-")
    venue = banner.get("venue", "-")
    environment = banner.get("environment", "-")
    latest_evaluation_time = banner.get("latest_evaluation_time", "-")
    latest_evaluation_age = banner.get("latest_evaluation_age", "-")
    primary_reason = ""
    if banner.get("safety_posture") in {"degraded", "blocked"} and banner.get(
        "primary_reason_code"
    ):
        primary_reason = (
            f"<div style='margin-top:0.5rem; font-size:0.95rem;'>"
            f"<strong>Active reason:</strong> {banner['primary_reason_code']}"
        )
        if banner.get("primary_reason_detail"):
            primary_reason += f" - {banner['primary_reason_detail']}"
        primary_reason += "</div>"

    st.markdown(
        f"""
        <div style="
            background:{theme['background']};
            border:2px solid {theme['border']};
            border-radius:0.85rem;
            padding:1rem 1.2rem;
            margin:0.25rem 0 1rem 0;
            color:{theme['text']};
        ">
            <div style="font-size:0.9rem; opacity:0.92;">Stream Alpha Operator Posture</div>
            <div style="display:flex; flex-wrap:wrap; gap:1.25rem; margin-top:0.35rem;">
                <div><strong>Mode</strong><br>{mode}</div>
                <div><strong>Safety posture</strong><br>{posture}</div>
                <div><strong>Venue</strong><br>{venue}</div>
                <div><strong>Environment</strong><br>{environment}</div>
                <div><strong>Latest evaluation</strong><br>{latest_evaluation_time}</div>
                <div><strong>Evaluation age</strong><br>{latest_evaluation_age}</div>
            </div>
            {primary_reason}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_critical_state_strip(strip: dict[str, object]) -> None:
    """Render a compact always-visible live critical-state strip near the banner."""
    if not bool(strip.get("visible")):
        return

    items = strip.get("items", [])
    if not isinstance(items, list) or not items:
        return

    st.markdown("**Live Critical State**")
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        severity = str(item.get("severity", "warning"))
        theme = _BANNER_THEME.get(severity, _BANNER_THEME["warning"])
        detail = item.get("detail")
        detail_html = ""
        if detail:
            detail_html = (
                f"<div style='margin-top:0.35rem; font-size:0.8rem; opacity:0.92;'>"
                f"{detail}</div>"
            )
        column.markdown(
            f"""
            <div style="
                background:{theme['background']};
                border:1px solid {theme['border']};
                border-radius:0.75rem;
                padding:0.65rem 0.7rem;
                min-height:6.2rem;
                color:{theme['text']};
                margin-bottom:0.5rem;
            ">
                <div style="font-size:0.75rem; opacity:0.92;">{item.get("label", "-")}</div>
                <div style="font-size:1.05rem; font-weight:700; margin-top:0.15rem;">
                    {item.get("value", "-")}
                </div>
                {detail_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if strip.get("primary_block_reason_code"):
        block_detail = strip.get("block_detail") or "Live submit is blocked."
        st.markdown(
            f"""
            <div style="
                background:{_BANNER_THEME['error']['background']};
                border:1px solid {_BANNER_THEME['error']['border']};
                border-radius:0.75rem;
                padding:0.7rem 0.9rem;
                color:{_BANNER_THEME['error']['text']};
                margin-bottom:1rem;
            ">
                <strong>Primary live block</strong>: {strip['primary_block_reason_code']}<br>
                <span style="font-size:0.9rem;">{block_detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_summary_cards(*, title: str, items: list[dict[str, str]]) -> None:
    """Render a row of operator-friendly summary cards."""
    st.subheader(title)
    if not items:
        st.info(f"No data available for {title.lower()}.")
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        container = column.container(border=True)
        container.caption(item.get("label", "Summary"))
        container.markdown(f"**{item.get('value', '-')}**")
        if item.get("detail"):
            container.caption(item["detail"])


def render_incidents_panel(rows: list[dict]) -> None:
    """Render the active incidents panel or an explicit empty-safe state."""
    st.subheader("Active Incidents")
    if not rows:
        st.success(
            "No active incidents are currently aggregated from live safety, "
            "reliability, freshness, breaker, or blocked-trade state."
        )
        return

    critical_count = sum(int(row.get("severity") == "CRITICAL") for row in rows)
    high_count = sum(int(row.get("severity") == "HIGH") for row in rows)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Active incidents", str(len(rows)))
    metric_columns[1].metric("Critical", str(critical_count))
    metric_columns[2].metric("High", str(high_count))
    st.dataframe(rows, width="stretch", hide_index=True)
