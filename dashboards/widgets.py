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
