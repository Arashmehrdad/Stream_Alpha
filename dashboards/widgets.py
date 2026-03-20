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
