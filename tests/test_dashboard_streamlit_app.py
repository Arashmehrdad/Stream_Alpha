"""Focused Streamlit app helper tests for the operator console."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from datetime import datetime, timezone

from dashboards.data_sources import ApiHealthSnapshot, DashboardSnapshot, DatabaseSnapshot
from dashboards.streamlit_app import resolve_display_runtime_profile


def test_runtime_profile_prefers_loaded_api_snapshot() -> None:
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="ok",
            runtime_profile="shadow",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert resolve_display_runtime_profile(snapshot=snapshot) == "SHADOW"


def test_runtime_profile_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "live")
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=False,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="unavailable",
            runtime_profile=None,
            error="api down",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert resolve_display_runtime_profile(snapshot=snapshot) == "LIVE"
