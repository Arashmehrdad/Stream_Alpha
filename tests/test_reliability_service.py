"""Focused service-level tests for M13 reliability truth semantics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.reliability.schemas import CircuitBreakerState
from app.reliability.service import (
    HEARTBEAT_STALE,
    build_signal_client_health_snapshot,
)


def _signal_client_state(
    *,
    checked_at: datetime,
    health_overall_status: str = "HEALTHY",
    breaker_state: str = "CLOSED",
    reason_code: str = "HEALTH_HEALTHY",
) -> CircuitBreakerState:
    return CircuitBreakerState(
        service_name="paper-trader",
        component_name="signal_client",
        breaker_state=breaker_state,
        health_overall_status=health_overall_status,
        last_heartbeat_at=checked_at,
        last_success_at=checked_at,
        reason_code=reason_code,
        detail="Signal client state",
        updated_at=checked_at,
    )


def test_signal_client_expected_idle_window_does_not_look_stale() -> None:
    """Expected quiet time between 5-minute candles should stay healthy."""
    evaluated_at = datetime(2026, 3, 31, 20, 15, tzinfo=timezone.utc)
    state = _signal_client_state(checked_at=evaluated_at - timedelta(minutes=4, seconds=30))

    snapshot = build_signal_client_health_snapshot(
        service_name="paper-trader",
        component_name="signal_client",
        state=state,
        evaluated_at=evaluated_at,
        heartbeat_stale_after_seconds=45,
        idle_healthy_after_seconds=345,
    )

    assert snapshot.health_overall_status == "HEALTHY"
    assert snapshot.heartbeat_freshness_status == "FRESH"
    assert snapshot.reason_code == "HEALTH_HEALTHY"


def test_signal_client_turns_unavailable_after_idle_window_expires() -> None:
    """Signal-client freshness should still fail closed after the idle grace expires."""
    evaluated_at = datetime(2026, 3, 31, 20, 15, tzinfo=timezone.utc)
    state = _signal_client_state(checked_at=evaluated_at - timedelta(minutes=6))

    snapshot = build_signal_client_health_snapshot(
        service_name="paper-trader",
        component_name="signal_client",
        state=state,
        evaluated_at=evaluated_at,
        heartbeat_stale_after_seconds=45,
        idle_healthy_after_seconds=345,
    )

    assert snapshot.health_overall_status == "UNAVAILABLE"
    assert snapshot.heartbeat_freshness_status == "STALE"
    assert snapshot.reason_code == HEARTBEAT_STALE


def test_signal_client_open_breaker_is_not_hidden_by_idle_window() -> None:
    """A real breaker-open state must remain unavailable even inside the idle window."""
    evaluated_at = datetime(2026, 3, 31, 20, 15, tzinfo=timezone.utc)
    state = _signal_client_state(
        checked_at=evaluated_at - timedelta(minutes=4),
        health_overall_status="UNAVAILABLE",
        breaker_state="OPEN",
        reason_code="BREAKER_OPENED",
    )

    snapshot = build_signal_client_health_snapshot(
        service_name="paper-trader",
        component_name="signal_client",
        state=state,
        evaluated_at=evaluated_at,
        heartbeat_stale_after_seconds=45,
        idle_healthy_after_seconds=345,
    )

    assert snapshot.health_overall_status == "UNAVAILABLE"
    assert snapshot.heartbeat_freshness_status == "FRESH"
    assert snapshot.reason_code == "BREAKER_OPENED"
