"""Pure primitive tests for the M13 reliability foundation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.reliability.config import CircuitBreakerConfig
from app.reliability.schemas import CircuitBreakerState
from app.reliability.service import (
    BREAKER_HALF_OPENED,
    BREAKER_OPENED,
    BREAKER_REOPENED,
    BREAKER_RESTORED,
    FEED_FRESH,
    FEATURE_STALE,
    HEALTH_DEGRADED_FRESHNESS,
    HEALTH_UNAVAILABLE_BREAKER_OPEN,
    PENDING_SIGNAL_EXPIRED,
    PENDING_SIGNAL_VALID,
    REGIME_FRESH,
    aggregate_health_status,
    evaluate_feed_freshness,
    evaluate_feature_freshness,
    evaluate_pending_signal_expiry,
    evaluate_regime_freshness,
    transition_circuit_breaker,
)


def test_feed_feature_and_regime_freshness_are_deterministic() -> None:
    """Freshness helpers should classify timestamps deterministically."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)

    feed_status = evaluate_feed_freshness(
        observed_at=now - timedelta(seconds=30),
        evaluated_at=now,
        max_age_seconds=90,
    )
    feature_status = evaluate_feature_freshness(
        observed_at=now - timedelta(seconds=121),
        evaluated_at=now,
        max_age_seconds=120,
    )
    regime_status = evaluate_regime_freshness(
        observed_at=now - timedelta(hours=1),
        evaluated_at=now,
        max_age_seconds=86400,
    )

    assert feed_status.freshness_status == "FRESH"
    assert feed_status.reason_code == FEED_FRESH
    assert feature_status.freshness_status == "STALE"
    assert feature_status.reason_code == FEATURE_STALE
    assert regime_status.freshness_status == "FRESH"
    assert regime_status.reason_code == REGIME_FRESH


def test_overall_health_aggregation_respects_freshness_and_breaker() -> None:
    """Health aggregation should degrade on stale freshness and fail on open breakers."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    stale_feature = evaluate_feature_freshness(
        observed_at=now - timedelta(seconds=300),
        evaluated_at=now,
        max_age_seconds=120,
    )
    fresh_feed = evaluate_feed_freshness(
        observed_at=now - timedelta(seconds=10),
        evaluated_at=now,
        max_age_seconds=90,
    )
    degraded = aggregate_health_status(
        freshness_statuses=(fresh_feed, stale_feature),
    )

    assert degraded.health_overall_status == "DEGRADED"
    assert degraded.reason_codes == (HEALTH_DEGRADED_FRESHNESS,)

    open_breaker = CircuitBreakerState(
        service_name="paper-trader",
        component_name="signal_client",
        breaker_state="OPEN",
        health_overall_status="UNAVAILABLE",
        opened_at=now - timedelta(seconds=5),
    )
    unavailable = aggregate_health_status(
        freshness_statuses=(fresh_feed, stale_feature),
        breaker_state=open_breaker,
    )

    assert unavailable.health_overall_status == "UNAVAILABLE"
    assert unavailable.reason_codes == (HEALTH_UNAVAILABLE_BREAKER_OPEN,)


def test_circuit_breaker_transitions_closed_open_half_open_and_restored() -> None:
    """Breaker transitions should stay explicit across failure and recovery states."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    config = CircuitBreakerConfig(
        failure_threshold=2,
        half_open_after_seconds=30,
        success_threshold=1,
    )
    initial = CircuitBreakerState(
        service_name="paper-trader",
        component_name="signal_client",
        breaker_state="CLOSED",
        health_overall_status="HEALTHY",
    )

    after_first_failure = transition_circuit_breaker(
        state=initial,
        config=config,
        evaluated_at=now,
        observed_success=False,
    )
    after_second_failure = transition_circuit_breaker(
        state=after_first_failure,
        config=config,
        evaluated_at=now + timedelta(seconds=1),
        observed_success=False,
    )
    half_open = transition_circuit_breaker(
        state=after_second_failure,
        config=config,
        evaluated_at=now + timedelta(seconds=31),
        observed_success=None,
    )
    restored = transition_circuit_breaker(
        state=half_open,
        config=config,
        evaluated_at=now + timedelta(seconds=32),
        observed_success=True,
    )
    reopened = transition_circuit_breaker(
        state=half_open,
        config=config,
        evaluated_at=now + timedelta(seconds=32),
        observed_success=False,
    )

    assert after_second_failure.breaker_state == "OPEN"
    assert after_second_failure.reason_code == BREAKER_OPENED
    assert half_open.breaker_state == "HALF_OPEN"
    assert half_open.reason_code == BREAKER_HALF_OPENED
    assert restored.breaker_state == "CLOSED"
    assert restored.reason_code == BREAKER_RESTORED
    assert reopened.breaker_state == "OPEN"
    assert reopened.reason_code == BREAKER_REOPENED


def test_pending_signal_expiry_rule_is_explicit() -> None:
    """Pending signals should expire only once they exceed the configured age."""
    signal_interval_begin = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)

    still_valid = evaluate_pending_signal_expiry(
        signal_interval_begin=signal_interval_begin,
        current_interval_begin=signal_interval_begin + timedelta(minutes=5),
        interval_minutes=5,
        max_age_intervals=1,
    )
    expired = evaluate_pending_signal_expiry(
        signal_interval_begin=signal_interval_begin,
        current_interval_begin=signal_interval_begin + timedelta(minutes=10),
        interval_minutes=5,
        max_age_intervals=1,
    )

    assert still_valid == (False, PENDING_SIGNAL_VALID)
    assert expired == (True, PENDING_SIGNAL_EXPIRED)
