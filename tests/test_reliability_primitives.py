"""Pure primitive tests for the M13 reliability foundation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.reliability.config import CircuitBreakerConfig
from app.reliability.schemas import CircuitBreakerState, RecoveryEvent, ServiceHeartbeat
from app.reliability.service import (
    BREAKER_HALF_OPENED,
    BREAKER_OPENED,
    BREAKER_REOPENED,
    BREAKER_RESTORED,
    FEED_FRESH,
    FEED_STALE,
    FEATURE_LAG_BREACH,
    FEATURE_LAG_MISSING_FEATURE_ROW,
    FEATURE_PROCESSING_LAG_BREACH,
    FEATURE_STALE,
    HEALTH_DEGRADED_FRESHNESS,
    HEALTH_UNAVAILABLE_BREAKER_OPEN,
    HEARTBEAT_MISSING,
    PENDING_SIGNAL_EXPIRED,
    PENDING_SIGNAL_VALID,
    REGIME_FRESH,
    aggregate_health_status,
    aggregate_system_reliability,
    build_service_health_snapshot,
    evaluate_feed_freshness,
    evaluate_feature_consumer_lag,
    evaluate_feature_freshness,
    evaluate_heartbeat_freshness,
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
        exact_row_resolved=True,
    )
    regime_incompatible = evaluate_regime_freshness(
        observed_at=now - timedelta(seconds=1),
        evaluated_at=now,
        max_age_seconds=86400,
        exact_row_resolved=False,
        detail="thresholds did not match the exact row",
    )

    assert feed_status.freshness_status == "FRESH"
    assert feed_status.reason_code == FEED_FRESH
    assert feature_status.freshness_status == "STALE"
    assert feature_status.reason_code == FEATURE_STALE
    assert regime_status.freshness_status == "FRESH"
    assert regime_status.reason_code == REGIME_FRESH
    assert regime_incompatible.freshness_status == "STALE"


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


def test_heartbeat_and_feed_freshness_are_visible_in_service_snapshot() -> None:
    """Producer service snapshots should surface heartbeat and feed staleness explicitly."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    missing = evaluate_heartbeat_freshness(
        observed_at=None,
        evaluated_at=now,
        max_age_seconds=45,
    )
    producer_snapshot = build_service_health_snapshot(
        service_name="producer",
        component_name="producer",
        heartbeat=ServiceHeartbeat(
            service_name="producer",
            component_name="producer",
            heartbeat_at=now,
            health_overall_status="HEALTHY",
            reason_code="SERVICE_HEARTBEAT_HEALTHY",
            detail='{"last_exchange_activity_at":"2026-03-21T11:57:00Z"}',
        ),
        evaluated_at=now,
        heartbeat_stale_after_seconds=45,
        feed_max_age_seconds=90,
    )

    assert missing.reason_code == HEARTBEAT_MISSING
    assert producer_snapshot.health_overall_status == "DEGRADED"
    assert producer_snapshot.feed_reason_code == FEED_STALE


def test_feature_consumer_lag_reports_breach_and_missing_rows_explicitly() -> None:
    """Lag evaluation should expose both time lag and processing lag reason codes."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    lag_snapshot = evaluate_feature_consumer_lag(
        service_name="features",
        component_name="features",
        symbol="BTC/USD",
        evaluated_at=now,
        latest_raw_event_received_at=now,
        latest_feature_interval_begin=now - timedelta(minutes=5),
        latest_feature_as_of_time=now - timedelta(minutes=10),
        feature_time_lag_max_seconds=390,
        consumer_processing_lag_max_seconds=390,
    )
    missing_snapshot = evaluate_feature_consumer_lag(
        service_name="features",
        component_name="features",
        symbol="ETH/USD",
        evaluated_at=now,
        latest_raw_event_received_at=now,
        latest_feature_interval_begin=None,
        latest_feature_as_of_time=None,
        feature_time_lag_max_seconds=390,
        consumer_processing_lag_max_seconds=390,
    )

    assert lag_snapshot.reason_code == FEATURE_LAG_BREACH
    assert lag_snapshot.processing_lag_reason_code == FEATURE_PROCESSING_LAG_BREACH
    assert missing_snapshot.reason_code == FEATURE_LAG_MISSING_FEATURE_ROW
    assert missing_snapshot.health_overall_status == "UNAVAILABLE"


def test_system_reliability_aggregation_keeps_reason_codes_explicit() -> None:
    """Cross-service reliability aggregation should preserve degraded reasons and lag state."""
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    service_snapshot = build_service_health_snapshot(
        service_name="producer",
        component_name="producer",
        heartbeat=ServiceHeartbeat(
            service_name="producer",
            component_name="producer",
            heartbeat_at=now,
            health_overall_status="HEALTHY",
            reason_code="SERVICE_HEARTBEAT_HEALTHY",
            detail='{"last_exchange_activity_at":"2026-03-21T12:00:00Z"}',
        ),
        evaluated_at=now,
        heartbeat_stale_after_seconds=45,
        feed_max_age_seconds=90,
    )
    lag_snapshot = evaluate_feature_consumer_lag(
        service_name="features",
        component_name="features",
        symbol="BTC/USD",
        evaluated_at=now,
        latest_raw_event_received_at=now,
        latest_feature_interval_begin=now - timedelta(minutes=5),
        latest_feature_as_of_time=now - timedelta(minutes=1),
        feature_time_lag_max_seconds=390,
        consumer_processing_lag_max_seconds=30,
    )
    snapshot = aggregate_system_reliability(
        service_name="streamalpha",
        evaluated_at=now,
        services=(service_snapshot,),
        lag_by_symbol=(lag_snapshot,),
        latest_recovery_event=RecoveryEvent(
            service_name="features",
            component_name="BTC/USD",
            event_type="FEATURE_LAG_TRANSITION",
            event_time=now,
            reason_code="FEATURE_LAG_BREACH_DETECTED",
            health_overall_status="DEGRADED",
        ),
    )

    assert snapshot.health_overall_status == "DEGRADED"
    assert snapshot.lag_breach_active is True
    assert snapshot.reason_codes == (FEATURE_LAG_BREACH,)
