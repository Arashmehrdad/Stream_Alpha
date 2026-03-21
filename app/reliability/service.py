"""Pure reliability helpers for freshness, breakers, and recovery."""

# pylint: disable=too-many-arguments,too-many-return-statements

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from math import floor
from typing import Sequence

from app.common.time import parse_rfc3339
from app.reliability.config import CircuitBreakerConfig
from app.reliability.schemas import (
    CircuitBreakerState,
    FeatureLagSnapshot,
    FreshnessStatus,
    HealthAggregation,
    RecoveryEvent,
    ServiceHeartbeat,
    ServiceHealthSnapshot,
    SystemReliabilitySnapshot,
)


FEED_FRESH = "FEED_FRESH"
FEED_STALE = "FEED_STALE"
HEARTBEAT_FRESH = "HEARTBEAT_FRESH"
HEARTBEAT_STALE = "HEARTBEAT_STALE"
HEARTBEAT_MISSING = "HEARTBEAT_MISSING"
FEATURE_FRESH = "FEATURE_FRESH"
FEATURE_STALE = "FEATURE_STALE"
FEATURE_ROW_MISSING = "FEATURE_ROW_MISSING"
FEATURE_INPUTS_MISSING = "FEATURE_INPUTS_MISSING"
FEATURE_TIME_LAG_OK = "FEATURE_TIME_LAG_OK"
FEATURE_TIME_LAG_BREACH = "FEATURE_TIME_LAG_BREACH"
FEATURE_PROCESSING_LAG_OK = "FEATURE_PROCESSING_LAG_OK"
FEATURE_PROCESSING_LAG_BREACH = "FEATURE_PROCESSING_LAG_BREACH"
FEATURE_LAG_OK = "FEATURE_LAG_OK"
FEATURE_LAG_BREACH = "FEATURE_LAG_BREACH"
FEATURE_LAG_MISSING_FEATURE_ROW = "FEATURE_LAG_MISSING_FEATURE_ROW"
REGIME_FRESH = "REGIME_FRESH"
REGIME_STALE = "REGIME_STALE"
REGIME_ROW_INCOMPATIBLE = "REGIME_ROW_INCOMPATIBLE"
HEALTH_HEALTHY = "HEALTH_HEALTHY"
HEALTH_DEGRADED_FRESHNESS = "HEALTH_DEGRADED_FRESHNESS"
HEALTH_UNAVAILABLE_BREAKER_OPEN = "HEALTH_UNAVAILABLE_BREAKER_OPEN"
SYSTEM_HEALTHY = "SYSTEM_HEALTHY"
SYSTEM_DEGRADED = "SYSTEM_DEGRADED"
SYSTEM_UNAVAILABLE = "SYSTEM_UNAVAILABLE"
BREAKER_OPENED = "BREAKER_OPENED"
BREAKER_HALF_OPENED = "BREAKER_HALF_OPENED"
BREAKER_RESTORED = "BREAKER_RESTORED"
BREAKER_REOPENED = "BREAKER_REOPENED"
BREAKER_CLOSED = "BREAKER_CLOSED"
SIGNAL_FETCH_FAILED = "SIGNAL_FETCH_FAILED"
SIGNAL_FETCH_SKIPPED_BREAKER_OPEN = "SIGNAL_FETCH_SKIPPED_BREAKER_OPEN"
RECOVERY_STALE_PENDING_SIGNAL_CLEARED = "RECOVERY_STALE_PENDING_SIGNAL_CLEARED"
RELIABILITY_HOLD_MISSING_FEATURE_ROW = "RELIABILITY_HOLD_MISSING_FEATURE_ROW"
RELIABILITY_HOLD_STALE_FEATURE_ROW = "RELIABILITY_HOLD_STALE_FEATURE_ROW"
RELIABILITY_HOLD_INPUTS_MISSING = "RELIABILITY_HOLD_INPUTS_MISSING"
SERVICE_HEARTBEAT_HEALTHY = "SERVICE_HEARTBEAT_HEALTHY"
SERVICE_HEARTBEAT_DEGRADED = "SERVICE_HEARTBEAT_DEGRADED"
PENDING_SIGNAL_VALID = "PENDING_SIGNAL_VALID"
PENDING_SIGNAL_EXPIRED = "PENDING_SIGNAL_EXPIRED"
FEATURE_LAG_BREACH_DETECTED = "FEATURE_LAG_BREACH_DETECTED"
FEATURE_LAG_BREACH_CLEARED = "FEATURE_LAG_BREACH_CLEARED"


def evaluate_feed_freshness(
    *,
    observed_at: datetime | None,
    evaluated_at: datetime,
    max_age_seconds: int,
) -> FreshnessStatus:
    """Evaluate raw-feed freshness with explicit reason codes."""
    return _evaluate_freshness(
        component_name="feed",
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        max_age_seconds=max_age_seconds,
        fresh_reason=FEED_FRESH,
        stale_reason=FEED_STALE,
    )


def evaluate_feature_freshness(
    *,
    observed_at: datetime | None,
    evaluated_at: datetime,
    max_age_seconds: int,
) -> FreshnessStatus:
    """Evaluate canonical feature freshness with explicit reason codes."""
    return _evaluate_freshness(
        component_name="feature_ohlc",
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        max_age_seconds=max_age_seconds,
        fresh_reason=FEATURE_FRESH,
        stale_reason=FEATURE_STALE,
    )


def evaluate_heartbeat_freshness(
    *,
    observed_at: datetime | None,
    evaluated_at: datetime,
    max_age_seconds: int,
) -> FreshnessStatus:
    """Evaluate component heartbeat freshness with explicit missing handling."""
    if observed_at is None:
        return FreshnessStatus(
            component_name="heartbeat",
            freshness_status="UNKNOWN",
            observed_at=None,
            evaluated_at=evaluated_at,
            age_seconds=None,
            max_age_seconds=max_age_seconds,
            reason_code=HEARTBEAT_MISSING,
            detail="No heartbeat has been written for this component",
        )
    return _evaluate_freshness(
        component_name="heartbeat",
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        max_age_seconds=max_age_seconds,
        fresh_reason=HEARTBEAT_FRESH,
        stale_reason=HEARTBEAT_STALE,
    )


def evaluate_regime_freshness(
    *,
    observed_at: datetime | None,
    evaluated_at: datetime,
    max_age_seconds: int,
    exact_row_resolved: bool,
    detail: str | None = None,
) -> FreshnessStatus:
    """Evaluate regime freshness from exact-row compatibility, not artifact age alone."""
    age_seconds = (
        None
        if observed_at is None
        else max(0.0, (evaluated_at - observed_at).total_seconds())
    )
    if not exact_row_resolved:
        return FreshnessStatus(
            component_name="regime",
            freshness_status="STALE" if observed_at is not None else "UNKNOWN",
            observed_at=observed_at,
            evaluated_at=evaluated_at,
            age_seconds=age_seconds,
            max_age_seconds=max_age_seconds,
            reason_code=REGIME_ROW_INCOMPATIBLE,
            detail=detail or "Exact-row regime resolution failed",
        )

    return FreshnessStatus(
        component_name="regime",
        freshness_status="FRESH",
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        age_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        reason_code=REGIME_FRESH,
        detail=detail or "Exact-row regime resolution succeeded",
    )


def evaluate_feature_consumer_lag(
    *,
    service_name: str,
    component_name: str,
    symbol: str,
    evaluated_at: datetime,
    latest_raw_event_received_at: datetime | None,
    latest_feature_interval_begin: datetime | None,
    latest_feature_as_of_time: datetime | None,
    feature_time_lag_max_seconds: int,
    consumer_processing_lag_max_seconds: int,
) -> FeatureLagSnapshot:
    """Evaluate finalized feature lag and consumer processing lag per symbol."""
    time_lag_seconds = (
        None
        if latest_feature_as_of_time is None
        else max(0.0, (evaluated_at - latest_feature_as_of_time).total_seconds())
    )
    processing_lag_seconds = (
        None
        if latest_raw_event_received_at is None or latest_feature_as_of_time is None
        else max(
            0.0,
            (latest_raw_event_received_at - latest_feature_as_of_time).total_seconds(),
        )
    )

    if latest_feature_as_of_time is None:
        return FeatureLagSnapshot(
            service_name=service_name,
            component_name=component_name,
            symbol=symbol,
            evaluated_at=evaluated_at,
            latest_raw_event_received_at=latest_raw_event_received_at,
            latest_feature_interval_begin=latest_feature_interval_begin,
            latest_feature_as_of_time=None,
            time_lag_seconds=None,
            processing_lag_seconds=processing_lag_seconds,
            time_lag_reason_code=FEATURE_TIME_LAG_BREACH,
            processing_lag_reason_code=FEATURE_PROCESSING_LAG_BREACH,
            lag_breach=True,
            health_overall_status="UNAVAILABLE",
            reason_code=FEATURE_LAG_MISSING_FEATURE_ROW,
            detail="No finalized canonical feature row is available for this symbol",
        )

    time_lag_breach = time_lag_seconds is not None and (
        time_lag_seconds > feature_time_lag_max_seconds
    )
    processing_lag_breach = processing_lag_seconds is not None and (
        processing_lag_seconds > consumer_processing_lag_max_seconds
    )
    lag_breach = time_lag_breach or processing_lag_breach
    return FeatureLagSnapshot(
        service_name=service_name,
        component_name=component_name,
        symbol=symbol,
        evaluated_at=evaluated_at,
        latest_raw_event_received_at=latest_raw_event_received_at,
        latest_feature_interval_begin=latest_feature_interval_begin,
        latest_feature_as_of_time=latest_feature_as_of_time,
        time_lag_seconds=time_lag_seconds,
        processing_lag_seconds=processing_lag_seconds,
        time_lag_reason_code=(
            FEATURE_TIME_LAG_BREACH if time_lag_breach else FEATURE_TIME_LAG_OK
        ),
        processing_lag_reason_code=(
            FEATURE_PROCESSING_LAG_BREACH
            if processing_lag_breach
            else FEATURE_PROCESSING_LAG_OK
        ),
        lag_breach=lag_breach,
        health_overall_status="DEGRADED" if lag_breach else "HEALTHY",
        reason_code=FEATURE_LAG_BREACH if lag_breach else FEATURE_LAG_OK,
        detail=(
            "feature_time_lag_seconds="
            f"{_format_lag(time_lag_seconds)}"
            f", processing_lag_seconds={_format_lag(processing_lag_seconds)}"
            f", thresholds=({feature_time_lag_max_seconds},"
            f" {consumer_processing_lag_max_seconds})"
        ),
    )


def aggregate_health_status(
    *,
    freshness_statuses: Sequence[FreshnessStatus],
    breaker_state: CircuitBreakerState | None = None,
) -> HealthAggregation:
    """Aggregate component freshness plus breaker state into one health status."""
    if breaker_state is not None and breaker_state.breaker_state == "OPEN":
        return HealthAggregation(
            health_overall_status="UNAVAILABLE",
            reason_codes=(HEALTH_UNAVAILABLE_BREAKER_OPEN,),
            freshness_statuses=tuple(freshness_statuses),
            breaker_state=breaker_state.breaker_state,
        )

    if any(status.freshness_status != "FRESH" for status in freshness_statuses):
        return HealthAggregation(
            health_overall_status="DEGRADED",
            reason_codes=(HEALTH_DEGRADED_FRESHNESS,),
            freshness_statuses=tuple(freshness_statuses),
            breaker_state=(
                None if breaker_state is None else breaker_state.breaker_state
            ),
        )

    return HealthAggregation(
        health_overall_status="HEALTHY",
        reason_codes=(HEALTH_HEALTHY,),
        freshness_statuses=tuple(freshness_statuses),
        breaker_state=None if breaker_state is None else breaker_state.breaker_state,
    )


def build_service_health_snapshot(
    *,
    service_name: str,
    component_name: str,
    heartbeat: ServiceHeartbeat | None,
    evaluated_at: datetime,
    heartbeat_stale_after_seconds: int,
    feed_max_age_seconds: int | None = None,
) -> ServiceHealthSnapshot:
    """Build one component-level health summary from the latest heartbeat."""
    heartbeat_freshness = evaluate_heartbeat_freshness(
        observed_at=None if heartbeat is None else heartbeat.heartbeat_at,
        evaluated_at=evaluated_at,
        max_age_seconds=heartbeat_stale_after_seconds,
    )
    if heartbeat is None:
        return ServiceHealthSnapshot(
            service_name=service_name,
            component_name=component_name,
            checked_at=evaluated_at,
            heartbeat_at=None,
            heartbeat_age_seconds=None,
            heartbeat_freshness_status=heartbeat_freshness.freshness_status,
            health_overall_status="UNAVAILABLE",
            reason_code=HEARTBEAT_MISSING,
            detail=heartbeat_freshness.detail,
        )

    health_overall_status = heartbeat.health_overall_status
    reason_code = heartbeat.reason_code
    detail = heartbeat.detail
    feed_freshness_status = None
    feed_reason_code = None
    feed_age_seconds = None

    if heartbeat_freshness.freshness_status != "FRESH":
        health_overall_status = "UNAVAILABLE"
        reason_code = heartbeat_freshness.reason_code
        detail = heartbeat_freshness.detail

    if (
        component_name == "producer"
        and feed_max_age_seconds is not None
        and heartbeat_freshness.freshness_status == "FRESH"
    ):
        exchange_activity_at = extract_last_exchange_activity(heartbeat.detail)
        feed_freshness = evaluate_feed_freshness(
            observed_at=exchange_activity_at,
            evaluated_at=evaluated_at,
            max_age_seconds=feed_max_age_seconds,
        )
        feed_freshness_status = feed_freshness.freshness_status
        feed_reason_code = feed_freshness.reason_code
        feed_age_seconds = feed_freshness.age_seconds
        if feed_freshness.freshness_status != "FRESH":
            health_overall_status = "DEGRADED"
            reason_code = feed_freshness.reason_code
            detail = feed_freshness.detail

    return ServiceHealthSnapshot(
        service_name=service_name,
        component_name=component_name,
        checked_at=evaluated_at,
        heartbeat_at=heartbeat.heartbeat_at,
        heartbeat_age_seconds=heartbeat_freshness.age_seconds,
        heartbeat_freshness_status=heartbeat_freshness.freshness_status,
        health_overall_status=health_overall_status,
        reason_code=reason_code,
        detail=detail,
        feed_freshness_status=feed_freshness_status,
        feed_reason_code=feed_reason_code,
        feed_age_seconds=feed_age_seconds,
    )


def build_signal_client_health_snapshot(
    *,
    service_name: str,
    component_name: str,
    state: CircuitBreakerState | None,
    evaluated_at: datetime,
    heartbeat_stale_after_seconds: int,
) -> ServiceHealthSnapshot:
    """Build the operator-facing signal-client health summary from breaker state."""
    if state is None:
        return ServiceHealthSnapshot(
            service_name=service_name,
            component_name=component_name,
            checked_at=evaluated_at,
            heartbeat_at=None,
            heartbeat_age_seconds=None,
            heartbeat_freshness_status="UNKNOWN",
            health_overall_status="UNAVAILABLE",
            reason_code=HEARTBEAT_MISSING,
            detail="No signal-client reliability state has been written",
        )

    reference_time = state.last_heartbeat_at or state.updated_at
    heartbeat_freshness = evaluate_heartbeat_freshness(
        observed_at=reference_time,
        evaluated_at=evaluated_at,
        max_age_seconds=heartbeat_stale_after_seconds,
    )
    if heartbeat_freshness.freshness_status != "FRESH":
        return ServiceHealthSnapshot(
            service_name=service_name,
            component_name=component_name,
            checked_at=evaluated_at,
            heartbeat_at=reference_time,
            heartbeat_age_seconds=heartbeat_freshness.age_seconds,
            heartbeat_freshness_status=heartbeat_freshness.freshness_status,
            health_overall_status="UNAVAILABLE",
            reason_code=heartbeat_freshness.reason_code,
            detail=heartbeat_freshness.detail,
        )

    return ServiceHealthSnapshot(
        service_name=service_name,
        component_name=component_name,
        checked_at=evaluated_at,
        heartbeat_at=reference_time,
        heartbeat_age_seconds=heartbeat_freshness.age_seconds,
        heartbeat_freshness_status=heartbeat_freshness.freshness_status,
        health_overall_status=state.health_overall_status,
        reason_code=state.reason_code or HEALTH_HEALTHY,
        detail=state.detail,
    )


def aggregate_system_reliability(
    *,
    service_name: str,
    evaluated_at: datetime,
    services: Sequence[ServiceHealthSnapshot],
    lag_by_symbol: Sequence[FeatureLagSnapshot],
    latest_recovery_event: RecoveryEvent | None,
) -> SystemReliabilitySnapshot:
    """Aggregate per-service health plus lag snapshots into one canonical summary."""
    lag_breach_active = any(snapshot.lag_breach for snapshot in lag_by_symbol)
    unhealthy_reason_codes = list(
        _unique_reason_codes(
            [
                snapshot.reason_code
                for snapshot in services
                if snapshot.health_overall_status != "HEALTHY"
            ]
            + [
                snapshot.reason_code
                for snapshot in lag_by_symbol
                if snapshot.lag_breach
            ]
        )
    )
    if any(snapshot.health_overall_status == "UNAVAILABLE" for snapshot in services):
        health_overall_status = "UNAVAILABLE"
        if not unhealthy_reason_codes:
            unhealthy_reason_codes.append(SYSTEM_UNAVAILABLE)
    elif (
        any(snapshot.health_overall_status == "DEGRADED" for snapshot in services)
        or lag_breach_active
    ):
        health_overall_status = "DEGRADED"
        if not unhealthy_reason_codes:
            unhealthy_reason_codes.append(SYSTEM_DEGRADED)
    else:
        health_overall_status = "HEALTHY"
        unhealthy_reason_codes = [SYSTEM_HEALTHY]

    return SystemReliabilitySnapshot(
        service_name=service_name,
        checked_at=evaluated_at,
        health_overall_status=health_overall_status,
        reason_codes=tuple(unhealthy_reason_codes),
        lag_breach_active=lag_breach_active,
        services=tuple(services),
        lag_by_symbol=tuple(lag_by_symbol),
        latest_recovery_event=latest_recovery_event,
    )


def transition_circuit_breaker(
    *,
    state: CircuitBreakerState,
    config: CircuitBreakerConfig,
    evaluated_at: datetime,
    observed_success: bool | None,
) -> CircuitBreakerState:
    """Apply one explicit CLOSED/OPEN/HALF_OPEN breaker transition."""
    if state.breaker_state == "OPEN":
        if state.opened_at is None:
            return replace(
                state,
                reason_code=BREAKER_OPENED,
                updated_at=evaluated_at,
            )
        open_duration = (evaluated_at - state.opened_at).total_seconds()
        if open_duration < config.half_open_after_seconds:
            return replace(
                state,
                reason_code=BREAKER_OPENED,
                updated_at=evaluated_at,
            )
        if observed_success is None:
            return replace(
                state,
                breaker_state="HALF_OPEN",
                success_count=0,
                reason_code=BREAKER_HALF_OPENED,
                updated_at=evaluated_at,
            )

    if state.breaker_state == "CLOSED":
        if observed_success is False:
            failure_count = state.failure_count + 1
            if failure_count >= config.failure_threshold:
                return replace(
                    state,
                    breaker_state="OPEN",
                    failure_count=failure_count,
                    success_count=0,
                    last_failure_at=evaluated_at,
                    opened_at=evaluated_at,
                    reason_code=BREAKER_OPENED,
                    updated_at=evaluated_at,
                )
            return replace(
                state,
                failure_count=failure_count,
                success_count=0,
                last_failure_at=evaluated_at,
                reason_code=BREAKER_CLOSED,
                updated_at=evaluated_at,
            )
        if observed_success is True:
            return replace(
                state,
                failure_count=0,
                success_count=0,
                last_success_at=evaluated_at,
                reason_code=BREAKER_CLOSED,
                updated_at=evaluated_at,
            )
        return replace(state, updated_at=evaluated_at)

    if state.breaker_state == "HALF_OPEN":
        if observed_success is True:
            success_count = state.success_count + 1
            if success_count >= config.success_threshold:
                return replace(
                    state,
                    breaker_state="CLOSED",
                    failure_count=0,
                    success_count=0,
                    last_success_at=evaluated_at,
                    opened_at=None,
                    reason_code=BREAKER_RESTORED,
                    updated_at=evaluated_at,
                )
            return replace(
                state,
                success_count=success_count,
                last_success_at=evaluated_at,
                reason_code=BREAKER_HALF_OPENED,
                updated_at=evaluated_at,
            )
        if observed_success is False:
            return replace(
                state,
                breaker_state="OPEN",
                failure_count=1,
                success_count=0,
                last_failure_at=evaluated_at,
                opened_at=evaluated_at,
                reason_code=BREAKER_REOPENED,
                updated_at=evaluated_at,
            )
        return replace(state, updated_at=evaluated_at)

    return replace(state, updated_at=evaluated_at)


def evaluate_pending_signal_expiry(
    *,
    signal_interval_begin: datetime,
    current_interval_begin: datetime,
    interval_minutes: int,
    max_age_intervals: int,
) -> tuple[bool, str]:
    """Return whether a persisted pending signal is stale and why."""
    elapsed_minutes = (current_interval_begin - signal_interval_begin).total_seconds() / 60.0
    age_intervals = floor(elapsed_minutes / float(interval_minutes))
    if age_intervals > max_age_intervals:
        return True, PENDING_SIGNAL_EXPIRED
    return False, PENDING_SIGNAL_VALID


def _evaluate_freshness(
    *,
    component_name: str,
    observed_at: datetime | None,
    evaluated_at: datetime,
    max_age_seconds: int,
    fresh_reason: str,
    stale_reason: str,
) -> FreshnessStatus:
    if observed_at is None:
        return FreshnessStatus(
            component_name=component_name,
            freshness_status="UNKNOWN",
            observed_at=None,
            evaluated_at=evaluated_at,
            age_seconds=None,
            max_age_seconds=max_age_seconds,
            reason_code=stale_reason,
            detail=f"{component_name} freshness cannot be evaluated without a timestamp",
        )

    age_seconds = max(0.0, (evaluated_at - observed_at).total_seconds())
    freshness_status = "FRESH" if age_seconds <= max_age_seconds else "STALE"
    return FreshnessStatus(
        component_name=component_name,
        freshness_status=freshness_status,
        observed_at=observed_at,
        evaluated_at=evaluated_at,
        age_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        reason_code=fresh_reason if freshness_status == "FRESH" else stale_reason,
        detail=(
            f"{component_name} age_seconds={age_seconds:.3f} "
            f"threshold={max_age_seconds}"
        ),
    )


def extract_last_exchange_activity(detail: str | None) -> datetime | None:
    """Extract the latest exchange-activity timestamp from a producer heartbeat."""
    if detail is None:
        return None
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None
    raw_value = payload.get("last_exchange_activity_at")
    if not isinstance(raw_value, str):
        return None
    try:
        return parse_rfc3339(raw_value)
    except ValueError:
        return None


def _unique_reason_codes(reason_codes: Sequence[str]) -> tuple[str, ...]:
    unique_codes: list[str] = []
    for reason_code in reason_codes:
        if reason_code not in unique_codes:
            unique_codes.append(reason_code)
    return tuple(unique_codes)


def _format_lag(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"
