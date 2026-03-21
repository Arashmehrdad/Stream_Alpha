"""Pure reliability helpers for freshness, breakers, and recovery."""

# pylint: disable=too-many-arguments,too-many-return-statements

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import floor
from typing import Sequence

from app.reliability.config import CircuitBreakerConfig
from app.reliability.schemas import (
    CircuitBreakerState,
    FreshnessStatus,
    HealthAggregation,
)


FEED_FRESH = "FEED_FRESH"
FEED_STALE = "FEED_STALE"
FEATURE_FRESH = "FEATURE_FRESH"
FEATURE_STALE = "FEATURE_STALE"
FEATURE_ROW_MISSING = "FEATURE_ROW_MISSING"
FEATURE_INPUTS_MISSING = "FEATURE_INPUTS_MISSING"
REGIME_FRESH = "REGIME_FRESH"
REGIME_STALE = "REGIME_STALE"
REGIME_ROW_INCOMPATIBLE = "REGIME_ROW_INCOMPATIBLE"
HEALTH_HEALTHY = "HEALTH_HEALTHY"
HEALTH_DEGRADED_FRESHNESS = "HEALTH_DEGRADED_FRESHNESS"
HEALTH_UNAVAILABLE_BREAKER_OPEN = "HEALTH_UNAVAILABLE_BREAKER_OPEN"
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
