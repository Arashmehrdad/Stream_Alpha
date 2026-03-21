"""Typed reliability dataclasses for the M13 foundation."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


FreshnessLevel = Literal["FRESH", "STALE", "UNKNOWN"]
HealthOverallStatus = Literal["HEALTHY", "DEGRADED", "UNAVAILABLE"]
BreakerStatus = Literal["CLOSED", "OPEN", "HALF_OPEN"]


@dataclass(frozen=True, slots=True)
class FreshnessStatus:
    """Inspectable freshness evaluation result for one component."""

    component_name: str
    freshness_status: FreshnessLevel
    evaluated_at: datetime
    max_age_seconds: int
    reason_code: str
    observed_at: datetime | None = None
    age_seconds: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ServiceHeartbeat:
    """Persisted heartbeat row for one service component."""

    service_name: str
    component_name: str
    heartbeat_at: datetime
    health_overall_status: HealthOverallStatus
    reason_code: str
    detail: str | None = None
    heartbeat_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CircuitBreakerState:
    """Persisted reliability and breaker state for one component."""

    service_name: str
    component_name: str
    breaker_state: BreakerStatus
    health_overall_status: HealthOverallStatus
    failure_count: int = 0
    success_count: int = 0
    freshness_status: FreshnessLevel | None = None
    last_heartbeat_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    opened_at: datetime | None = None
    reason_code: str | None = None
    detail: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReliabilityState:
    """Persisted reliability state row exposed through PostgreSQL."""

    service_name: str
    component_name: str
    health_overall_status: HealthOverallStatus
    breaker_state: BreakerStatus
    failure_count: int = 0
    success_count: int = 0
    freshness_status: FreshnessLevel | None = None
    last_heartbeat_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    opened_at: datetime | None = None
    reason_code: str | None = None
    detail: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    """Explicit recovery or reliability event audit row."""

    service_name: str
    component_name: str
    event_type: str
    event_time: datetime
    reason_code: str
    health_overall_status: HealthOverallStatus | None = None
    freshness_status: FreshnessLevel | None = None
    breaker_state: BreakerStatus | None = None
    detail: str | None = None
    event_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class HealthAggregation:
    """Aggregated health snapshot built from freshness and breaker primitives."""

    health_overall_status: HealthOverallStatus
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    freshness_statuses: tuple[FreshnessStatus, ...] = field(default_factory=tuple)
    breaker_state: BreakerStatus | None = None


@dataclass(frozen=True, slots=True)
class SymbolFreshnessSnapshot:
    """Exact-row freshness summary for one symbol."""

    symbol: str
    row_id: str | None
    interval_begin: datetime | None
    as_of_time: datetime | None
    health_overall_status: HealthOverallStatus
    freshness_status: FreshnessLevel
    reason_code: str
    feature_freshness: FreshnessStatus
    regime_freshness: FreshnessStatus


@dataclass(frozen=True, slots=True)
class ReliabilityHealthSnapshot:
    """Inspectable overall health artifact payload."""

    service_name: str
    checked_at: datetime
    health_overall_status: HealthOverallStatus
    reason_code: str
    freshness_status: FreshnessLevel
    symbols: tuple[SymbolFreshnessSnapshot, ...] = field(default_factory=tuple)
