"""Typed M17 alerting schemas and artifact payloads."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


AlertCategory = Literal[
    "FEED_STALE",
    "CONSUMER_LAG",
    "ORDER_FAILURE_SPIKE",
    "DRAWDOWN_BREACH",
    "SIGNAL_SILENCE",
    "SIGNAL_FLOOD",
    "LIVE_MODE_ACTIVATION",
    "STARTUP_SAFETY",
]
AlertSeverity = Literal["INFO", "WARNING", "CRITICAL"]
AlertEventState = Literal["OPEN", "UPDATED", "CLEARED", "INFO"]


@dataclass(frozen=True, slots=True)
class OperationalAlertEvent:
    """Canonical M17 incident timeline event."""

    service_name: str
    execution_mode: str
    category: AlertCategory
    severity: AlertSeverity
    event_state: AlertEventState
    reason_code: str
    source_component: str
    fingerprint: str
    summary_text: str
    event_time: datetime
    symbol: str | None = None
    detail: str | None = None
    related_order_request_id: int | None = None
    related_decision_trace_id: int | None = None
    payload_json: dict[str, Any] = field(default_factory=dict)
    event_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OperationalAlertState:
    """Canonical M17 current-state row keyed by fingerprint."""

    fingerprint: str
    service_name: str
    execution_mode: str
    category: AlertCategory
    source_component: str
    is_active: bool
    severity: AlertSeverity
    reason_code: str
    opened_at: datetime
    last_seen_at: datetime
    symbol: str | None = None
    last_event_id: int | None = None
    occurrence_count: int = 1


@dataclass(frozen=True, slots=True)
class StartupSafetySection:
    """Inspectable startup-safety sub-section."""

    report_path: str
    report_exists: bool
    startup_validation_passed: bool | None = None
    checklist_path: str | None = None
    checklist_exists: bool = False
    checklist_passed: bool | None = None
    primary_reason_code: str | None = None
    summary_text: str | None = None
    detail: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StartupSafetyReport:
    """Canonical M17 startup-safety artifact payload."""

    schema_version: str
    generated_at: str
    service_name: str
    execution_mode: str
    runtime_profile: str
    startup_safety_passed: bool
    primary_reason_code: str
    summary_text: str
    startup_validation: StartupSafetySection
    live_startup: StartupSafetySection


@dataclass(frozen=True, slots=True)
class DailyOperationsSummary:
    """Canonical M17 daily operations summary artifact payload."""

    schema_version: str
    generated_at: str
    service_name: str
    execution_mode: str
    runtime_profile: str
    summary_date: str
    counts_by_category: dict[str, int]
    unresolved_count: int
    highest_severity: AlertSeverity
    startup_safety_status: dict[str, Any]
    order_failure_counts: dict[str, Any]
    drawdown_state: dict[str, Any]
    actionable_signal_counts: dict[str, Any]
    silence_flood_episodes: dict[str, int]
    live_mode_activation_count: int
