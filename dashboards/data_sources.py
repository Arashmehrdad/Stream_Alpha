"""Read-only HTTP and PostgreSQL data sources for the Stream Alpha M6 dashboard."""

# pylint: disable=duplicate-code,too-few-public-methods,too-many-instance-attributes
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from typing import Any, Mapping, Sequence

import asyncpg
import httpx

from app.common.config import Settings
from app.common.time import parse_rfc3339, utc_now
from app.explainability.schemas import DecisionTracePayload
from app.trading.config import PaperTradingConfig
from app.trading.repository import (
    DECISION_TRACES_TABLE,
    LEDGER_TABLE,
    LIVE_SAFETY_TABLE,
    ORDER_EVENTS_TABLE,
    POSITIONS_TABLE,
    RELIABILITY_EVENTS_TABLE,
    RELIABILITY_STATE_TABLE,
    STATE_TABLE,
)
from app.trading.schemas import PaperPosition


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HTTP_TIMEOUT_SECONDS = 5.0


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _quote_table_name(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {name}")
    return ".".join(_quote_identifier(part) for part in parts)


@dataclass(frozen=True, slots=True)
class ApiHealthSnapshot:
    """Current availability snapshot for the accepted M4 inference API."""

    available: bool
    checked_at: datetime
    status: str
    service: str | None = None
    runtime_profile: str | None = None
    model_loaded: bool = False
    model_name: str | None = None
    model_artifact_path: str | None = None
    regime_loaded: bool = False
    regime_run_id: str | None = None
    regime_artifact_path: str | None = None
    database: str | None = None
    started_at: datetime | None = None
    health_overall_status: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    active_alert_count: int | None = None
    max_alert_severity: str | None = None
    startup_safety_status: str | None = None
    startup_safety_reason_code: str | None = None
    active_adaptation_count: int | None = None
    adaptation_status: str | None = None
    adaptation_evidence_backed: bool | None = None
    ensemble_profile_id: str | None = None
    ensemble_status: str | None = None
    ensemble_candidate_count: int | None = None
    ensemble_roster_status: str | None = None
    ensemble_roster_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    active_continual_learning_profile_id: str | None = None
    continual_learning_status: str | None = None
    continual_learning_drift_cap_status: str | None = None
    continual_learning_evidence_backed: bool | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SignalSnapshot:
    """Current M4 signal payload or degraded state for one configured symbol."""

    symbol: str
    checked_at: datetime
    available: bool
    signal: str | None = None
    reason: str | None = None
    prob_up: float | None = None
    prob_down: float | None = None
    confidence: float | None = None
    predicted_class: str | None = None
    row_id: str | None = None
    as_of_time: datetime | None = None
    model_name: str | None = None
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None
    buy_threshold: float | None = None
    sell_threshold: float | None = None
    signal_status: str | None = None
    decision_source: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    health_overall_status: str | None = None
    adaptation_profile_id: str | None = None
    calibrated_confidence: float | None = None
    adaptive_size_multiplier: float | None = None
    drift_status: str | None = None
    frozen_by_health_gate: bool = False
    ensemble_profile_id: str | None = None
    ensemble_active: bool = False
    ensemble_candidate_count: int | None = None
    ensemble_fallback_reason: str | None = None
    ensemble_agreement_band: str | None = None
    ensemble_effective_confidence: float | None = None
    ensemble_roster_status: str | None = None
    ensemble_roster_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    continual_learning_status: str | None = None
    continual_learning_profile_id: str | None = None
    continual_learning_evidence_backed: bool | None = None
    continual_learning_frozen: bool = False
    continual_learning_candidate_type: str | None = None
    continual_learning_promotion_stage: str | None = None
    continual_learning_baseline_target_type: str | None = None
    continual_learning_baseline_target_id: str | None = None
    continual_learning_drift_cap_status: str | None = None
    continual_learning_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FreshnessSnapshot:
    """Exact-row freshness snapshot from the M13 `/freshness` endpoint."""

    symbol: str
    checked_at: datetime
    available: bool
    row_id: str | None = None
    interval_begin: datetime | None = None
    as_of_time: datetime | None = None
    health_overall_status: str | None = None
    freshness_status: str | None = None
    reason_code: str | None = None
    feature_freshness_status: str | None = None
    feature_reason_code: str | None = None
    feature_age_seconds: float | None = None
    regime_freshness_status: str | None = None
    regime_reason_code: str | None = None
    regime_age_seconds: float | None = None
    detail: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LatestFeatureSnapshot:  # pylint: disable=too-many-instance-attributes
    """Stable feature whitelist shown in the M6 dashboard."""

    symbol: str
    interval_begin: datetime
    as_of_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    log_return_1: float
    log_return_3: float
    rsi_14: float
    macd_line_12_26: float
    close_zscore_12: float
    volume_zscore_12: float


@dataclass(frozen=True, slots=True)
class LedgerEntrySnapshot:
    """Read-only recent ledger row for the trading tab."""

    ledger_id: int
    symbol: str
    action: str
    reason: str
    fill_interval_begin: datetime
    fill_time: datetime
    fill_price: float
    quantity: float
    notional: float
    fee: float
    cash_flow: float
    realized_pnl: float | None
    signal_row_id: str | None
    model_name: str | None
    confidence: float | None
    regime_label: str | None
    decision_trace_id: int | None = None


@dataclass(frozen=True, slots=True)
class OrderAuditSnapshot:
    """Read-only recent order lifecycle row for the trading tab."""

    event_id: int
    order_request_id: int
    symbol: str
    action: str
    lifecycle_state: str
    event_time: datetime
    reason_code: str | None
    details: str | None
    external_order_id: str | None = None
    external_status: str | None = None
    account_id: str | None = None
    environment_name: str | None = None
    broker_name: str | None = None
    probe_policy_active: bool = False
    probe_symbol: str | None = None
    probe_qty: int | None = None
    decision_trace_id: int | None = None


@dataclass(frozen=True, slots=True)
class LiveSafetySnapshot:
    """Read-only guarded-live execution safety state."""

    service_name: str
    execution_mode: str
    broker_name: str
    live_enabled: bool
    startup_checks_passed: bool
    startup_checks_passed_at: datetime | None
    account_validated: bool
    account_id: str | None
    environment_name: str | None
    manual_disable_active: bool
    consecutive_live_failures: int
    failure_hard_stop_active: bool
    last_failure_reason: str | None
    updated_at: datetime
    system_health_status: str | None = None
    system_health_reason_code: str | None = None
    system_health_checked_at: datetime | None = None
    health_gate_status: str | None = None
    health_gate_reason_code: str | None = None
    health_gate_detail: str | None = None
    broker_cash: float | None = None
    broker_equity: float | None = None
    reconciliation_status: str | None = None
    reconciliation_reason_code: str | None = None
    reconciliation_checked_at: datetime | None = None
    unresolved_incident_count: int = 0
    can_submit_live_now: bool = False
    primary_block_reason_code: str | None = None
    block_detail: str | None = None


@dataclass(frozen=True, slots=True)
class ReliabilityStateSnapshot:
    """Read-only trader-side reliability breaker state."""

    service_name: str
    component_name: str
    health_overall_status: str
    freshness_status: str | None
    breaker_state: str
    failure_count: int
    success_count: int
    reason_code: str | None
    detail: str | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryEventSnapshot:
    """Latest persisted reliability or recovery event."""

    service_name: str
    component_name: str
    event_type: str
    event_time: datetime
    reason_code: str
    health_overall_status: str | None = None
    freshness_status: str | None = None
    breaker_state: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ServiceHealthSummarySnapshot:
    """Per-service status from the canonical reliability summary endpoint."""

    service_name: str
    component_name: str
    checked_at: datetime
    heartbeat_at: datetime | None
    heartbeat_age_seconds: float | None
    heartbeat_freshness_status: str
    health_overall_status: str
    reason_code: str
    detail: str | None = None
    feed_freshness_status: str | None = None
    feed_reason_code: str | None = None
    feed_age_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class FeatureLagSummarySnapshot:
    """Per-symbol feature lag status from the canonical reliability summary."""

    service_name: str
    component_name: str
    symbol: str
    evaluated_at: datetime
    latest_raw_event_received_at: datetime | None
    latest_feature_interval_begin: datetime | None
    latest_feature_as_of_time: datetime | None
    time_lag_seconds: float | None
    processing_lag_seconds: float | None
    time_lag_reason_code: str
    processing_lag_reason_code: str
    lag_breach: bool
    health_overall_status: str
    reason_code: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SystemReliabilitySnapshot:
    """Canonical cross-service reliability snapshot from the API."""

    available: bool
    checked_at: datetime
    service_name: str | None = None
    health_overall_status: str | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    lag_breach_active: bool | None = None
    services: tuple[ServiceHealthSummarySnapshot, ...] = field(default_factory=tuple)
    lag_by_symbol: tuple[FeatureLagSummarySnapshot, ...] = field(default_factory=tuple)
    latest_recovery_event: RecoveryEventSnapshot | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveAlertSnapshot:
    """Current active M17 alert state from the API."""

    fingerprint: str
    service_name: str
    execution_mode: str
    category: str
    severity: str
    reason_code: str
    source_component: str
    is_active: bool
    opened_at: datetime
    last_seen_at: datetime
    symbol: str | None = None
    last_event_id: int | None = None
    occurrence_count: int = 0


@dataclass(frozen=True, slots=True)
class ActiveAlertsSnapshot:
    """Read-only M17 active-alert collection from the API."""

    available: bool
    checked_at: datetime
    items: tuple[ActiveAlertSnapshot, ...] = field(default_factory=tuple)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AlertTimelineEventSnapshot:
    """Canonical M17 timeline event from the API."""

    event_id: int | None
    service_name: str
    execution_mode: str
    category: str
    severity: str
    event_state: str
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
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AlertTimelineSnapshot:
    """Read-only M17 timeline collection from the API."""

    available: bool
    checked_at: datetime
    items: tuple[AlertTimelineEventSnapshot, ...] = field(default_factory=tuple)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StartupSafetySnapshot:
    """Canonical M17 startup-safety artifact snapshot from the API."""

    available: bool
    checked_at: datetime
    generated_at: datetime | None = None
    service_name: str | None = None
    execution_mode: str | None = None
    runtime_profile: str | None = None
    startup_safety_passed: bool | None = None
    primary_reason_code: str | None = None
    summary_text: str | None = None
    startup_report_path: str | None = None
    startup_validation_passed: bool | None = None
    live_startup_reason_code: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DailyOperationsSummarySnapshot:
    """Canonical M17 daily operations summary artifact snapshot from the API."""

    available: bool
    checked_at: datetime
    generated_at: datetime | None = None
    service_name: str | None = None
    execution_mode: str | None = None
    runtime_profile: str | None = None
    summary_date: str | None = None
    unresolved_count: int | None = None
    highest_severity: str | None = None
    counts_by_category: dict[str, int] = field(default_factory=dict)
    startup_safety_status: dict[str, Any] = field(default_factory=dict)
    order_failure_counts: dict[str, Any] = field(default_factory=dict)
    drawdown_state: dict[str, Any] = field(default_factory=dict)
    actionable_signal_counts: dict[str, Any] = field(default_factory=dict)
    silence_flood_episodes: dict[str, int] = field(default_factory=dict)
    live_mode_activation_count: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AdaptationSummarySnapshot:
    """Read-only M19 adaptation summary snapshot from the API."""

    available: bool
    checked_at: datetime
    enabled: bool = False
    active_profile_count: int = 0
    active_profile_id: str | None = None
    adaptation_status: str | None = None
    evidence_backed: bool = False
    frozen_by_health_gate: bool = False
    latest_drift_status: str | None = None
    latest_drift_updated_at: datetime | None = None
    latest_promotion_decision: str | None = None
    latest_performance_window_id: str | None = None
    latest_performance_trade_count: int | None = None
    latest_performance_created_at: datetime | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AdaptationDriftItemSnapshot:
    """Read-only adaptive drift row for operator visibility."""

    symbol: str
    regime_label: str
    detector_name: str
    window_id: str
    drift_score: float
    status: str
    reason_code: str
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AdaptationPerformanceItemSnapshot:
    """Read-only adaptive performance row for operator visibility."""

    execution_mode: str
    symbol: str
    regime_label: str
    window_id: str
    window_type: str
    trade_count: int
    net_pnl_after_costs: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    blocked_trade_rate: float
    shadow_divergence_rate: float


@dataclass(frozen=True, slots=True)
class AdaptationProfileItemSnapshot:
    """Read-only adaptive profile row for operator visibility."""

    profile_id: str
    status: str
    execution_mode_scope: str
    symbol_scope: str
    regime_scope: str
    rollback_target_profile_id: str | None = None
    activated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AdaptationPromotionItemSnapshot:
    """Read-only adaptive promotion decision row for operator visibility."""

    decision_id: str
    target_type: str
    target_id: str
    decision: str
    summary_text: str
    decided_at: datetime
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AdaptationSnapshot:
    """Combined read-only M19 adaptation snapshot from the API."""

    summary: AdaptationSummarySnapshot
    drift: tuple[AdaptationDriftItemSnapshot, ...] = field(default_factory=tuple)
    performance: tuple[AdaptationPerformanceItemSnapshot, ...] = field(default_factory=tuple)
    profiles: tuple[AdaptationProfileItemSnapshot, ...] = field(default_factory=tuple)
    promotions: tuple[AdaptationPromotionItemSnapshot, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContinualLearningSummarySnapshot:
    """Read-only M21 continual-learning summary snapshot from the API."""

    available: bool
    checked_at: datetime
    enabled: bool = False
    active_profile_count: int = 0
    active_profile_id: str | None = None
    continual_learning_status: str | None = None
    evidence_backed: bool = False
    latest_drift_cap_status: str | None = None
    latest_drift_cap_updated_at: datetime | None = None
    latest_promotion_decision: str | None = None
    latest_event_type: str | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    aggregated_scope: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ContinualLearningProfileItemSnapshot:
    """Read-only M21 profile row for operator visibility."""

    profile_id: str
    status: str
    candidate_type: str
    execution_mode_scope: str
    symbol_scope: str
    regime_scope: str
    baseline_target_type: str
    baseline_target_id: str
    source_experiment_id: str | None = None
    promotion_stage: str | None = None
    live_eligible: bool = False
    rollback_target_profile_id: str | None = None
    activated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ContinualLearningDriftCapItemSnapshot:
    """Read-only M21 drift-cap row for operator visibility."""

    cap_id: str
    execution_mode_scope: str
    symbol_scope: str
    regime_scope: str
    candidate_type: str
    status: str
    observed_drift_score: float
    reason_code: str
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ContinualLearningPromotionItemSnapshot:
    """Read-only M21 promotion decision row for operator visibility."""

    decision_id: str
    target_type: str
    target_id: str
    decision: str
    summary_text: str
    decided_at: datetime
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContinualLearningEventItemSnapshot:
    """Read-only M21 event row for operator visibility."""

    event_id: str
    event_type: str
    profile_id: str | None
    experiment_id: str | None
    decision_id: str | None
    reason_code: str
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ContinualLearningSnapshot:
    """Combined read-only M21 continual-learning snapshot from the API."""

    summary: ContinualLearningSummarySnapshot
    profiles: tuple[ContinualLearningProfileItemSnapshot, ...] = field(default_factory=tuple)
    drift_caps: tuple[ContinualLearningDriftCapItemSnapshot, ...] = field(default_factory=tuple)
    promotions: tuple[ContinualLearningPromotionItemSnapshot, ...] = field(default_factory=tuple)
    events: tuple[ContinualLearningEventItemSnapshot, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EngineStateSnapshot:
    """Persisted per-symbol paper-trading engine state."""

    service_name: str
    symbol: str
    last_processed_interval_begin: datetime | None
    cooldown_until_interval_begin: datetime | None
    pending_signal_action: str | None
    pending_regime_label: str | None
    updated_at: datetime
    execution_mode: str = "paper"


@dataclass(frozen=True, slots=True)
class DecisionTraceSnapshot:
    """Read-only recent decision trace summary for M14 operator visibility."""

    decision_trace_id: int
    service_name: str
    execution_mode: str
    symbol: str
    signal: str
    signal_row_id: str
    signal_as_of_time: datetime
    model_name: str
    model_version: str
    risk_outcome: str | None
    primary_reason_code: str | None = None
    reason_texts: tuple[str, ...] = field(default_factory=tuple)
    blocked_stage: str | None = None
    json_report_path: str | None = None
    markdown_report_path: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    regime_label: str | None = None
    regime_run_id: str | None = None
    allow_new_long_entries: bool | None = None
    signal_reason_code: str | None = None
    signal_freshness_status: str | None = None
    signal_health_overall_status: str | None = None
    prediction_summary_text: str | None = None
    signal_summary_text: str | None = None
    top_feature_count: int = 0
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    requested_notional: float | None = None
    approved_notional: float | None = None
    ordered_adjustment_count: int = 0


@dataclass(frozen=True, slots=True)
class DatabaseSnapshot:
    """Read-only PostgreSQL snapshot for the dashboard."""

    available: bool
    checked_at: datetime
    latest_features: tuple[LatestFeatureSnapshot, ...] = field(default_factory=tuple)
    positions: tuple[PaperPosition, ...] = field(default_factory=tuple)
    recent_closed_positions: tuple[PaperPosition, ...] = field(default_factory=tuple)
    recent_ledger_entries: tuple[LedgerEntrySnapshot, ...] = field(default_factory=tuple)
    recent_order_events: tuple[OrderAuditSnapshot, ...] = field(default_factory=tuple)
    recent_decision_traces: tuple[DecisionTraceSnapshot, ...] = field(default_factory=tuple)
    latest_blocked_trade: DecisionTraceSnapshot | None = None
    engine_states: tuple[EngineStateSnapshot, ...] = field(default_factory=tuple)
    live_safety_state: LiveSafetySnapshot | None = None
    reliability_states: tuple[ReliabilityStateSnapshot, ...] = field(default_factory=tuple)
    latest_recovery_event: RecoveryEventSnapshot | None = None
    latest_prices: dict[str, float] = field(default_factory=dict)
    cash_balance: float | None = None
    error: str | None = None


def _default_active_alerts_snapshot() -> ActiveAlertsSnapshot:
    return ActiveAlertsSnapshot(
        available=False,
        checked_at=utc_now(),
        error="M17 active alert snapshot unavailable",
    )


def _default_alert_timeline_snapshot() -> AlertTimelineSnapshot:
    return AlertTimelineSnapshot(
        available=False,
        checked_at=utc_now(),
        error="M17 alert timeline snapshot unavailable",
    )


def _default_startup_safety_snapshot() -> StartupSafetySnapshot:
    return StartupSafetySnapshot(
        available=False,
        checked_at=utc_now(),
        error="M17 startup safety snapshot unavailable",
    )


def _default_daily_operations_summary_snapshot() -> DailyOperationsSummarySnapshot:
    return DailyOperationsSummarySnapshot(
        available=False,
        checked_at=utc_now(),
        error="M17 daily operations summary unavailable",
    )


def _default_adaptation_snapshot() -> AdaptationSnapshot:
    return AdaptationSnapshot(
        summary=AdaptationSummarySnapshot(
            available=False,
            checked_at=utc_now(),
            error="M19 adaptation snapshot unavailable",
        )
    )


def _default_continual_learning_snapshot() -> ContinualLearningSnapshot:
    return ContinualLearningSnapshot(
        summary=ContinualLearningSummarySnapshot(
            available=False,
            checked_at=utc_now(),
            error="M21 continual-learning snapshot unavailable",
        )
    )


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Combined API and database state used by the Streamlit app."""

    api_health: ApiHealthSnapshot
    signals: tuple[SignalSnapshot, ...]
    freshness: tuple[FreshnessSnapshot, ...]
    database: DatabaseSnapshot
    system_reliability: SystemReliabilitySnapshot | None = None
    active_alerts: ActiveAlertsSnapshot = field(default_factory=_default_active_alerts_snapshot)
    alert_timeline: AlertTimelineSnapshot = field(default_factory=_default_alert_timeline_snapshot)
    startup_safety: StartupSafetySnapshot = field(default_factory=_default_startup_safety_snapshot)
    daily_operations_summary: DailyOperationsSummarySnapshot = field(
        default_factory=_default_daily_operations_summary_snapshot
    )
    adaptation: AdaptationSnapshot = field(default_factory=_default_adaptation_snapshot)
    continual_learning: ContinualLearningSnapshot = field(
        default_factory=_default_continual_learning_snapshot
    )


class DashboardDataSources:
    """Read-only loaders for the accepted M4 and M5 data contracts."""

    def __init__(
        self,
        *,
        settings: Settings,
        trading_config: PaperTradingConfig,
        http_client: Any | None = None,
        db_connect: Any = asyncpg.connect,
    ) -> None:
        self._settings = settings
        self._trading_config = trading_config
        self._http_client = http_client
        self._db_connect = db_connect
        self._feature_table = _quote_table_name(settings.tables.feature_ohlc)
        self._positions_table = _quote_table_name(POSITIONS_TABLE)
        self._ledger_table = _quote_table_name(LEDGER_TABLE)
        self._state_table = _quote_table_name(STATE_TABLE)
        self._order_events_table = _quote_table_name(ORDER_EVENTS_TABLE)
        self._decision_traces_table = _quote_table_name(DECISION_TRACES_TABLE)
        self._live_safety_table = _quote_table_name(LIVE_SAFETY_TABLE)
        self._reliability_state_table = _quote_table_name(RELIABILITY_STATE_TABLE)
        self._reliability_events_table = _quote_table_name(RELIABILITY_EVENTS_TABLE)

    async def load_snapshot(self) -> DashboardSnapshot:
        """Load one point-in-time dashboard snapshot from API and PostgreSQL."""
        if self._http_client is None:
            async with httpx.AsyncClient(
                base_url=self._settings.dashboard.inference_api_base_url,
                timeout=_HTTP_TIMEOUT_SECONDS,
            ) as http_client:
                return await self._load_with_http_client(http_client)
        return await self._load_with_http_client(self._http_client)

    async def _load_with_http_client(self, http_client: Any) -> DashboardSnapshot:
        api_health = await self._load_api_health(http_client)
        system_reliability = await self._load_system_reliability(http_client)
        active_alerts = await self._load_active_alerts(http_client)
        alert_timeline = await self._load_alert_timeline(http_client)
        startup_safety = await self._load_startup_safety(http_client)
        daily_operations_summary = await self._load_daily_operations_summary(http_client)
        adaptation = await self._load_adaptation(http_client)
        continual_learning = await self._load_continual_learning(http_client)
        signals = await self._load_signals(http_client)
        freshness = await self._load_freshness(http_client)
        database = await self._load_database_snapshot()
        return DashboardSnapshot(
            api_health=api_health,
            signals=signals,
            freshness=freshness,
            database=database,
            system_reliability=system_reliability,
            active_alerts=active_alerts,
            alert_timeline=alert_timeline,
            startup_safety=startup_safety,
            daily_operations_summary=daily_operations_summary,
            adaptation=adaptation,
            continual_learning=continual_learning,
        )

    async def _load_api_health(self, http_client: Any) -> ApiHealthSnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/health")
            payload = response.json()
        except Exception as error:  # pylint: disable=broad-exception-caught
            return ApiHealthSnapshot(
                available=False,
                checked_at=checked_at,
                status="unavailable",
                error=str(error),
            )

        started_at = None
        if isinstance(payload.get("started_at"), str):
            started_at = parse_rfc3339(str(payload["started_at"]))
        return ApiHealthSnapshot(
            available=(
                response.status_code == 200
                and bool(payload.get("model_loaded"))
                and payload.get("database") == "healthy"
            ),
            checked_at=checked_at,
            status=str(payload.get("status", "unknown")),
            service=None if payload.get("service") is None else str(payload["service"]),
            runtime_profile=(
                None
                if payload.get("runtime_profile") is None
                else str(payload["runtime_profile"])
            ),
            model_loaded=bool(payload.get("model_loaded")),
            model_name=None if payload.get("model_name") is None else str(payload["model_name"]),
            model_artifact_path=(
                None
                if payload.get("model_artifact_path") is None
                else str(payload["model_artifact_path"])
            ),
            regime_loaded=bool(payload.get("regime_loaded")),
            regime_run_id=(
                None if payload.get("regime_run_id") is None else str(payload["regime_run_id"])
            ),
            regime_artifact_path=(
                None
                if payload.get("regime_artifact_path") is None
                else str(payload["regime_artifact_path"])
            ),
            database=None if payload.get("database") is None else str(payload["database"]),
            started_at=started_at,
            health_overall_status=(
                None
                if payload.get("health_overall_status") is None
                else str(payload["health_overall_status"])
            ),
            reason_code=(
                None if payload.get("reason_code") is None else str(payload["reason_code"])
            ),
            freshness_status=(
                None
                if payload.get("freshness_status") is None
                else str(payload["freshness_status"])
            ),
            active_alert_count=(
                None
                if payload.get("active_alert_count") is None
                else int(payload["active_alert_count"])
            ),
            max_alert_severity=(
                None
                if payload.get("max_alert_severity") is None
                else str(payload["max_alert_severity"])
            ),
            startup_safety_status=(
                None
                if payload.get("startup_safety_status") is None
                else str(payload["startup_safety_status"])
            ),
            startup_safety_reason_code=(
                None
                if payload.get("startup_safety_reason_code") is None
                else str(payload["startup_safety_reason_code"])
            ),
            active_adaptation_count=(
                None
                if payload.get("active_adaptation_count") is None
                else int(payload["active_adaptation_count"])
            ),
            adaptation_status=(
                None
                if payload.get("adaptation_status") is None
                else str(payload["adaptation_status"])
            ),
            adaptation_evidence_backed=(
                None
                if payload.get("adaptation_evidence_backed") is None
                else bool(payload["adaptation_evidence_backed"])
            ),
            ensemble_profile_id=(
                None
                if payload.get("ensemble_profile_id") is None
                else str(payload["ensemble_profile_id"])
            ),
            ensemble_status=(
                None
                if payload.get("ensemble_status") is None
                else str(payload["ensemble_status"])
            ),
            ensemble_candidate_count=(
                None
                if payload.get("ensemble_candidate_count") is None
                else int(payload["ensemble_candidate_count"])
            ),
            ensemble_roster_status=(
                None
                if payload.get("ensemble_roster_status") is None
                else str(payload["ensemble_roster_status"])
            ),
            ensemble_roster_reason_codes=tuple(
                str(item) for item in payload.get("ensemble_roster_reason_codes", [])
            ),
            active_continual_learning_profile_id=(
                None
                if payload.get("active_continual_learning_profile_id") is None
                else str(payload["active_continual_learning_profile_id"])
            ),
            continual_learning_status=(
                None
                if payload.get("continual_learning_status") is None
                else str(payload["continual_learning_status"])
            ),
            continual_learning_drift_cap_status=(
                None
                if payload.get("continual_learning_drift_cap_status") is None
                else str(payload["continual_learning_drift_cap_status"])
            ),
            continual_learning_evidence_backed=(
                None
                if payload.get("continual_learning_evidence_backed") is None
                else bool(payload["continual_learning_evidence_backed"])
            ),
            error=None if response.status_code == 200 else f"HTTP {response.status_code}",
        )

    async def _load_adaptation(self, http_client: Any) -> AdaptationSnapshot:
        checked_at = utc_now()
        try:
            summary_response = await http_client.get("/adaptation/summary")
            drift_response = await http_client.get("/adaptation/drift")
            performance_response = await http_client.get("/adaptation/performance")
            profiles_response = await http_client.get("/adaptation/profiles")
            promotions_response = await http_client.get("/adaptation/promotions")
            if any(
                response.status_code != 200
                for response in (
                    summary_response,
                    drift_response,
                    performance_response,
                    profiles_response,
                    promotions_response,
                )
            ):
                raise RuntimeError("M19 adaptation endpoints returned non-200 status")
            summary_payload = summary_response.json()
            drift_payload = drift_response.json()
            performance_payload = performance_response.json()
            profiles_payload = profiles_response.json()
            promotions_payload = promotions_response.json()
        except Exception as error:  # pylint: disable=broad-exception-caught
            return _default_adaptation_snapshot_with_error(checked_at=checked_at, error=str(error))
        return AdaptationSnapshot(
            summary=_adaptation_summary_from_payload(summary_payload, checked_at=checked_at),
            drift=tuple(
                _adaptation_drift_item_from_payload(item)
                for item in drift_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            performance=tuple(
                _adaptation_performance_item_from_payload(item)
                for item in performance_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            profiles=tuple(
                _adaptation_profile_item_from_payload(item)
                for item in profiles_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            promotions=tuple(
                _adaptation_promotion_item_from_payload(item)
                for item in promotions_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
        )

    async def _load_continual_learning(self, http_client: Any) -> ContinualLearningSnapshot:
        checked_at = utc_now()
        scope_params = {
            "execution_mode": self._trading_config.execution.mode,
            "symbol": "ALL",
            "regime_label": "ALL",
        }
        try:
            summary_response = await http_client.get(
                "/continual-learning/summary",
                params=scope_params,
            )
            profiles_response = await http_client.get(
                "/continual-learning/profiles",
                params={**scope_params, "limit": 50},
            )
            drift_caps_response = await http_client.get(
                "/continual-learning/drift-caps",
                params={**scope_params, "limit": 50},
            )
            promotions_response = await http_client.get(
                "/continual-learning/promotions",
                params={"limit": 50},
            )
            events_response = await http_client.get(
                "/continual-learning/events",
                params={"limit": 50},
            )
            if any(
                response.status_code != 200
                for response in (
                    summary_response,
                    profiles_response,
                    drift_caps_response,
                    promotions_response,
                    events_response,
                )
            ):
                raise RuntimeError(
                    "M21 continual-learning endpoints returned non-200 status"
                )
            summary_payload = summary_response.json()
            profiles_payload = profiles_response.json()
            drift_caps_payload = drift_caps_response.json()
            promotions_payload = promotions_response.json()
            events_payload = events_response.json()
        except Exception as error:  # pylint: disable=broad-exception-caught
            return _default_continual_learning_snapshot_with_error(
                checked_at=checked_at,
                error=str(error),
            )

        return ContinualLearningSnapshot(
            summary=_continual_learning_summary_from_payload(
                summary_payload,
                checked_at=checked_at,
                aggregated_scope=True,
            ),
            profiles=tuple(
                _continual_learning_profile_item_from_payload(item)
                for item in profiles_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            drift_caps=tuple(
                _continual_learning_drift_cap_item_from_payload(item)
                for item in drift_caps_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            promotions=tuple(
                _continual_learning_promotion_item_from_payload(item)
                for item in promotions_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
            events=tuple(
                _continual_learning_event_item_from_payload(item)
                for item in events_payload.get("items", [])
                if isinstance(item, Mapping)
            ),
        )

    async def _load_system_reliability(self, http_client: Any) -> SystemReliabilitySnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/reliability/system")
            payload = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
        except Exception as error:  # pylint: disable=broad-exception-caught
            return SystemReliabilitySnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
        )
        return _system_reliability_from_payload(payload=payload, checked_at=checked_at)

    async def _load_active_alerts(self, http_client: Any) -> ActiveAlertsSnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/alerts/active")
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("`/alerts/active` payload must be a list")
        except Exception as error:  # pylint: disable=broad-exception-caught
            return ActiveAlertsSnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
            )
        return ActiveAlertsSnapshot(
            available=True,
            checked_at=checked_at,
            items=tuple(
                _active_alert_from_payload(item)
                for item in payload
                if isinstance(item, Mapping)
            ),
        )

    async def _load_alert_timeline(self, http_client: Any) -> AlertTimelineSnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/alerts/timeline", params={"limit": 50})
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("`/alerts/timeline` payload must be a list")
        except Exception as error:  # pylint: disable=broad-exception-caught
            return AlertTimelineSnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
            )
        return AlertTimelineSnapshot(
            available=True,
            checked_at=checked_at,
            items=tuple(
                _alert_timeline_event_from_payload(item)
                for item in payload
                if isinstance(item, Mapping)
            ),
        )

    async def _load_startup_safety(self, http_client: Any) -> StartupSafetySnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/operations/startup-safety")
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("`/operations/startup-safety` payload must be a mapping")
        except Exception as error:  # pylint: disable=broad-exception-caught
            return StartupSafetySnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
            )
        return _startup_safety_from_payload(payload=payload, checked_at=checked_at)

    async def _load_daily_operations_summary(
        self,
        http_client: Any,
    ) -> DailyOperationsSummarySnapshot:
        checked_at = utc_now()
        try:
            response = await http_client.get("/operations/daily-summary")
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("`/operations/daily-summary` payload must be a mapping")
        except Exception as error:  # pylint: disable=broad-exception-caught
            return DailyOperationsSummarySnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
            )
        return _daily_operations_summary_from_payload(
            payload=payload,
            checked_at=checked_at,
        )

    async def _load_signals(self, http_client: Any) -> tuple[SignalSnapshot, ...]:
        signals: list[SignalSnapshot] = []
        for symbol in self._trading_config.symbols:
            checked_at = utc_now()
            try:
                response = await http_client.get("/signal", params={"symbol": symbol})
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                payload = response.json()
                signals.append(
                    _signal_from_payload(
                        symbol=symbol,
                        payload=payload,
                        checked_at=checked_at,
                    )
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                signals.append(
                    SignalSnapshot(
                        symbol=symbol,
                        checked_at=checked_at,
                        available=False,
                        error=str(error),
                    )
                )
        return tuple(signals)

    async def _load_freshness(self, http_client: Any) -> tuple[FreshnessSnapshot, ...]:
        freshness_rows: list[FreshnessSnapshot] = []
        for symbol in self._trading_config.symbols:
            checked_at = utc_now()
            try:
                response = await http_client.get("/freshness", params={"symbol": symbol})
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                payload = response.json()
                freshness_rows.append(
                    _freshness_from_payload(
                        symbol=symbol,
                        payload=payload,
                        checked_at=checked_at,
                    )
                )
            except Exception as error:  # pylint: disable=broad-exception-caught
                freshness_rows.append(
                    FreshnessSnapshot(
                        symbol=symbol,
                        checked_at=checked_at,
                        available=False,
                        error=str(error),
                    )
                )
        return tuple(freshness_rows)

    async def _load_database_snapshot(self) -> DatabaseSnapshot:  # pylint: disable=too-many-locals
        # One read-only snapshot is easier to reason about than interleaved queries
        # across multiple transient connections inside a Streamlit rerun.
        checked_at = utc_now()
        connection = None
        try:
            connection = await self._db_connect(self._settings.postgres.dsn)
            await connection.fetchval("SELECT 1")
            latest_feature_rows = await connection.fetch(
                f"""
                SELECT DISTINCT ON (symbol)
                       symbol, interval_begin, as_of_time, open_price, high_price,
                       low_price, close_price, volume, log_return_1, log_return_3,
                       rsi_14, macd_line_12_26, close_zscore_12, volume_zscore_12
                FROM {self._feature_table}
                WHERE source_exchange = 'kraken'
                  AND interval_minutes = $1
                  AND symbol = ANY($2::text[])
                ORDER BY symbol ASC, as_of_time DESC, interval_begin DESC
                """,
                self._trading_config.interval_minutes,
                list(self._trading_config.symbols),
            )
            position_rows = await connection.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY entry_fill_interval_begin ASC, id ASC
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
            )
            recent_closed_rows = await connection.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1 AND execution_mode = $2 AND status = 'CLOSED'
                ORDER BY closed_at DESC NULLS LAST, id DESC
                LIMIT $3
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
                self._settings.dashboard.recent_trades_limit,
            )
            recent_ledger_rows = await connection.fetch(
                f"""
                SELECT id, symbol, action, reason, fill_interval_begin, fill_time,
                       fill_price, quantity, notional, fee, cash_flow,
                       realized_pnl, signal_row_id, model_name, confidence, regime_label,
                       decision_trace_id
                FROM {self._ledger_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY fill_time DESC, id DESC
                LIMIT $3
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
                self._settings.dashboard.recent_ledger_limit,
            )
            recent_order_rows = await connection.fetch(
                f"""
                SELECT id, order_request_id, symbol, action, lifecycle_state, event_time,
                       reason_code, details, external_order_id, external_status,
                       account_id, environment_name, broker_name,
                       probe_policy_active, probe_symbol, probe_qty,
                       decision_trace_id
                FROM {self._order_events_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY event_time DESC, id DESC
                LIMIT $3
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
                self._settings.dashboard.recent_ledger_limit,
            )
            recent_decision_trace_rows = await connection.fetch(
                f"""
                SELECT id, service_name, execution_mode, symbol, signal, signal_row_id,
                       signal_as_of_time, model_name, model_version, risk_outcome,
                       trace_payload, json_report_path, markdown_report_path,
                       created_at, updated_at
                FROM {self._decision_traces_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY signal_as_of_time DESC, id DESC
                LIMIT $3
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
                self._settings.dashboard.recent_trades_limit,
            )
            latest_blocked_trade_row = await connection.fetchrow(
                f"""
                SELECT id, service_name, execution_mode, symbol, signal, signal_row_id,
                       signal_as_of_time, model_name, model_version, risk_outcome,
                       trace_payload, json_report_path, markdown_report_path,
                       created_at, updated_at
                FROM {self._decision_traces_table}
                WHERE service_name = $1
                  AND execution_mode = $2
                  AND risk_outcome = 'BLOCKED'
                ORDER BY signal_as_of_time DESC, id DESC
                LIMIT 1
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
            )
            engine_state_rows = await connection.fetch(
                f"""
                SELECT service_name, execution_mode, symbol, last_processed_interval_begin,
                       cooldown_until_interval_begin, pending_signal_action,
                       pending_regime_label,
                       updated_at
                FROM {self._state_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY symbol ASC
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
            )
            live_safety_row = await connection.fetchrow(
                f"""
                SELECT service_name, execution_mode, broker_name, live_enabled,
                       startup_checks_passed, startup_checks_passed_at,
                       account_validated, account_id, environment_name,
                       manual_disable_active, consecutive_live_failures,
                       failure_hard_stop_active, last_failure_reason,
                       system_health_status, system_health_reason_code,
                       system_health_checked_at, health_gate_status,
                       health_gate_reason_code, health_gate_detail,
                       broker_cash, broker_equity, reconciliation_status,
                       reconciliation_reason_code, reconciliation_checked_at,
                       unresolved_incident_count, updated_at
                FROM {self._live_safety_table}
                WHERE service_name = $1 AND execution_mode = $2
                """,
                self._trading_config.service_name,
                self._trading_config.execution.mode,
            )
            reliability_state_rows = await connection.fetch(
                f"""
                SELECT service_name, component_name, health_overall_status,
                       freshness_status, breaker_state, failure_count,
                       success_count, reason_code, details, updated_at
                FROM {self._reliability_state_table}
                WHERE service_name = $1
                ORDER BY component_name ASC
                """,
                self._trading_config.service_name,
            )
            latest_recovery_row = await connection.fetchrow(
                f"""
                SELECT service_name, component_name, event_type, event_time,
                       reason_code, health_overall_status, freshness_status,
                       breaker_state, details
                FROM {self._reliability_events_table}
                WHERE service_name = $1
                ORDER BY event_time DESC, id DESC
                LIMIT 1
                """,
                self._trading_config.service_name,
            )
            cash_delta = float(
                await connection.fetchval(
                    f"""
                    SELECT COALESCE(SUM(cash_flow), 0.0)
                    FROM {self._ledger_table}
                    WHERE service_name = $1 AND execution_mode = $2
                    """,
                    self._trading_config.service_name,
                    self._trading_config.execution.mode,
                )
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            return DatabaseSnapshot(
                available=False,
                checked_at=checked_at,
                error=str(error),
            )
        finally:
            if connection is not None:
                await connection.close()

        latest_features = shape_latest_feature_rows(
            symbols=self._trading_config.symbols,
            rows=[dict(row) for row in latest_feature_rows],
        )
        positions = tuple(_position_from_row(row) for row in position_rows)
        recent_closed_positions = tuple(_position_from_row(row) for row in recent_closed_rows)
        recent_ledger_entries = tuple(
            _ledger_entry_from_row(row) for row in recent_ledger_rows
        )
        recent_order_events = tuple(
            _order_audit_from_row(row) for row in recent_order_rows
        )
        recent_decision_traces = tuple(
            _decision_trace_from_row(row) for row in recent_decision_trace_rows
        )
        latest_blocked_trade = (
            None
            if latest_blocked_trade_row is None
            else _decision_trace_from_row(latest_blocked_trade_row)
        )
        engine_states = tuple(_engine_state_from_row(row) for row in engine_state_rows)
        live_safety_state = (
            None if live_safety_row is None else _live_safety_from_row(live_safety_row)
        )
        reliability_states = tuple(
            _reliability_state_from_row(row) for row in reliability_state_rows
        )
        latest_recovery_event = (
            None
            if latest_recovery_row is None
            else _recovery_event_from_row(latest_recovery_row)
        )
        latest_prices = {row.symbol: row.close_price for row in latest_features}
        return DatabaseSnapshot(
            available=True,
            checked_at=checked_at,
            latest_features=latest_features,
            positions=positions,
            recent_closed_positions=recent_closed_positions,
            recent_ledger_entries=recent_ledger_entries,
            recent_order_events=recent_order_events,
            recent_decision_traces=recent_decision_traces,
            latest_blocked_trade=latest_blocked_trade,
            engine_states=engine_states,
            live_safety_state=live_safety_state,
            reliability_states=reliability_states,
            latest_recovery_event=latest_recovery_event,
            latest_prices=latest_prices,
            cash_balance=self._trading_config.risk.initial_cash + cash_delta,
        )


def shape_latest_feature_rows(
    *,
    symbols: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[LatestFeatureSnapshot, ...]:
    """Pick the newest canonical feature row per asset and return symbol-ordered rows."""
    latest_by_symbol: dict[str, LatestFeatureSnapshot] = {}
    for row in rows:
        snapshot = _feature_row_from_mapping(row)
        existing = latest_by_symbol.get(snapshot.symbol)
        if existing is None or (
            snapshot.as_of_time,
            snapshot.interval_begin,
        ) > (
            existing.as_of_time,
            existing.interval_begin,
        ):
            latest_by_symbol[snapshot.symbol] = snapshot

    return tuple(
        latest_by_symbol[symbol]
        for symbol in symbols
        if symbol in latest_by_symbol
    )


def _signal_from_payload(
    *,
    symbol: str,
    payload: Mapping[str, Any],
    checked_at: datetime,
) -> SignalSnapshot:
    thresholds = payload.get("thresholds", {})
    as_of_time = None
    if isinstance(payload.get("as_of_time"), str):
        as_of_time = parse_rfc3339(str(payload["as_of_time"]))
    continual_learning_payload = payload.get("continual_learning")
    continual_learning_reason_codes: tuple[str, ...] = ()
    if isinstance(continual_learning_payload, Mapping):
        continual_learning_reason_codes = tuple(
            str(item)
            for item in continual_learning_payload.get("reason_codes", [])
        )
    return SignalSnapshot(
        symbol=symbol,
        checked_at=checked_at,
        available=True,
        signal=str(payload["signal"]),
        reason=str(payload["reason"]),
        prob_up=float(payload["prob_up"]),
        prob_down=float(payload["prob_down"]),
        confidence=float(payload["confidence"]),
        predicted_class=str(payload["predicted_class"]),
        row_id=str(payload["row_id"]),
        as_of_time=as_of_time,
        model_name=str(payload["model_name"]),
        regime_label=(
            None if payload.get("regime_label") is None else str(payload["regime_label"])
        ),
        regime_run_id=(
            None if payload.get("regime_run_id") is None else str(payload["regime_run_id"])
        ),
        trade_allowed=(
            None
            if payload.get("trade_allowed") is None
            else bool(payload["trade_allowed"])
        ),
        buy_threshold=float(thresholds["buy_prob_up"]),
        sell_threshold=float(thresholds["sell_prob_up"]),
        signal_status=(
            None if payload.get("signal_status") is None else str(payload["signal_status"])
        ),
        decision_source=(
            None
            if payload.get("decision_source") is None
            else str(payload["decision_source"])
        ),
        reason_code=(
            None if payload.get("reason_code") is None else str(payload["reason_code"])
        ),
        freshness_status=(
            None
            if payload.get("freshness_status") is None
            else str(payload["freshness_status"])
        ),
        health_overall_status=(
            None
            if payload.get("health_overall_status") is None
            else str(payload["health_overall_status"])
        ),
        adaptation_profile_id=(
            None
            if payload.get("adaptation_profile_id") is None
            else str(payload["adaptation_profile_id"])
        ),
        calibrated_confidence=(
            None
            if payload.get("calibrated_confidence") is None
            else float(payload["calibrated_confidence"])
        ),
        adaptive_size_multiplier=(
            None
            if payload.get("adaptive_size_multiplier") is None
            else float(payload["adaptive_size_multiplier"])
        ),
        drift_status=(
            None
            if payload.get("drift_status") is None
            else str(payload["drift_status"])
        ),
        frozen_by_health_gate=bool(payload.get("frozen_by_health_gate", False)),
        ensemble_profile_id=(
            None
            if payload.get("ensemble_profile_id") is None
            else str(payload["ensemble_profile_id"])
        ),
        ensemble_active=bool(payload.get("ensemble_active", False)),
        ensemble_candidate_count=(
            None
            if payload.get("ensemble_candidate_count") is None
            else int(payload["ensemble_candidate_count"])
        ),
        ensemble_fallback_reason=(
            None
            if payload.get("ensemble_fallback_reason") is None
            else str(payload["ensemble_fallback_reason"])
        ),
        ensemble_agreement_band=(
            None
            if payload.get("ensemble_agreement_band") is None
            else str(payload["ensemble_agreement_band"])
        ),
        ensemble_effective_confidence=(
            None
            if payload.get("ensemble_effective_confidence") is None
            else float(payload["ensemble_effective_confidence"])
        ),
        ensemble_roster_status=(
            None
            if payload.get("ensemble_roster_status") is None
            else str(payload["ensemble_roster_status"])
        ),
        ensemble_roster_reason_codes=tuple(
            str(item) for item in payload.get("ensemble_roster_reason_codes", [])
        ),
        continual_learning_status=(
            None
            if payload.get("continual_learning_status") is None
            else str(payload["continual_learning_status"])
        ),
        continual_learning_profile_id=(
            None
            if payload.get("continual_learning_profile_id") is None
            else str(payload["continual_learning_profile_id"])
        ),
        continual_learning_evidence_backed=(
            None
            if payload.get("continual_learning_evidence_backed") is None
            else bool(payload["continual_learning_evidence_backed"])
        ),
        continual_learning_frozen=bool(payload.get("continual_learning_frozen", False)),
        continual_learning_candidate_type=(
            None
            if not isinstance(continual_learning_payload, Mapping)
            or continual_learning_payload.get("candidate_type") is None
            else str(continual_learning_payload.get("candidate_type"))
        ),
        continual_learning_promotion_stage=(
            None
            if not isinstance(continual_learning_payload, Mapping)
            or continual_learning_payload.get("promotion_stage") is None
            else str(continual_learning_payload.get("promotion_stage"))
        ),
        continual_learning_baseline_target_type=(
            None
            if not isinstance(continual_learning_payload, Mapping)
            or continual_learning_payload.get("baseline_target_type") is None
            else str(continual_learning_payload.get("baseline_target_type"))
        ),
        continual_learning_baseline_target_id=(
            None
            if not isinstance(continual_learning_payload, Mapping)
            or continual_learning_payload.get("baseline_target_id") is None
            else str(continual_learning_payload.get("baseline_target_id"))
        ),
        continual_learning_drift_cap_status=(
            None
            if not isinstance(continual_learning_payload, Mapping)
            or continual_learning_payload.get("drift_cap_status") is None
            else str(continual_learning_payload.get("drift_cap_status"))
        ),
        continual_learning_reason_codes=continual_learning_reason_codes,
    )


def _default_adaptation_snapshot_with_error(
    *,
    checked_at: datetime,
    error: str,
) -> AdaptationSnapshot:
    return AdaptationSnapshot(
        summary=AdaptationSummarySnapshot(
            available=False,
            checked_at=checked_at,
            error=error,
        )
    )


def _default_continual_learning_snapshot_with_error(
    *,
    checked_at: datetime,
    error: str,
) -> ContinualLearningSnapshot:
    return ContinualLearningSnapshot(
        summary=ContinualLearningSummarySnapshot(
            available=False,
            checked_at=checked_at,
            error=error,
        )
    )


def _adaptation_summary_from_payload(
    payload: Mapping[str, Any],
    *,
    checked_at: datetime,
) -> AdaptationSummarySnapshot:
    latest_drift_updated_at = None
    if isinstance(payload.get("latest_drift_updated_at"), str):
        latest_drift_updated_at = parse_rfc3339(str(payload["latest_drift_updated_at"]))
    latest_performance_created_at = None
    if isinstance(payload.get("latest_performance_created_at"), str):
        latest_performance_created_at = parse_rfc3339(
            str(payload["latest_performance_created_at"])
        )
    return AdaptationSummarySnapshot(
        available=True,
        checked_at=checked_at,
        enabled=bool(payload.get("enabled", False)),
        active_profile_count=int(payload.get("active_profile_count", 0)),
        active_profile_id=(
            None
            if payload.get("active_profile_id") is None
            else str(payload["active_profile_id"])
        ),
        adaptation_status=(
            None
            if payload.get("adaptation_status") is None
            else str(payload["adaptation_status"])
        ),
        evidence_backed=bool(payload.get("evidence_backed", False)),
        frozen_by_health_gate=bool(payload.get("frozen_by_health_gate", False)),
        latest_drift_status=(
            None
            if payload.get("latest_drift_status") is None
            else str(payload["latest_drift_status"])
        ),
        latest_drift_updated_at=latest_drift_updated_at,
        latest_promotion_decision=(
            None
            if payload.get("latest_promotion_decision") is None
            else str(payload["latest_promotion_decision"])
        ),
        latest_performance_window_id=(
            None
            if payload.get("latest_performance_window_id") is None
            else str(payload["latest_performance_window_id"])
        ),
        latest_performance_trade_count=(
            None
            if payload.get("latest_performance_trade_count") is None
            else int(payload["latest_performance_trade_count"])
        ),
        latest_performance_created_at=latest_performance_created_at,
        reason_codes=tuple(str(item) for item in payload.get("reason_codes", [])),
    )


def _adaptation_drift_item_from_payload(
    payload: Mapping[str, Any],
) -> AdaptationDriftItemSnapshot:
    updated_at = None
    if isinstance(payload.get("updated_at"), str):
        updated_at = parse_rfc3339(str(payload["updated_at"]))
    return AdaptationDriftItemSnapshot(
        symbol=str(payload["symbol"]),
        regime_label=str(payload["regime_label"]),
        detector_name=str(payload["detector_name"]),
        window_id=str(payload["window_id"]),
        drift_score=float(payload["drift_score"]),
        status=str(payload["status"]),
        reason_code=str(payload["reason_code"]),
        updated_at=updated_at,
    )


def _adaptation_performance_item_from_payload(
    payload: Mapping[str, Any],
) -> AdaptationPerformanceItemSnapshot:
    return AdaptationPerformanceItemSnapshot(
        execution_mode=str(payload["execution_mode"]),
        symbol=str(payload["symbol"]),
        regime_label=str(payload["regime_label"]),
        window_id=str(payload["window_id"]),
        window_type=str(payload["window_type"]),
        trade_count=int(payload["trade_count"]),
        net_pnl_after_costs=float(payload["net_pnl_after_costs"]),
        max_drawdown=float(payload["max_drawdown"]),
        profit_factor=float(payload["profit_factor"]),
        win_rate=float(payload["win_rate"]),
        blocked_trade_rate=float(payload["blocked_trade_rate"]),
        shadow_divergence_rate=float(payload["shadow_divergence_rate"]),
    )


def _adaptation_profile_item_from_payload(
    payload: Mapping[str, Any],
) -> AdaptationProfileItemSnapshot:
    activated_at = None
    if isinstance(payload.get("activated_at"), str):
        activated_at = parse_rfc3339(str(payload["activated_at"]))
    return AdaptationProfileItemSnapshot(
        profile_id=str(payload["profile_id"]),
        status=str(payload["status"]),
        execution_mode_scope=str(payload["execution_mode_scope"]),
        symbol_scope=str(payload["symbol_scope"]),
        regime_scope=str(payload["regime_scope"]),
        rollback_target_profile_id=(
            None
            if payload.get("rollback_target_profile_id") is None
            else str(payload["rollback_target_profile_id"])
        ),
        activated_at=activated_at,
    )


def _adaptation_promotion_item_from_payload(
    payload: Mapping[str, Any],
) -> AdaptationPromotionItemSnapshot:
    return AdaptationPromotionItemSnapshot(
        decision_id=str(payload["decision_id"]),
        target_type=str(payload["target_type"]),
        target_id=str(payload["target_id"]),
        decision=str(payload["decision"]),
        summary_text=str(payload["summary_text"]),
        decided_at=parse_rfc3339(str(payload["decided_at"])),
        reason_codes=tuple(str(item) for item in payload.get("reason_codes", [])),
    )


def _continual_learning_summary_from_payload(
    payload: Mapping[str, Any],
    *,
    checked_at: datetime,
    aggregated_scope: bool,
) -> ContinualLearningSummarySnapshot:
    latest_drift_cap_updated_at = None
    if isinstance(payload.get("latest_drift_cap_updated_at"), str):
        latest_drift_cap_updated_at = parse_rfc3339(
            str(payload["latest_drift_cap_updated_at"])
        )
    return ContinualLearningSummarySnapshot(
        available=True,
        checked_at=checked_at,
        enabled=bool(payload.get("enabled", False)),
        active_profile_count=int(payload.get("active_profile_count", 0)),
        active_profile_id=(
            None
            if payload.get("active_profile_id") is None
            else str(payload["active_profile_id"])
        ),
        continual_learning_status=(
            None
            if payload.get("continual_learning_status") is None
            else str(payload["continual_learning_status"])
        ),
        evidence_backed=bool(payload.get("evidence_backed", False)),
        latest_drift_cap_status=(
            None
            if payload.get("latest_drift_cap_status") is None
            else str(payload["latest_drift_cap_status"])
        ),
        latest_drift_cap_updated_at=latest_drift_cap_updated_at,
        latest_promotion_decision=(
            None
            if payload.get("latest_promotion_decision") is None
            else str(payload["latest_promotion_decision"])
        ),
        latest_event_type=(
            None
            if payload.get("latest_event_type") is None
            else str(payload["latest_event_type"])
        ),
        reason_codes=tuple(str(item) for item in payload.get("reason_codes", [])),
        aggregated_scope=aggregated_scope,
    )


def _continual_learning_profile_item_from_payload(
    payload: Mapping[str, Any],
) -> ContinualLearningProfileItemSnapshot:
    activated_at = None
    if isinstance(payload.get("activated_at"), str):
        activated_at = parse_rfc3339(str(payload["activated_at"]))
    return ContinualLearningProfileItemSnapshot(
        profile_id=str(payload["profile_id"]),
        status=str(payload["status"]),
        candidate_type=str(payload["candidate_type"]),
        execution_mode_scope=str(payload["execution_mode_scope"]),
        symbol_scope=str(payload["symbol_scope"]),
        regime_scope=str(payload["regime_scope"]),
        baseline_target_type=str(payload["baseline_target_type"]),
        baseline_target_id=str(payload["baseline_target_id"]),
        source_experiment_id=(
            None
            if payload.get("source_experiment_id") is None
            else str(payload["source_experiment_id"])
        ),
        promotion_stage=(
            None
            if payload.get("promotion_stage") is None
            else str(payload["promotion_stage"])
        ),
        live_eligible=bool(payload.get("live_eligible", False)),
        rollback_target_profile_id=(
            None
            if payload.get("rollback_target_profile_id") is None
            else str(payload["rollback_target_profile_id"])
        ),
        activated_at=activated_at,
    )


def _continual_learning_drift_cap_item_from_payload(
    payload: Mapping[str, Any],
) -> ContinualLearningDriftCapItemSnapshot:
    updated_at = None
    if isinstance(payload.get("updated_at"), str):
        updated_at = parse_rfc3339(str(payload["updated_at"]))
    return ContinualLearningDriftCapItemSnapshot(
        cap_id=str(payload["cap_id"]),
        execution_mode_scope=str(payload["execution_mode_scope"]),
        symbol_scope=str(payload["symbol_scope"]),
        regime_scope=str(payload["regime_scope"]),
        candidate_type=str(payload["candidate_type"]),
        status=str(payload["status"]),
        observed_drift_score=float(payload["observed_drift_score"]),
        reason_code=str(payload["reason_code"]),
        updated_at=updated_at,
    )


def _continual_learning_promotion_item_from_payload(
    payload: Mapping[str, Any],
) -> ContinualLearningPromotionItemSnapshot:
    return ContinualLearningPromotionItemSnapshot(
        decision_id=str(payload["decision_id"]),
        target_type=str(payload["target_type"]),
        target_id=str(payload["target_id"]),
        decision=str(payload["decision"]),
        summary_text=str(payload["summary_text"]),
        decided_at=parse_rfc3339(str(payload["decided_at"])),
        reason_codes=tuple(str(item) for item in payload.get("reason_codes", [])),
    )


def _continual_learning_event_item_from_payload(
    payload: Mapping[str, Any],
) -> ContinualLearningEventItemSnapshot:
    created_at = None
    if isinstance(payload.get("created_at"), str):
        created_at = parse_rfc3339(str(payload["created_at"]))
    return ContinualLearningEventItemSnapshot(
        event_id=str(payload["event_id"]),
        event_type=str(payload["event_type"]),
        profile_id=(
            None if payload.get("profile_id") is None else str(payload["profile_id"])
        ),
        experiment_id=(
            None
            if payload.get("experiment_id") is None
            else str(payload["experiment_id"])
        ),
        decision_id=(
            None
            if payload.get("decision_id") is None
            else str(payload["decision_id"])
        ),
        reason_code=str(payload["reason_code"]),
        created_at=created_at,
    )


def _freshness_from_payload(
    *,
    symbol: str,
    payload: Mapping[str, Any],
    checked_at: datetime,
) -> FreshnessSnapshot:
    interval_begin = None
    if isinstance(payload.get("interval_begin"), str):
        interval_begin = parse_rfc3339(str(payload["interval_begin"]))
    as_of_time = None
    if isinstance(payload.get("as_of_time"), str):
        as_of_time = parse_rfc3339(str(payload["as_of_time"]))
    return FreshnessSnapshot(
        symbol=symbol,
        checked_at=checked_at,
        available=True,
        row_id=None if payload.get("row_id") is None else str(payload["row_id"]),
        interval_begin=interval_begin,
        as_of_time=as_of_time,
        health_overall_status=str(payload["health_overall_status"]),
        freshness_status=str(payload["freshness_status"]),
        reason_code=str(payload["reason_code"]),
        feature_freshness_status=str(payload["feature_freshness_status"]),
        feature_reason_code=str(payload["feature_reason_code"]),
        feature_age_seconds=(
            None
            if payload.get("feature_age_seconds") is None
            else float(payload["feature_age_seconds"])
        ),
        regime_freshness_status=str(payload["regime_freshness_status"]),
        regime_reason_code=str(payload["regime_reason_code"]),
        regime_age_seconds=(
            None
            if payload.get("regime_age_seconds") is None
            else float(payload["regime_age_seconds"])
        ),
        detail=None if payload.get("detail") is None else str(payload["detail"]),
    )


def _system_reliability_from_payload(
    *,
    payload: Mapping[str, Any],
    checked_at: datetime,
) -> SystemReliabilitySnapshot:
    latest_recovery_event_payload = payload.get("latest_recovery_event")
    latest_recovery_event = None
    if isinstance(latest_recovery_event_payload, Mapping):
        latest_recovery_event = _recovery_event_from_row(
            latest_recovery_event_payload
        )
    services_payload = payload.get("services", [])
    lag_payload = payload.get("lag_by_symbol", [])
    return SystemReliabilitySnapshot(
        available=True,
        checked_at=checked_at,
        service_name=(
            None if payload.get("service_name") is None else str(payload["service_name"])
        ),
        health_overall_status=(
            None
            if payload.get("health_overall_status") is None
            else str(payload["health_overall_status"])
        ),
        reason_codes=tuple(str(code) for code in payload.get("reason_codes", [])),
        lag_breach_active=(
            None
            if payload.get("lag_breach_active") is None
            else bool(payload["lag_breach_active"])
        ),
        services=tuple(
            _service_health_from_payload(service_payload)
            for service_payload in services_payload
            if isinstance(service_payload, Mapping)
        ),
        lag_by_symbol=tuple(
            _feature_lag_from_payload(lag_snapshot)
            for lag_snapshot in lag_payload
            if isinstance(lag_snapshot, Mapping)
        ),
        latest_recovery_event=latest_recovery_event,
    )


def _active_alert_from_payload(
    payload: Mapping[str, Any],
) -> ActiveAlertSnapshot:
    return ActiveAlertSnapshot(
        fingerprint=str(payload["fingerprint"]),
        service_name=str(payload["service_name"]),
        execution_mode=str(payload["execution_mode"]),
        category=str(payload["category"]),
        severity=str(payload["severity"]),
        reason_code=str(payload["reason_code"]),
        source_component=str(payload["source_component"]),
        is_active=bool(payload["is_active"]),
        opened_at=parse_rfc3339(str(payload["opened_at"])),
        last_seen_at=parse_rfc3339(str(payload["last_seen_at"])),
        symbol=None if payload.get("symbol") is None else str(payload["symbol"]),
        last_event_id=(
            None if payload.get("last_event_id") is None else int(payload["last_event_id"])
        ),
        occurrence_count=int(payload.get("occurrence_count", 0)),
    )


def _alert_timeline_event_from_payload(
    payload: Mapping[str, Any],
) -> AlertTimelineEventSnapshot:
    created_at = None
    if isinstance(payload.get("created_at"), str):
        created_at = parse_rfc3339(str(payload["created_at"]))
    raw_payload_json = payload.get("payload_json", {})
    payload_json = (
        dict(raw_payload_json)
        if isinstance(raw_payload_json, Mapping)
        else {}
    )
    return AlertTimelineEventSnapshot(
        event_id=None if payload.get("id") is None else int(payload["id"]),
        service_name=str(payload["service_name"]),
        execution_mode=str(payload["execution_mode"]),
        category=str(payload["category"]),
        severity=str(payload["severity"]),
        event_state=str(payload["event_state"]),
        reason_code=str(payload["reason_code"]),
        source_component=str(payload["source_component"]),
        fingerprint=str(payload["fingerprint"]),
        summary_text=str(payload["summary_text"]),
        event_time=parse_rfc3339(str(payload["event_time"])),
        symbol=None if payload.get("symbol") is None else str(payload["symbol"]),
        detail=None if payload.get("detail") is None else str(payload["detail"]),
        related_order_request_id=(
            None
            if payload.get("related_order_request_id") is None
            else int(payload["related_order_request_id"])
        ),
        related_decision_trace_id=(
            None
            if payload.get("related_decision_trace_id") is None
            else int(payload["related_decision_trace_id"])
        ),
        payload_json=payload_json,
        created_at=created_at,
    )


def _startup_safety_from_payload(
    *,
    payload: Mapping[str, Any],
    checked_at: datetime,
) -> StartupSafetySnapshot:
    generated_at = None
    if isinstance(payload.get("generated_at"), str):
        generated_at = parse_rfc3339(str(payload["generated_at"]))
    startup_validation_payload = payload.get("startup_validation", {})
    live_startup_payload = payload.get("live_startup", {})
    startup_report_path = None
    startup_validation_passed = None
    if isinstance(startup_validation_payload, Mapping):
        startup_report_path = (
            None
            if startup_validation_payload.get("report_path") is None
            else str(startup_validation_payload["report_path"])
        )
        if startup_validation_payload.get("startup_validation_passed") is not None:
            startup_validation_passed = bool(
                startup_validation_payload["startup_validation_passed"]
            )
    live_reason_code = None
    if isinstance(live_startup_payload, Mapping):
        live_reason_code = (
            None
            if live_startup_payload.get("primary_reason_code") is None
            else str(live_startup_payload["primary_reason_code"])
        )
    return StartupSafetySnapshot(
        available=True,
        checked_at=checked_at,
        generated_at=generated_at,
        service_name=(
            None if payload.get("service_name") is None else str(payload["service_name"])
        ),
        execution_mode=(
            None
            if payload.get("execution_mode") is None
            else str(payload["execution_mode"])
        ),
        runtime_profile=(
            None
            if payload.get("runtime_profile") is None
            else str(payload["runtime_profile"])
        ),
        startup_safety_passed=(
            None
            if payload.get("startup_safety_passed") is None
            else bool(payload["startup_safety_passed"])
        ),
        primary_reason_code=(
            None
            if payload.get("primary_reason_code") is None
            else str(payload["primary_reason_code"])
        ),
        summary_text=(
            None if payload.get("summary_text") is None else str(payload["summary_text"])
        ),
        startup_report_path=startup_report_path,
        startup_validation_passed=startup_validation_passed,
        live_startup_reason_code=live_reason_code,
    )


def _daily_operations_summary_from_payload(
    *,
    payload: Mapping[str, Any],
    checked_at: datetime,
) -> DailyOperationsSummarySnapshot:
    generated_at = None
    if isinstance(payload.get("generated_at"), str):
        generated_at = parse_rfc3339(str(payload["generated_at"]))
    counts_by_category = payload.get("counts_by_category", {})
    startup_safety_status = payload.get("startup_safety_status", {})
    order_failure_counts = payload.get("order_failure_counts", {})
    drawdown_state = payload.get("drawdown_state", {})
    actionable_signal_counts = payload.get("actionable_signal_counts", {})
    silence_flood_episodes = payload.get("silence_flood_episodes", {})
    return DailyOperationsSummarySnapshot(
        available=True,
        checked_at=checked_at,
        generated_at=generated_at,
        service_name=(
            None if payload.get("service_name") is None else str(payload["service_name"])
        ),
        execution_mode=(
            None
            if payload.get("execution_mode") is None
            else str(payload["execution_mode"])
        ),
        runtime_profile=(
            None
            if payload.get("runtime_profile") is None
            else str(payload["runtime_profile"])
        ),
        summary_date=(
            None if payload.get("summary_date") is None else str(payload["summary_date"])
        ),
        unresolved_count=(
            None
            if payload.get("unresolved_count") is None
            else int(payload["unresolved_count"])
        ),
        highest_severity=(
            None
            if payload.get("highest_severity") is None
            else str(payload["highest_severity"])
        ),
        counts_by_category=(
            {
                str(key): int(value)
                for key, value in counts_by_category.items()
            }
            if isinstance(counts_by_category, Mapping)
            else {}
        ),
        startup_safety_status=(
            dict(startup_safety_status)
            if isinstance(startup_safety_status, Mapping)
            else {}
        ),
        order_failure_counts=(
            dict(order_failure_counts)
            if isinstance(order_failure_counts, Mapping)
            else {}
        ),
        drawdown_state=(
            dict(drawdown_state)
            if isinstance(drawdown_state, Mapping)
            else {}
        ),
        actionable_signal_counts=(
            dict(actionable_signal_counts)
            if isinstance(actionable_signal_counts, Mapping)
            else {}
        ),
        silence_flood_episodes=(
            {
                str(key): int(value)
                for key, value in silence_flood_episodes.items()
            }
            if isinstance(silence_flood_episodes, Mapping)
            else {}
        ),
        live_mode_activation_count=(
            None
            if payload.get("live_mode_activation_count") is None
            else int(payload["live_mode_activation_count"])
        ),
    )


def _service_health_from_payload(
    payload: Mapping[str, Any],
) -> ServiceHealthSummarySnapshot:
    heartbeat_at = None
    if isinstance(payload.get("heartbeat_at"), str):
        heartbeat_at = parse_rfc3339(str(payload["heartbeat_at"]))
    checked_at = (
        utc_now()
        if not isinstance(payload.get("checked_at"), str)
        else parse_rfc3339(str(payload["checked_at"]))
    )
    return ServiceHealthSummarySnapshot(
        service_name=str(payload["service_name"]),
        component_name=str(payload["component_name"]),
        checked_at=checked_at,
        heartbeat_at=heartbeat_at,
        heartbeat_age_seconds=(
            None
            if payload.get("heartbeat_age_seconds") is None
            else float(payload["heartbeat_age_seconds"])
        ),
        heartbeat_freshness_status=str(payload["heartbeat_freshness_status"]),
        health_overall_status=str(payload["health_overall_status"]),
        reason_code=str(payload["reason_code"]),
        detail=None if payload.get("detail") is None else str(payload["detail"]),
        feed_freshness_status=(
            None
            if payload.get("feed_freshness_status") is None
            else str(payload["feed_freshness_status"])
        ),
        feed_reason_code=(
            None if payload.get("feed_reason_code") is None else str(payload["feed_reason_code"])
        ),
        feed_age_seconds=(
            None
            if payload.get("feed_age_seconds") is None
            else float(payload["feed_age_seconds"])
        ),
    )


def _feature_lag_from_payload(
    payload: Mapping[str, Any],
) -> FeatureLagSummarySnapshot:
    latest_raw_event_received_at = None
    if isinstance(payload.get("latest_raw_event_received_at"), str):
        latest_raw_event_received_at = parse_rfc3339(
            str(payload["latest_raw_event_received_at"])
        )
    latest_feature_interval_begin = None
    if isinstance(payload.get("latest_feature_interval_begin"), str):
        latest_feature_interval_begin = parse_rfc3339(
            str(payload["latest_feature_interval_begin"])
        )
    latest_feature_as_of_time = None
    if isinstance(payload.get("latest_feature_as_of_time"), str):
        latest_feature_as_of_time = parse_rfc3339(
            str(payload["latest_feature_as_of_time"])
        )
    return FeatureLagSummarySnapshot(
        service_name=str(payload["service_name"]),
        component_name=str(payload["component_name"]),
        symbol=str(payload["symbol"]),
        evaluated_at=parse_rfc3339(str(payload["evaluated_at"])),
        latest_raw_event_received_at=latest_raw_event_received_at,
        latest_feature_interval_begin=latest_feature_interval_begin,
        latest_feature_as_of_time=latest_feature_as_of_time,
        time_lag_seconds=(
            None if payload.get("time_lag_seconds") is None else float(payload["time_lag_seconds"])
        ),
        processing_lag_seconds=(
            None
            if payload.get("processing_lag_seconds") is None
            else float(payload["processing_lag_seconds"])
        ),
        time_lag_reason_code=str(payload["time_lag_reason_code"]),
        processing_lag_reason_code=str(payload["processing_lag_reason_code"]),
        lag_breach=bool(payload["lag_breach"]),
        health_overall_status=str(payload["health_overall_status"]),
        reason_code=str(payload["reason_code"]),
        detail=None if payload.get("detail") is None else str(payload["detail"]),
    )


def _feature_row_from_mapping(row: Mapping[str, Any]) -> LatestFeatureSnapshot:
    return LatestFeatureSnapshot(
        symbol=str(row["symbol"]),
        interval_begin=row["interval_begin"],
        as_of_time=row["as_of_time"],
        open_price=float(row["open_price"]),
        high_price=float(row["high_price"]),
        low_price=float(row["low_price"]),
        close_price=float(row["close_price"]),
        volume=float(row["volume"]),
        log_return_1=float(row["log_return_1"]),
        log_return_3=float(row["log_return_3"]),
        rsi_14=float(row["rsi_14"]),
        macd_line_12_26=float(row["macd_line_12_26"]),
        close_zscore_12=float(row["close_zscore_12"]),
        volume_zscore_12=float(row["volume_zscore_12"]),
    )


def _position_from_row(row: Mapping[str, Any]) -> PaperPosition:
    return PaperPosition(
        service_name=str(row["service_name"]),
        symbol=str(row["symbol"]),
        status=str(row["status"]),
        entry_signal_interval_begin=row["entry_signal_interval_begin"],
        entry_signal_as_of_time=row["entry_signal_as_of_time"],
        entry_signal_row_id=str(row["entry_signal_row_id"]),
        entry_reason=str(row["entry_reason"]),
        entry_model_name=str(row["entry_model_name"]),
        entry_prob_up=float(row["entry_prob_up"]),
        entry_confidence=float(row["entry_confidence"]),
        entry_fill_interval_begin=row["entry_fill_interval_begin"],
        entry_fill_time=row["entry_fill_time"],
        entry_price=float(row["entry_price"]),
        quantity=float(row["quantity"]),
        entry_notional=float(row["entry_notional"]),
        entry_fee=float(row["entry_fee"]),
        stop_loss_price=float(row["stop_loss_price"]),
        take_profit_price=float(row["take_profit_price"]),
        entry_regime_label=(
            None if row["entry_regime_label"] is None else str(row["entry_regime_label"])
        ),
        entry_decision_trace_id=(
            None
            if row["entry_decision_trace_id"] is None
            else int(row["entry_decision_trace_id"])
        ),
        position_id=int(row["id"]),
        exit_reason=None if row["exit_reason"] is None else str(row["exit_reason"]),
        exit_signal_interval_begin=row["exit_signal_interval_begin"],
        exit_signal_as_of_time=row["exit_signal_as_of_time"],
        exit_signal_row_id=None
        if row["exit_signal_row_id"] is None
        else str(row["exit_signal_row_id"]),
        exit_model_name=None if row["exit_model_name"] is None else str(row["exit_model_name"]),
        exit_prob_up=None if row["exit_prob_up"] is None else float(row["exit_prob_up"]),
        exit_confidence=None
        if row["exit_confidence"] is None
        else float(row["exit_confidence"]),
        exit_fill_interval_begin=row["exit_fill_interval_begin"],
        exit_fill_time=row["exit_fill_time"],
        exit_price=None if row["exit_price"] is None else float(row["exit_price"]),
        exit_notional=None if row["exit_notional"] is None else float(row["exit_notional"]),
        exit_fee=None if row["exit_fee"] is None else float(row["exit_fee"]),
        realized_pnl=None if row["realized_pnl"] is None else float(row["realized_pnl"]),
        realized_return=None
        if row["realized_return"] is None
        else float(row["realized_return"]),
        exit_regime_label=(
            None if row["exit_regime_label"] is None else str(row["exit_regime_label"])
        ),
        exit_decision_trace_id=(
            None
            if row["exit_decision_trace_id"] is None
            else int(row["exit_decision_trace_id"])
        ),
        opened_at=row["opened_at"],
        closed_at=row["closed_at"],
        updated_at=row["updated_at"],
    )


def _ledger_entry_from_row(row: Mapping[str, Any]) -> LedgerEntrySnapshot:
    return LedgerEntrySnapshot(
        ledger_id=int(row["id"]),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        reason=str(row["reason"]),
        fill_interval_begin=row["fill_interval_begin"],
        fill_time=row["fill_time"],
        fill_price=float(row["fill_price"]),
        quantity=float(row["quantity"]),
        notional=float(row["notional"]),
        fee=float(row["fee"]),
        cash_flow=float(row["cash_flow"]),
        realized_pnl=None if row["realized_pnl"] is None else float(row["realized_pnl"]),
        signal_row_id=None if row["signal_row_id"] is None else str(row["signal_row_id"]),
        model_name=None if row["model_name"] is None else str(row["model_name"]),
        confidence=None if row["confidence"] is None else float(row["confidence"]),
        regime_label=None if row["regime_label"] is None else str(row["regime_label"]),
        decision_trace_id=(
            None
            if row.get("decision_trace_id") is None
            else int(row.get("decision_trace_id"))
        ),
    )


def _engine_state_from_row(row: Mapping[str, Any]) -> EngineStateSnapshot:
    return EngineStateSnapshot(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        last_processed_interval_begin=row["last_processed_interval_begin"],
        cooldown_until_interval_begin=row["cooldown_until_interval_begin"],
        pending_signal_action=(
            None if row["pending_signal_action"] is None else str(row["pending_signal_action"])
        ),
        pending_regime_label=(
            None if row["pending_regime_label"] is None else str(row["pending_regime_label"])
        ),
        updated_at=row["updated_at"],
    )


def _coerce_mapping_payload(value: Any, *, field_name: str) -> Mapping[str, Any]:
    """Accept JSONB payloads returned as either native mappings or serialized JSON strings."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError(f"{field_name} must be a mapping-compatible payload")


def _decision_trace_from_row(row: Mapping[str, Any]) -> DecisionTraceSnapshot:
    trace_payload = DecisionTracePayload.model_validate(
        _coerce_mapping_payload(row["trace_payload"], field_name="trace_payload")
    )
    risk_payload = trace_payload.risk
    blocked_trade = trace_payload.blocked_trade
    threshold_snapshot = trace_payload.threshold_snapshot
    regime_reason = trace_payload.regime_reason
    prediction_explanation = trace_payload.prediction.prediction_explanation
    signal_explanation = trace_payload.signal.signal_explanation
    return DecisionTraceSnapshot(
        decision_trace_id=int(row["id"]),
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        signal=str(row["signal"]),
        signal_row_id=str(row["signal_row_id"]),
        signal_as_of_time=row["signal_as_of_time"],
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        risk_outcome=(
            None if row["risk_outcome"] is None else str(row["risk_outcome"])
        ),
        primary_reason_code=(
            None if risk_payload is None else risk_payload.primary_reason_code
        ),
        reason_texts=(
            ()
            if risk_payload is None
            else tuple(str(text) for text in risk_payload.reason_texts)
        ),
        blocked_stage=None if blocked_trade is None else blocked_trade.blocked_stage,
        json_report_path=(
            None if row["json_report_path"] is None else str(row["json_report_path"])
        ),
        markdown_report_path=(
            None
            if row["markdown_report_path"] is None
            else str(row["markdown_report_path"])
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        regime_label=(
            None
            if regime_reason is None
            else regime_reason.regime_label
        )
        or (
            None
            if threshold_snapshot is None
            else threshold_snapshot.regime_label
        ),
        regime_run_id=(
            None
            if regime_reason is None
            else regime_reason.regime_run_id
        )
        or (
            None
            if threshold_snapshot is None
            else threshold_snapshot.regime_run_id
        ),
        allow_new_long_entries=(
            None
            if threshold_snapshot is None
            else threshold_snapshot.allow_new_long_entries
        ),
        signal_reason_code=trace_payload.signal.reason_code,
        signal_freshness_status=trace_payload.signal.freshness_status,
        signal_health_overall_status=trace_payload.signal.health_overall_status,
        prediction_summary_text=(
            None
            if prediction_explanation is None
            else prediction_explanation.summary_text
        ),
        signal_summary_text=(
            None
            if signal_explanation is None
            else signal_explanation.summary_text
        ),
        top_feature_count=len(trace_payload.prediction.top_features),
        risk_reason_codes=(
            ()
            if risk_payload is None
            else tuple(str(code) for code in risk_payload.reason_codes)
        ),
        requested_notional=(
            None
            if risk_payload is None
            else risk_payload.requested_notional
        ),
        approved_notional=(
            None
            if risk_payload is None
            else risk_payload.approved_notional
        ),
        ordered_adjustment_count=(
            0
            if risk_payload is None
            else len(risk_payload.ordered_adjustments)
        ),
    )


def _order_audit_from_row(row: Mapping[str, Any]) -> OrderAuditSnapshot:
    return OrderAuditSnapshot(
        event_id=int(row["id"]),
        order_request_id=int(row["order_request_id"]),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        lifecycle_state=str(row["lifecycle_state"]),
        event_time=row["event_time"],
        reason_code=None if row["reason_code"] is None else str(row["reason_code"]),
        details=None if row["details"] is None else str(row["details"]),
        external_order_id=(
            None
            if row["external_order_id"] is None
            else str(row["external_order_id"])
        ),
        external_status=(
            None if row["external_status"] is None else str(row["external_status"])
        ),
        account_id=None if row["account_id"] is None else str(row["account_id"]),
        environment_name=(
            None if row["environment_name"] is None else str(row["environment_name"])
        ),
        broker_name=None if row["broker_name"] is None else str(row["broker_name"]),
        probe_policy_active=bool(row.get("probe_policy_active", False)),
        probe_symbol=(
            None
            if row.get("probe_symbol") is None
            else str(row.get("probe_symbol"))
        ),
        probe_qty=None if row.get("probe_qty") is None else int(row.get("probe_qty")),
        decision_trace_id=(
            None
            if row.get("decision_trace_id") is None
            else int(row.get("decision_trace_id"))
        ),
    )


def _live_safety_from_row(row: Mapping[str, Any]) -> LiveSafetySnapshot:
    return LiveSafetySnapshot(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        broker_name=str(row["broker_name"]),
        live_enabled=bool(row["live_enabled"]),
        startup_checks_passed=bool(row["startup_checks_passed"]),
        startup_checks_passed_at=row["startup_checks_passed_at"],
        account_validated=bool(row["account_validated"]),
        account_id=None if row["account_id"] is None else str(row["account_id"]),
        environment_name=(
            None if row["environment_name"] is None else str(row["environment_name"])
        ),
        manual_disable_active=bool(row["manual_disable_active"]),
        consecutive_live_failures=int(row["consecutive_live_failures"]),
        failure_hard_stop_active=bool(row["failure_hard_stop_active"]),
        last_failure_reason=(
            None
            if row["last_failure_reason"] is None
            else str(row["last_failure_reason"])
        ),
        updated_at=row["updated_at"],
        system_health_status=(
            None
            if row.get("system_health_status") is None
            else str(row.get("system_health_status"))
        ),
        system_health_reason_code=(
            None
            if row.get("system_health_reason_code") is None
            else str(row.get("system_health_reason_code"))
        ),
        system_health_checked_at=row.get("system_health_checked_at"),
        health_gate_status=(
            None
            if row.get("health_gate_status") is None
            else str(row.get("health_gate_status"))
        ),
        health_gate_reason_code=(
            None
            if row.get("health_gate_reason_code") is None
            else str(row.get("health_gate_reason_code"))
        ),
        health_gate_detail=(
            None
            if row.get("health_gate_detail") is None
            else str(row.get("health_gate_detail"))
        ),
        broker_cash=(
            None
            if row.get("broker_cash") is None
            else float(row.get("broker_cash"))
        ),
        broker_equity=(
            None
            if row.get("broker_equity") is None
            else float(row.get("broker_equity"))
        ),
        reconciliation_status=(
            None
            if row.get("reconciliation_status") is None
            else str(row.get("reconciliation_status"))
        ),
        reconciliation_reason_code=(
            None
            if row.get("reconciliation_reason_code") is None
            else str(row.get("reconciliation_reason_code"))
        ),
        reconciliation_checked_at=row.get("reconciliation_checked_at"),
        unresolved_incident_count=int(row.get("unresolved_incident_count", 0)),
        can_submit_live_now=bool(row.get("can_submit_live_now", False)),
        primary_block_reason_code=(
            None
            if row.get("primary_block_reason_code") is None
            else str(row.get("primary_block_reason_code"))
        ),
        block_detail=(
            None if row.get("block_detail") is None else str(row.get("block_detail"))
        ),
    )


def _reliability_state_from_row(row: Mapping[str, Any]) -> ReliabilityStateSnapshot:
    return ReliabilityStateSnapshot(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        health_overall_status=str(row["health_overall_status"]),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=str(row["breaker_state"]),
        failure_count=int(row["failure_count"]),
        success_count=int(row["success_count"]),
        reason_code=None if row["reason_code"] is None else str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
        updated_at=row["updated_at"],
    )


def _recovery_event_from_row(row: Mapping[str, Any]) -> RecoveryEventSnapshot:
    event_time = row["event_time"]
    if isinstance(event_time, str):
        event_time = parse_rfc3339(event_time)
    detail = row.get("details", row.get("detail"))
    return RecoveryEventSnapshot(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        event_type=str(row["event_type"]),
        event_time=event_time,
        reason_code=str(row["reason_code"]),
        health_overall_status=(
            None
            if row["health_overall_status"] is None
            else str(row["health_overall_status"])
        ),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=(
            None if row["breaker_state"] is None else str(row["breaker_state"])
        ),
        detail=None if detail is None else str(detail),
    )
