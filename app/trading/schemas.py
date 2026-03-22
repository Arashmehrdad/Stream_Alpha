"""Typed state and event models for the Stream Alpha M5 paper trader."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from app.adaptation.schemas import AdaptiveRecentPerformanceSummary
from app.common.time import to_rfc3339
from app.explainability.schemas import (
    DecisionTracePayload,
    PredictionExplanation,
    RegimeReason,
    SignalExplanation,
    ThresholdSnapshot,
    TopFeatureContribution,
)


SignalAction = Literal["BUY", "SELL", "HOLD"]
PositionStatus = Literal["OPEN", "CLOSED"]
ExitReason = Literal["SELL_SIGNAL", "STOP_LOSS", "TAKE_PROFIT"]
TradeAction = Literal["BUY", "SELL"]
RiskOutcome = Literal["APPROVED", "MODIFIED", "BLOCKED"]
ExecutionMode = Literal["paper", "shadow", "live"]
OrderLifecycleState = Literal[
    "CREATED",
    "SUBMITTED",
    "ACCEPTED",
    "PARTIALLY_FILLED",
    "FILLED",
    "REJECTED",
    "CANCELED",
    "FAILED",
]


@dataclass(frozen=True, slots=True)
class FeatureCandle:  # pylint: disable=too-many-instance-attributes
    """Canonical finalized feature row subset required by M5."""

    id: int
    source_exchange: str
    symbol: str
    interval_minutes: int
    interval_begin: datetime
    interval_end: datetime
    as_of_time: datetime
    raw_event_id: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    realized_vol_12: float = 0.0

    @property
    def row_id(self) -> str:
        """Return the stable candle identifier used across M4 and M5."""
        return f"{self.symbol}|{to_rfc3339(self.interval_begin)}"


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Authoritative signal payload returned by the M4 inference service."""

    symbol: str
    signal: SignalAction
    reason: str
    prob_up: float
    prob_down: float
    confidence: float
    predicted_class: str
    row_id: str
    as_of_time: datetime
    model_name: str
    model_version: str | None = None
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None
    signal_status: str | None = None
    decision_source: str | None = None
    reason_code: str | None = None
    freshness_status: str | None = None
    health_overall_status: str | None = None
    approved_notional: float | None = None
    risk_outcome: RiskOutcome | None = None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    top_features: tuple[TopFeatureContribution, ...] = field(default_factory=tuple)
    prediction_explanation: PredictionExplanation | None = None
    threshold_snapshot: ThresholdSnapshot | None = None
    regime_reason: RegimeReason | None = None
    signal_explanation: SignalExplanation | None = None
    adaptation_profile_id: str | None = None
    calibrated_confidence: float | None = None
    effective_buy_prob_up: float | None = None
    effective_sell_prob_up: float | None = None
    adaptation_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    adaptive_size_multiplier: float | None = None
    drift_status: str | None = None
    recent_performance_summary: AdaptiveRecentPerformanceSummary | None = None
    frozen_by_health_gate: bool = False


@dataclass(frozen=True, slots=True)
class PendingSignalState:
    """Persisted signal waiting to fill at the next candle open."""

    signal: SignalAction
    signal_interval_begin: datetime
    signal_as_of_time: datetime
    row_id: str
    reason: str
    prob_up: float
    prob_down: float
    confidence: float
    predicted_class: str
    model_name: str
    regime_label: str | None = None
    regime_run_id: str | None = None
    approved_notional: float | None = None
    risk_outcome: RiskOutcome | None = None
    order_request_id: int | None = None
    order_request_idempotency_key: str | None = None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    adaptation_profile_id: str | None = None
    calibrated_confidence: float | None = None
    adaptation_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    adaptive_size_multiplier: float | None = None


@dataclass(frozen=True, slots=True)
class PaperEngineState:
    """Restart-safe per-symbol engine state."""

    service_name: str
    symbol: str
    execution_mode: ExecutionMode = "paper"
    last_processed_interval_begin: datetime | None = None
    cooldown_until_interval_begin: datetime | None = None
    pending_signal: PendingSignalState | None = None
    pending_decision_trace_id: int | None = None


@dataclass(frozen=True, slots=True)
class PaperPosition:  # pylint: disable=too-many-instance-attributes
    """Persisted paper position state for one long-only spot trade."""

    service_name: str
    symbol: str
    status: PositionStatus
    entry_signal_interval_begin: datetime
    entry_signal_as_of_time: datetime
    entry_signal_row_id: str
    entry_reason: str
    entry_model_name: str
    entry_prob_up: float
    entry_confidence: float
    entry_fill_interval_begin: datetime
    entry_fill_time: datetime
    entry_price: float
    quantity: float
    entry_notional: float
    entry_fee: float
    stop_loss_price: float
    take_profit_price: float
    execution_mode: ExecutionMode = "paper"
    entry_regime_label: str | None = None
    entry_approved_notional: float | None = None
    entry_risk_outcome: RiskOutcome | None = None
    entry_order_request_id: int | None = None
    entry_decision_trace_id: int | None = None
    entry_risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    position_id: int | None = None
    exit_reason: ExitReason | None = None
    exit_signal_interval_begin: datetime | None = None
    exit_signal_as_of_time: datetime | None = None
    exit_signal_row_id: str | None = None
    exit_model_name: str | None = None
    exit_prob_up: float | None = None
    exit_confidence: float | None = None
    exit_fill_interval_begin: datetime | None = None
    exit_fill_time: datetime | None = None
    exit_price: float | None = None
    exit_notional: float | None = None
    exit_fee: float | None = None
    realized_pnl: float | None = None
    realized_return: float | None = None
    exit_regime_label: str | None = None
    exit_order_request_id: int | None = None
    exit_decision_trace_id: int | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TradeLedgerEntry:  # pylint: disable=too-many-instance-attributes
    """Persisted simulated fill record."""

    service_name: str
    symbol: str
    action: TradeAction
    reason: str
    fill_interval_begin: datetime
    fill_time: datetime
    fill_price: float
    quantity: float
    notional: float
    fee: float
    slippage_bps: float
    cash_flow: float
    execution_mode: ExecutionMode = "paper"
    position_id: int | None = None
    order_request_id: int | None = None
    decision_trace_id: int | None = None
    signal_interval_begin: datetime | None = None
    signal_as_of_time: datetime | None = None
    signal_row_id: str | None = None
    model_name: str | None = None
    prob_up: float | None = None
    prob_down: float | None = None
    confidence: float | None = None
    regime_label: str | None = None
    approved_notional: float | None = None
    risk_outcome: RiskOutcome | None = None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    realized_pnl: float | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    """Portfolio-wide context needed by the pure engine."""

    available_cash: float
    open_position_count: int
    current_equity: float = 0.0
    total_open_exposure_notional: float = 0.0
    current_symbol_exposure_notional: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Pure risk-engine output for one fetched signal."""

    service_name: str
    symbol: str
    signal: SignalAction
    outcome: RiskOutcome
    approved_notional: float
    requested_notional: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    primary_reason_code: str | None = None
    reason_texts: tuple[str, ...] = field(default_factory=tuple)
    ordered_adjustments: tuple[RiskAdjustmentStep, ...] = field(default_factory=tuple)
    blocked_stage: str | None = None
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None


@dataclass(frozen=True, slots=True)
class RiskAdjustmentStep:
    """One ordered M10 sizing adjustment step."""

    step_index: int
    reason_code: str
    reason_text: str
    before_notional: float
    after_notional: float


@dataclass(frozen=True, slots=True)
class ServiceRiskState:
    """Restart-safe service-level risk state used for M10 guards."""

    service_name: str
    trading_day: date
    realized_pnl_today: float
    equity_high_watermark: float
    current_equity: float
    loss_streak_count: int
    execution_mode: ExecutionMode = "paper"
    loss_streak_cooldown_until_interval_begin: datetime | None = None
    kill_switch_enabled: bool = False
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LiveStartupCheck:
    """One explicit M12 startup validation check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class LiveStartupChecklist:
    """Redacted M12 startup checklist artifact payload."""

    service_name: str
    execution_mode: ExecutionMode
    broker_name: str
    checked_at: datetime
    passed: bool
    expected_account_id: str | None
    validated_account_id: str | None
    expected_environment: str
    validated_environment: str | None
    live_enabled: bool
    runtime_armed: bool
    runtime_confirmation_phrase: str
    manual_disable_path: str
    symbol_whitelist: tuple[str, ...]
    max_order_notional: float
    failure_hard_stop_threshold: int
    checks: tuple[LiveStartupCheck, ...]


@dataclass(frozen=True, slots=True)
class LiveSafetyState:
    """Persisted M12 live execution safety state."""

    service_name: str
    execution_mode: ExecutionMode
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
    last_failure_reason: str | None = None
    system_health_status: str = "UNKNOWN"
    system_health_reason_code: str | None = None
    system_health_checked_at: datetime | None = None
    health_gate_status: str = "UNKNOWN"
    health_gate_reason_code: str | None = None
    health_gate_detail: str | None = None
    broker_cash: float | None = None
    broker_equity: float | None = None
    reconciliation_status: str = "UNKNOWN"
    reconciliation_reason_code: str | None = None
    reconciliation_checked_at: datetime | None = None
    unresolved_incident_count: int = 0
    can_submit_live_now: bool = False
    primary_block_reason_code: str | None = None
    block_detail: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CanonicalRecoveryEvent:
    """Canonical recovery event summary from the M13 system-health endpoint."""

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
class CanonicalServiceHealth:
    """Per-service canonical health summary used by the live health gate."""

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
class CanonicalFeatureLag:
    """Per-symbol canonical feature lag summary used by the live health gate."""

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
class CanonicalSystemReliability:
    """Canonical M13 system-health snapshot consumed by the live gate."""

    service_name: str
    checked_at: datetime
    health_overall_status: str
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    lag_breach_active: bool = False
    services: tuple[CanonicalServiceHealth, ...] = field(default_factory=tuple)
    lag_by_symbol: tuple[CanonicalFeatureLag, ...] = field(default_factory=tuple)
    latest_recovery_event: CanonicalRecoveryEvent | None = None


@dataclass(frozen=True, slots=True)
class RiskDecisionLogEntry:
    """Persisted audit row for one evaluated M4 signal."""

    service_name: str
    symbol: str
    signal: SignalAction
    signal_interval_begin: datetime
    signal_as_of_time: datetime
    signal_row_id: str
    outcome: RiskOutcome
    reason_codes: tuple[str, ...]
    requested_notional: float
    approved_notional: float
    available_cash: float
    current_equity: float
    current_symbol_exposure_notional: float
    total_open_exposure_notional: float
    realized_vol_12: float
    confidence: float
    execution_mode: ExecutionMode = "paper"
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None
    decision_trace_id: int | None = None
    model_version: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    """Normalized broker-account validation payload for guarded live startup."""

    broker_name: str
    account_id: str
    environment_name: str
    status: str | None = None
    cash: float | None = None
    equity: float | None = None


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    """Minimal broker-truth order snapshot used by live reconciliation."""

    broker_name: str
    external_order_id: str
    symbol: str
    side: str
    status: str
    account_id: str | None = None
    environment_name: str | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    qty: float | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    """Minimal broker-truth open-position snapshot used by live reconciliation."""

    broker_name: str
    symbol: str
    quantity: float
    avg_entry_price: float | None = None
    market_value: float | None = None
    account_id: str | None = None
    environment_name: str | None = None
    side: str = "long"


@dataclass(frozen=True, slots=True)
class BrokerSubmitResult:
    """Normalized broker submit response used by the live execution adapter."""

    broker_name: str
    external_order_id: str
    external_status: str
    account_id: str
    environment_name: str
    details: str | None = None
    probe_policy_active: bool = False
    probe_symbol: str | None = None
    probe_qty: int | None = None


@dataclass(frozen=True, slots=True)
class OrderRequest:  # pylint: disable=too-many-instance-attributes
    """Deterministic execution request created after M10 risk approval."""

    service_name: str
    execution_mode: ExecutionMode
    symbol: str
    action: TradeAction
    signal_interval_begin: datetime
    signal_as_of_time: datetime
    signal_row_id: str
    target_fill_interval_begin: datetime
    requested_notional: float
    approved_notional: float
    idempotency_key: str
    model_name: str | None = None
    confidence: float | None = None
    regime_label: str | None = None
    regime_run_id: str | None = None
    risk_outcome: RiskOutcome | None = None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    decision_trace_id: int | None = None
    model_version: str | None = None
    order_request_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class DecisionTraceRecord:
    """Canonical persisted M14 decision trace row."""

    service_name: str
    execution_mode: ExecutionMode
    symbol: str
    signal: SignalAction
    signal_interval_begin: datetime
    signal_as_of_time: datetime
    signal_row_id: str
    model_name: str
    model_version: str
    payload: DecisionTracePayload
    risk_outcome: RiskOutcome | None = None
    json_report_path: str | None = None
    markdown_report_path: str | None = None
    decision_trace_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class OrderLifecycleEvent:
    """Explicit lifecycle event for one order request."""

    order_request_id: int
    service_name: str
    execution_mode: ExecutionMode
    symbol: str
    action: TradeAction
    lifecycle_state: OrderLifecycleState
    event_time: datetime
    reason_code: str | None = None
    details: str | None = None
    external_order_id: str | None = None
    external_status: str | None = None
    account_id: str | None = None
    environment_name: str | None = None
    broker_name: str | None = None
    probe_policy_active: bool = False
    probe_symbol: str | None = None
    probe_qty: int | None = None
    decision_trace_id: int | None = None
    event_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Adapter result for one execution-phase candle evaluation."""

    state: PaperEngineState
    open_position: PaperPosition | None
    lifecycle_events: tuple[OrderLifecycleEvent, ...] = field(default_factory=tuple)
    created_position: PaperPosition | None = None
    closed_position: PaperPosition | None = None
    ledger_entries: tuple[TradeLedgerEntry, ...] = field(default_factory=tuple)
    cash_delta: float = 0.0
    live_safety_state: LiveSafetyState | None = None


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Result of processing one finalized candle exactly once."""

    state: PaperEngineState
    open_position: PaperPosition | None
    created_position: PaperPosition | None = None
    closed_position: PaperPosition | None = None
    ledger_entries: tuple[TradeLedgerEntry, ...] = field(default_factory=tuple)
    cash_delta: float = 0.0
