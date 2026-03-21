"""Typed state and event models for the Stream Alpha M5 paper trader."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from app.common.time import to_rfc3339


SignalAction = Literal["BUY", "SELL", "HOLD"]
PositionStatus = Literal["OPEN", "CLOSED"]
ExitReason = Literal["SELL_SIGNAL", "STOP_LOSS", "TAKE_PROFIT"]
TradeAction = Literal["BUY", "SELL"]
RiskOutcome = Literal["APPROVED", "MODIFIED", "BLOCKED"]
ExecutionMode = Literal["paper", "shadow", "live"]
OrderLifecycleState = Literal["CREATED", "ACCEPTED", "FILLED", "REJECTED"]


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
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None
    approved_notional: float | None = None
    risk_outcome: RiskOutcome | None = None
    risk_reason_codes: tuple[str, ...] = field(default_factory=tuple)


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


@dataclass(frozen=True, slots=True)
class PaperEngineState:
    """Restart-safe per-symbol engine state."""

    service_name: str
    symbol: str
    execution_mode: ExecutionMode = "paper"
    last_processed_interval_begin: datetime | None = None
    cooldown_until_interval_begin: datetime | None = None
    pending_signal: PendingSignalState | None = None


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
    regime_label: str | None = None
    regime_run_id: str | None = None
    trade_allowed: bool | None = None


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
    updated_at: datetime | None = None


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
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    """Normalized broker-account validation payload for guarded live startup."""

    broker_name: str
    account_id: str
    environment_name: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class BrokerSubmitResult:
    """Normalized broker submit response used by the live execution adapter."""

    broker_name: str
    external_order_id: str
    external_status: str
    account_id: str
    environment_name: str
    details: str | None = None


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
    order_request_id: int | None = None
    created_at: datetime | None = None


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
