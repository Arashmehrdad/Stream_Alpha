"""Typed state and event models for the Stream Alpha M5 paper trader."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from app.common.time import to_rfc3339


SignalAction = Literal["BUY", "SELL", "HOLD"]
PositionStatus = Literal["OPEN", "CLOSED"]
ExitReason = Literal["SELL_SIGNAL", "STOP_LOSS", "TAKE_PROFIT"]
TradeAction = Literal["BUY", "SELL"]


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


@dataclass(frozen=True, slots=True)
class PaperEngineState:
    """Restart-safe per-symbol engine state."""

    service_name: str
    symbol: str
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
    entry_regime_label: str | None = None
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
    position_id: int | None = None
    signal_interval_begin: datetime | None = None
    signal_as_of_time: datetime | None = None
    signal_row_id: str | None = None
    model_name: str | None = None
    prob_up: float | None = None
    prob_down: float | None = None
    confidence: float | None = None
    regime_label: str | None = None
    realized_pnl: float | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    """Portfolio-wide context needed by the pure engine."""

    available_cash: float
    open_position_count: int


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Result of processing one finalized candle exactly once."""

    state: PaperEngineState
    open_position: PaperPosition | None
    created_position: PaperPosition | None = None
    closed_position: PaperPosition | None = None
    ledger_entries: tuple[TradeLedgerEntry, ...] = field(default_factory=tuple)
    cash_delta: float = 0.0
