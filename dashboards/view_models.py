"""Pure view-model helpers for the Stream Alpha M6 dashboard."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.common.time import to_rfc3339, utc_now
from app.trading.config import PaperTradingConfig
from app.trading.metrics import build_summary
from app.trading.risk import calculate_fee
from app.trading.schemas import PaperPosition

from dashboards.data_sources import (
    DashboardSnapshot,
    EngineStateSnapshot,
    LatestFeatureSnapshot,
    LedgerEntrySnapshot,
    LiveSafetySnapshot,
    OrderAuditSnapshot,
    SignalSnapshot,
)


@dataclass(frozen=True, slots=True)
class TradingOverview:
    """Overview KPI payload displayed on the M6 dashboard."""

    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    max_drawdown: float
    open_position_count: int
    win_rate: float
    average_trade_return: float
    turnover: float
    sharpe_like: float
    cash_balance: float
    hit_rate_by_asset: dict[str, float]
    hit_rate_by_regime: dict[str, float]
    realized_pnl_by_regime: dict[str, float]
    closed_position_count_by_regime: dict[str, int]


@dataclass(frozen=True, slots=True)
class TraderFreshness:
    """Compact freshness/state summary for the M5 paper trader."""

    state: str
    symbols_tracked: int
    latest_processed_interval_begin: datetime | None
    slowest_processed_interval_begin: datetime | None
    pending_signal_count: int
    latest_update_at: datetime | None
    message: str


def build_overview_metrics(
    *,
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
) -> TradingOverview | None:
    """Build consistent headline KPIs from the accepted M5 persistence model."""
    database = snapshot.database
    if not database.available or database.cash_balance is None:
        return None

    summary = build_summary(
        config=trading_config,
        positions=list(database.positions),
        latest_prices=database.latest_prices,
        cash_balance=database.cash_balance,
    )
    overall = summary["overall"]
    return TradingOverview(
        realized_pnl=float(overall["cumulative_pnl_realized"]),
        unrealized_pnl=float(overall["cumulative_pnl_total"])
        - float(overall["cumulative_pnl_realized"]),
        total_pnl=float(overall["cumulative_pnl_total"]),
        max_drawdown=float(overall["max_drawdown"]),
        open_position_count=int(overall["open_position_count"]),
        win_rate=float(overall["win_rate"]),
        average_trade_return=float(overall["average_trade_return"]),
        turnover=float(overall["turnover"]),
        sharpe_like=float(overall["sharpe_like"]),
        cash_balance=float(overall["cash_balance"]),
        hit_rate_by_asset=dict(overall["hit_rate_by_asset"]),
        hit_rate_by_regime=dict(overall["hit_rate_by_regime"]),
        realized_pnl_by_regime=dict(overall["realized_pnl_by_regime"]),
        closed_position_count_by_regime=dict(overall["closed_position_count_by_regime"]),
    )


def build_latest_signal_rows(
    *,
    symbols: tuple[str, ...],
    signals: tuple[SignalSnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a stable latest-signal table in configured asset order."""
    reference_time = utc_now() if now is None else now
    signals_by_symbol = {row.symbol: row for row in signals}
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        signal = signals_by_symbol.get(symbol)
        if signal is None or not signal.available:
            rows.append(
                {
                    "symbol": symbol,
                    "signal": "UNAVAILABLE",
                    "prob_up": None,
                    "confidence": None,
                    "predicted_class": None,
                    "as_of_time": None,
                    "age": None,
                    "reason": None if signal is None else signal.error,
                    "model_name": None if signal is None else signal.model_name,
                    "regime_label": None if signal is None else signal.regime_label,
                    "regime_run_id": None if signal is None else signal.regime_run_id,
                    "trade_allowed": None if signal is None else signal.trade_allowed,
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "signal": signal.signal,
                "prob_up": round_or_none(signal.prob_up, 4),
                "confidence": round_or_none(signal.confidence, 4),
                "predicted_class": signal.predicted_class,
                "as_of_time": (
                    to_rfc3339(signal.as_of_time) if signal.as_of_time is not None else None
                ),
                "age": age_text(signal.as_of_time, reference_time),
                "reason": signal.reason,
                "model_name": signal.model_name,
                "regime_label": signal.regime_label,
                "regime_run_id": signal.regime_run_id,
                "trade_allowed": signal.trade_allowed,
            }
        )
    return rows


def build_feature_snapshot_rows(
    *,
    symbols: tuple[str, ...],
    features: tuple[LatestFeatureSnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a small stable latest-feature table in configured asset order."""
    reference_time = utc_now() if now is None else now
    features_by_symbol = {row.symbol: row for row in features}
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        feature = features_by_symbol.get(symbol)
        if feature is None:
            rows.append(
                {
                    "symbol": symbol,
                    "interval_begin": None,
                    "as_of_time": None,
                    "age": None,
                    "open_price": None,
                    "high_price": None,
                    "low_price": None,
                    "close_price": None,
                    "volume": None,
                    "log_return_1": None,
                    "log_return_3": None,
                    "rsi_14": None,
                    "macd_line_12_26": None,
                    "close_zscore_12": None,
                    "volume_zscore_12": None,
                }
            )
            continue
        rows.append(
            {
                "symbol": symbol,
                "interval_begin": to_rfc3339(feature.interval_begin),
                "as_of_time": to_rfc3339(feature.as_of_time),
                "age": age_text(feature.as_of_time, reference_time),
                "open_price": round(feature.open_price, 6),
                "high_price": round(feature.high_price, 6),
                "low_price": round(feature.low_price, 6),
                "close_price": round(feature.close_price, 6),
                "volume": round(feature.volume, 6),
                "log_return_1": round(feature.log_return_1, 6),
                "log_return_3": round(feature.log_return_3, 6),
                "rsi_14": round(feature.rsi_14, 6),
                "macd_line_12_26": round(feature.macd_line_12_26, 6),
                "close_zscore_12": round(feature.close_zscore_12, 6),
                "volume_zscore_12": round(feature.volume_zscore_12, 6),
            }
        )
    return rows


def build_open_position_rows(
    *,
    positions: tuple[PaperPosition, ...],
    latest_prices: dict[str, float],
    fee_bps: float,
) -> list[dict[str, Any]]:
    """Build the open-positions trading table with live unrealized PnL."""
    rows: list[dict[str, Any]] = []
    for position in sorted(
        (row for row in positions if row.status == "OPEN"),
        key=lambda row: row.entry_fill_time,
        reverse=True,
    ):
        latest_price = latest_prices.get(position.symbol)
        unrealized_pnl = None
        if latest_price is not None:
            exit_notional = latest_price * position.quantity
            exit_fee = calculate_fee(exit_notional, fee_bps)
            unrealized_pnl = (
                exit_notional - exit_fee
            ) - (
                position.entry_notional + position.entry_fee
            )
        rows.append(
            {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "entry_fill_time": to_rfc3339(position.entry_fill_time),
                "entry_price": round(position.entry_price, 6),
                "quantity": round(position.quantity, 8),
                "entry_notional": round(position.entry_notional, 6),
                "latest_price": round_or_none(latest_price, 6),
                "unrealized_pnl": round_or_none(unrealized_pnl, 6),
                "stop_loss_price": round(position.stop_loss_price, 6),
                "take_profit_price": round(position.take_profit_price, 6),
                "entry_regime_label": position.entry_regime_label,
                "entry_model_name": position.entry_model_name,
                "entry_signal_row_id": position.entry_signal_row_id,
            }
        )
    return rows


def build_recent_closed_trade_rows(
    positions: tuple[PaperPosition, ...],
) -> list[dict[str, Any]]:
    """Build the recent closed-trades table from persisted paper positions."""
    rows: list[dict[str, Any]] = []
    for position in positions:
        rows.append(
            {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "closed_at": (
                    None if position.closed_at is None else to_rfc3339(position.closed_at)
                ),
                "entry_price": round(position.entry_price, 6),
                "exit_price": round_or_none(position.exit_price, 6),
                "realized_pnl": round_or_none(position.realized_pnl, 6),
                "realized_return": round_or_none(position.realized_return, 6),
                "exit_reason": position.exit_reason,
                "entry_regime_label": position.entry_regime_label,
                "exit_regime_label": position.exit_regime_label,
                "entry_model_name": position.entry_model_name,
                "exit_model_name": position.exit_model_name,
            }
        )
    return rows


def build_recent_ledger_rows(
    entries: tuple[LedgerEntrySnapshot, ...],
) -> list[dict[str, Any]]:
    """Build the recent-ledger table for the trading tab."""
    return [
        {
            "ledger_id": entry.ledger_id,
            "symbol": entry.symbol,
            "action": entry.action,
            "reason": entry.reason,
            "fill_time": to_rfc3339(entry.fill_time),
            "fill_price": round(entry.fill_price, 6),
            "quantity": round(entry.quantity, 8),
            "notional": round(entry.notional, 6),
            "fee": round(entry.fee, 6),
            "cash_flow": round(entry.cash_flow, 6),
            "realized_pnl": round_or_none(entry.realized_pnl, 6),
            "confidence": round_or_none(entry.confidence, 4),
            "model_name": entry.model_name,
            "signal_row_id": entry.signal_row_id,
            "regime_label": entry.regime_label,
        }
        for entry in entries
    ]


def build_recent_order_audit_rows(
    entries: tuple[OrderAuditSnapshot, ...],
) -> list[dict[str, Any]]:
    """Build the recent order-audit table for the trading tab."""
    return [
        {
            "event_id": entry.event_id,
            "order_request_id": entry.order_request_id,
            "symbol": entry.symbol,
            "action": entry.action,
            "lifecycle_state": entry.lifecycle_state,
            "event_time": to_rfc3339(entry.event_time),
            "reason_code": entry.reason_code,
            "details": entry.details,
            "broker_name": entry.broker_name,
            "external_status": entry.external_status,
        }
        for entry in entries
    ]


def build_live_status_rows(
    *,
    trading_config: PaperTradingConfig,
    live_safety_state: LiveSafetySnapshot | None,
) -> list[dict[str, Any]]:
    """Build a compact guarded-live status table for the dashboard."""
    if live_safety_state is None:
        return [
            {
                "execution_mode": trading_config.execution.mode,
                "broker_name": "alpaca",
                "startup_checks_passed": False,
                "startup_checks_passed_at": None,
                "manual_disable_active": None,
                "failure_hard_stop_active": None,
                "consecutive_live_failures": None,
                "expected_environment": trading_config.execution.live.expected_environment,
                "validated_environment": None,
                "expected_account_id": trading_config.execution.live.expected_account_id,
                "validated_account_id": None,
                "symbol_whitelist": ",".join(trading_config.execution.live.symbol_whitelist),
                "live_max_order_notional": trading_config.execution.live.max_order_notional,
                "last_failure_reason": None,
            }
        ]

    return [
        {
            "execution_mode": live_safety_state.execution_mode,
            "broker_name": live_safety_state.broker_name,
            "startup_checks_passed": live_safety_state.startup_checks_passed,
            "startup_checks_passed_at": (
                None
                if live_safety_state.startup_checks_passed_at is None
                else to_rfc3339(live_safety_state.startup_checks_passed_at)
            ),
            "manual_disable_active": live_safety_state.manual_disable_active,
            "failure_hard_stop_active": live_safety_state.failure_hard_stop_active,
            "consecutive_live_failures": live_safety_state.consecutive_live_failures,
            "expected_environment": trading_config.execution.live.expected_environment,
            "validated_environment": live_safety_state.environment_name,
            "expected_account_id": trading_config.execution.live.expected_account_id,
            "validated_account_id": live_safety_state.account_id,
            "symbol_whitelist": ",".join(trading_config.execution.live.symbol_whitelist),
            "live_max_order_notional": trading_config.execution.live.max_order_notional,
            "last_failure_reason": live_safety_state.last_failure_reason,
        }
    ]


def build_performance_by_regime_rows(
    *,
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
) -> list[dict[str, Any]]:
    """Build a compact by-regime performance table from persisted M5 state."""
    if not snapshot.database.available or snapshot.database.cash_balance is None:
        return []

    summary = build_summary(
        config=trading_config,
        positions=list(snapshot.database.positions),
        latest_prices=snapshot.database.latest_prices,
        cash_balance=snapshot.database.cash_balance,
    )
    return [
        {
            "regime_label": row["regime_label"],
            "open_positions": int(row["open_positions"]),
            "closed_positions": int(row["closed_positions"]),
            "realized_pnl": round(float(row["realized_pnl"]), 6),
            "total_pnl": round(float(row["total_pnl"]), 6),
            "win_rate": round(float(row["win_rate"]), 4),
        }
        for row in summary["by_regime"]
    ]


def build_trader_freshness(
    engine_states: tuple[EngineStateSnapshot, ...],
) -> TraderFreshness:
    """Build a compact freshness summary from persisted paper-engine state."""
    if not engine_states:
        return TraderFreshness(
            state="unavailable",
            symbols_tracked=0,
            latest_processed_interval_begin=None,
            slowest_processed_interval_begin=None,
            pending_signal_count=0,
            latest_update_at=None,
            message="No paper trader state rows were found",
        )

    processed = [
        row.last_processed_interval_begin
        for row in engine_states
        if row.last_processed_interval_begin
    ]
    updates = [row.updated_at for row in engine_states]
    pending_count = sum(int(row.pending_signal_action is not None) for row in engine_states)
    latest_processed = max(processed) if processed else None
    slowest_processed = min(processed) if processed else None
    latest_update = max(updates) if updates else None
    return TraderFreshness(
        state="healthy",
        symbols_tracked=len(engine_states),
        latest_processed_interval_begin=latest_processed,
        slowest_processed_interval_begin=slowest_processed,
        pending_signal_count=pending_count,
        latest_update_at=latest_update,
        message=f"Tracked {len(engine_states)} symbols with {pending_count} pending signals",
    )


def build_equity_curve_rows(
    *,
    positions: tuple[PaperPosition, ...],
    initial_cash: float,
    latest_prices: dict[str, float],
    fee_bps: float,
    as_of_time: datetime | None,
) -> list[dict[str, Any]]:
    """Build an equity/PnL line series from persisted positions plus latest prices."""
    closed_positions = sorted(
        (row for row in positions if row.status == "CLOSED"),
        key=lambda row: row.exit_fill_time or row.entry_fill_time,
    )
    equity = initial_cash
    rows = [
        {
            "timestamp": "START",
            "equity": round(equity, 6),
            "cumulative_pnl": 0.0,
        }
    ]
    for position in closed_positions:
        equity += position.realized_pnl or 0.0
        rows.append(
            {
                "timestamp": to_rfc3339(position.exit_fill_time or position.entry_fill_time),
                "equity": round(equity, 6),
                "cumulative_pnl": round(equity - initial_cash, 6),
            }
        )

    open_positions = [row for row in positions if row.status == "OPEN"]
    if open_positions:
        unrealized_pnl = 0.0
        for position in open_positions:
            latest_price = latest_prices.get(position.symbol)
            if latest_price is None:
                continue
            exit_notional = latest_price * position.quantity
            exit_fee = calculate_fee(exit_notional, fee_bps)
            unrealized_pnl += (
                exit_notional - exit_fee
            ) - (
                position.entry_notional + position.entry_fee
            )
        final_equity = equity + unrealized_pnl
    else:
        final_equity = equity

    if as_of_time is not None:
        rows.append(
            {
                "timestamp": to_rfc3339(as_of_time),
                "equity": round(final_equity, 6),
                "cumulative_pnl": round(final_equity - initial_cash, 6),
            }
        )
    return rows


def build_drawdown_curve_rows(
    equity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a drawdown line series from the portfolio equity curve."""
    peak = 0.0
    rows: list[dict[str, Any]] = []
    for row in equity_rows:
        equity = float(row["equity"])
        peak = max(peak, equity)
        drawdown = 0.0 if peak <= 0 else (peak - equity) / peak
        rows.append(
            {
                "timestamp": row["timestamp"],
                "drawdown": round(drawdown, 6),
            }
        )
    return rows


def latest_feature_as_of(features: tuple[LatestFeatureSnapshot, ...]) -> datetime | None:
    """Return the newest canonical feature timestamp across assets."""
    if not features:
        return None
    return max(row.as_of_time for row in features)


def age_text(timestamp: datetime | None, reference_time: datetime | None = None) -> str | None:
    """Render a compact age string for the dashboard."""
    if timestamp is None:
        return None
    now = utc_now() if reference_time is None else reference_time
    total_seconds = max(0, int((now - timestamp).total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def round_or_none(value: float | None, digits: int) -> float | None:
    """Round numeric values while preserving missing data."""
    if value is None:
        return None
    return round(value, digits)
