"""Pure view-model helpers for the Stream Alpha M6 dashboard."""

# pylint: disable=too-many-instance-attributes,too-many-lines
# pylint: disable=too-many-branches,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.common.config import Settings
from app.common.time import to_rfc3339, utc_now
from app.trading.config import PaperTradingConfig
from app.trading.metrics import build_summary
from app.trading.risk import calculate_fee
from app.trading.schemas import PaperPosition

from dashboards.data_sources import (
    DashboardSnapshot,
    DecisionTraceSnapshot,
    EngineStateSnapshot,
    FeatureLagSummarySnapshot,
    FreshnessSnapshot,
    LatestFeatureSnapshot,
    LedgerEntrySnapshot,
    LiveSafetySnapshot,
    OrderAuditSnapshot,
    RecoveryEventSnapshot,
    ReliabilityStateSnapshot,
    SignalSnapshot,
    SystemReliabilitySnapshot,
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
                    "signal_status": None if signal is None else signal.signal_status,
                    "decision_source": None if signal is None else signal.decision_source,
                    "reason_code": None if signal is None else signal.reason_code,
                    "freshness_status": None if signal is None else signal.freshness_status,
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
                "signal_status": signal.signal_status,
                "decision_source": signal.decision_source,
                "reason_code": signal.reason_code,
                "freshness_status": signal.freshness_status,
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
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build the recent closed-trades table from persisted paper positions."""
    reference_time = utc_now() if now is None else now
    rows: list[dict[str, Any]] = []
    for position in positions:
        closed_at = position.closed_at or position.exit_fill_time
        rows.append(
            {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "closed_at": None if closed_at is None else to_rfc3339(closed_at),
                "closed_age": age_text(closed_at, reference_time),
                "entry_price": round(position.entry_price, 6),
                "exit_price": round_or_none(position.exit_price, 6),
                "realized_pnl": round_or_none(position.realized_pnl, 6),
                "realized_return": round_or_none(position.realized_return, 6),
                "exit_reason": position.exit_reason,
                "entry_regime_label": position.entry_regime_label,
                "exit_regime_label": position.exit_regime_label,
                "entry_model_name": position.entry_model_name,
                "exit_model_name": position.exit_model_name,
                "entry_decision_trace_id": position.entry_decision_trace_id,
                "exit_decision_trace_id": position.exit_decision_trace_id,
            }
        )
    return rows


def build_recent_ledger_rows(
    entries: tuple[LedgerEntrySnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build the recent-ledger table for the trading tab."""
    reference_time = utc_now() if now is None else now
    return [
        {
            "ledger_id": entry.ledger_id,
            "symbol": entry.symbol,
            "action": entry.action,
            "reason": entry.reason,
            "fill_time": to_rfc3339(entry.fill_time),
            "fill_age": age_text(entry.fill_time, reference_time),
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
            "decision_trace_id": entry.decision_trace_id,
        }
        for entry in entries
    ]


def build_recent_order_audit_rows(
    entries: tuple[OrderAuditSnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build the recent order-audit table for the trading tab."""
    reference_time = utc_now() if now is None else now
    return [
        {
            "event_id": entry.event_id,
            "order_request_id": entry.order_request_id,
            "decision_trace_id": entry.decision_trace_id,
            "symbol": entry.symbol,
            "action": entry.action,
            "lifecycle_state": entry.lifecycle_state,
            "event_time": to_rfc3339(entry.event_time),
            "event_age": age_text(entry.event_time, reference_time),
            "reason_code": entry.reason_code,
            "details": entry.details,
            "broker_name": entry.broker_name,
            "external_status": entry.external_status,
            "environment_name": entry.environment_name,
            "account_id": entry.account_id,
        }
        for entry in entries
    ]


def build_recent_decision_trace_rows(
    traces: tuple[DecisionTraceSnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build the recent decision-trace table for minimal M14 visibility."""
    reference_time = utc_now() if now is None else now
    return [
        {
            "decision_trace_id": trace.decision_trace_id,
            "execution_mode": trace.execution_mode,
            "symbol": trace.symbol,
            "signal": trace.signal,
            "regime_label": trace.regime_label,
            "risk_outcome": trace.risk_outcome,
            "primary_reason_code": trace.primary_reason_code,
            "signal_as_of_time": to_rfc3339(trace.signal_as_of_time),
            "signal_age": age_text(trace.signal_as_of_time, reference_time),
            "model_version": trace.model_version,
            "top_feature_count": trace.top_feature_count,
            "ordered_adjustment_count": trace.ordered_adjustment_count,
            "requested_notional": round_or_none(trace.requested_notional, 6),
            "approved_notional": round_or_none(trace.approved_notional, 6),
            "updated_at": None if trace.updated_at is None else to_rfc3339(trace.updated_at),
            "updated_age": age_text(trace.updated_at, reference_time),
            "json_report_path": trace.json_report_path,
            "markdown_report_path": trace.markdown_report_path,
        }
        for trace in traces
    ]


def build_latest_blocked_trade_rows(
    trace: DecisionTraceSnapshot | None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build the latest blocked-trade rationale summary."""
    if trace is None:
        return []
    reference_time = utc_now() if now is None else now
    return [
        {
            "decision_trace_id": trace.decision_trace_id,
            "symbol": trace.symbol,
            "signal": trace.signal,
            "regime_label": trace.regime_label,
            "risk_outcome": trace.risk_outcome,
            "blocked_stage": trace.blocked_stage,
            "primary_reason_code": trace.primary_reason_code,
            "reason_texts": "; ".join(trace.reason_texts),
            "signal_as_of_time": to_rfc3339(trace.signal_as_of_time),
            "signal_age": age_text(trace.signal_as_of_time, reference_time),
            "updated_at": None if trace.updated_at is None else to_rfc3339(trace.updated_at),
            "updated_age": age_text(trace.updated_at, reference_time),
            "json_report_path": trace.json_report_path,
            "markdown_report_path": trace.markdown_report_path,
        }
    ]


def build_symbol_freshness_rows(
    freshness_rows: tuple[FreshnessSnapshot, ...],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a compact per-symbol freshness table."""
    reference_time = utc_now() if now is None else now
    rows: list[dict[str, Any]] = []
    for row in freshness_rows:
        rows.append(
            {
                "symbol": row.symbol,
                "health_overall_status": row.health_overall_status,
                "freshness_status": row.freshness_status,
                "feature_freshness_status": row.feature_freshness_status,
                "feature_reason_code": row.feature_reason_code,
                "regime_freshness_status": row.regime_freshness_status,
                "regime_reason_code": row.regime_reason_code,
                "row_id": row.row_id,
                "checked_at": to_rfc3339(row.checked_at),
                "checked_age": age_text(row.checked_at, reference_time),
                "as_of_time": (
                    None if row.as_of_time is None else to_rfc3339(row.as_of_time)
                ),
                "as_of_age": age_text(row.as_of_time, reference_time),
                "feature_age_seconds": round_or_none(row.feature_age_seconds, 3),
                "feature_age": age_from_seconds(row.feature_age_seconds),
                "regime_age_seconds": round_or_none(row.regime_age_seconds, 3),
                "regime_age": age_from_seconds(row.regime_age_seconds),
                "detail": row.detail or row.error,
            }
        )
    return rows


def build_reliability_status_rows(
    *,
    api_health,
    system_reliability: SystemReliabilitySnapshot | None = None,
    reliability_states: tuple[ReliabilityStateSnapshot, ...],
    latest_recovery_event: RecoveryEventSnapshot | None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build one compact reliability summary table."""
    reference_time = utc_now() if now is None else now
    if system_reliability is not None and system_reliability.available:
        return [
            {
                "overall_health": system_reliability.health_overall_status,
                "health_reason_codes": ",".join(system_reliability.reason_codes),
                "lag_breach_active": system_reliability.lag_breach_active,
                "checked_at": to_rfc3339(system_reliability.checked_at),
                "checked_age": age_text(system_reliability.checked_at, reference_time),
                "latest_recovery_event_type": (
                    None
                    if system_reliability.latest_recovery_event is None
                    else system_reliability.latest_recovery_event.event_type
                ),
                "latest_recovery_reason_code": (
                    None
                    if system_reliability.latest_recovery_event is None
                    else system_reliability.latest_recovery_event.reason_code
                ),
                "latest_recovery_time": (
                    None
                    if system_reliability.latest_recovery_event is None
                    else to_rfc3339(system_reliability.latest_recovery_event.event_time)
                ),
                "latest_recovery_age": (
                    None
                    if system_reliability.latest_recovery_event is None
                    else age_text(
                        system_reliability.latest_recovery_event.event_time,
                        reference_time,
                    )
                ),
            }
        ]

    if reliability_states:
        primary_state = reliability_states[0]
        breaker_state = primary_state.breaker_state
        breaker_reason = primary_state.reason_code
    else:
        breaker_state = None
        breaker_reason = None

    return [
        {
            "overall_health": api_health.health_overall_status or api_health.status,
            "health_reason_codes": api_health.reason_code,
            "lag_breach_active": None,
            "checked_at": to_rfc3339(api_health.checked_at),
            "checked_age": age_text(api_health.checked_at, reference_time),
            "breaker_state": breaker_state,
            "breaker_reason_code": breaker_reason,
            "latest_recovery_event_type": (
                None if latest_recovery_event is None else latest_recovery_event.event_type
            ),
            "latest_recovery_reason_code": (
                None if latest_recovery_event is None else latest_recovery_event.reason_code
            ),
            "latest_recovery_time": (
                None
                if latest_recovery_event is None
                else to_rfc3339(latest_recovery_event.event_time)
            ),
            "latest_recovery_age": (
                None
                if latest_recovery_event is None
                else age_text(latest_recovery_event.event_time, reference_time)
            ),
        }
    ]


def build_service_health_rows(
    system_reliability: SystemReliabilitySnapshot | None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build per-service heartbeat rows from the canonical reliability summary."""
    reference_time = utc_now() if now is None else now
    if system_reliability is None or not system_reliability.available:
        return [
            {
                "service_name": None,
                "component_name": "system",
                "health_overall_status": "UNAVAILABLE",
                "heartbeat_freshness_status": None,
                "heartbeat_at": None,
                "heartbeat_age": None,
                "heartbeat_age_seconds": None,
                "checked_at": None,
                "checked_age": None,
                "reason_code": None if system_reliability is None else system_reliability.error,
                "feed_freshness_status": None,
                "feed_age": None,
                "feed_age_seconds": None,
                "detail": None if system_reliability is None else system_reliability.error,
            }
        ]

    return [
        {
            "service_name": service.service_name,
            "component_name": service.component_name,
            "health_overall_status": service.health_overall_status,
            "heartbeat_freshness_status": service.heartbeat_freshness_status,
            "heartbeat_at": (
                None
                if service.heartbeat_at is None
                else to_rfc3339(service.heartbeat_at)
            ),
            "heartbeat_age": age_text(service.heartbeat_at, reference_time),
            "heartbeat_age_seconds": round_or_none(service.heartbeat_age_seconds, 3),
            "checked_at": to_rfc3339(service.checked_at),
            "checked_age": age_text(service.checked_at, reference_time),
            "reason_code": service.reason_code,
            "feed_freshness_status": service.feed_freshness_status,
            "feed_age": age_from_seconds(service.feed_age_seconds),
            "feed_age_seconds": round_or_none(service.feed_age_seconds, 3),
            "detail": service.detail,
        }
        for service in system_reliability.services
    ]


def build_feature_lag_rows(
    system_reliability: SystemReliabilitySnapshot | None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build per-symbol feature lag rows from the canonical reliability summary."""
    if system_reliability is None or not system_reliability.available:
        return []
    reference_time = utc_now() if now is None else now
    return [
        _feature_lag_row(snapshot, reference_time=reference_time)
        for snapshot in system_reliability.lag_by_symbol
    ]


def _feature_lag_row(
    snapshot: FeatureLagSummarySnapshot,
    *,
    reference_time: datetime,
) -> dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "lag_breach": snapshot.lag_breach,
        "health_overall_status": snapshot.health_overall_status,
        "reason_code": snapshot.reason_code,
        "evaluated_at": to_rfc3339(snapshot.evaluated_at),
        "evaluation_age": age_text(snapshot.evaluated_at, reference_time),
        "time_lag_seconds": round_or_none(snapshot.time_lag_seconds, 3),
        "time_lag_reason_code": snapshot.time_lag_reason_code,
        "processing_lag_seconds": round_or_none(snapshot.processing_lag_seconds, 3),
        "processing_lag_reason_code": snapshot.processing_lag_reason_code,
        "latest_raw_event_received_at": (
            None
            if snapshot.latest_raw_event_received_at is None
            else to_rfc3339(snapshot.latest_raw_event_received_at)
        ),
        "latest_raw_event_age": age_text(
            snapshot.latest_raw_event_received_at,
            reference_time,
        ),
        "latest_feature_as_of_time": (
            None
            if snapshot.latest_feature_as_of_time is None
            else to_rfc3339(snapshot.latest_feature_as_of_time)
        ),
        "latest_feature_age": age_text(
            snapshot.latest_feature_as_of_time,
            reference_time,
        ),
        "detail": snapshot.detail,
    }


def build_live_status_rows(
    *,
    trading_config: PaperTradingConfig,
    live_safety_state: LiveSafetySnapshot | None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a compact guarded-live status table for the dashboard."""
    reference_time = utc_now() if now is None else now
    if live_safety_state is None:
        return [
            {
                "execution_mode": trading_config.execution.mode,
                "broker_name": "alpaca",
                "live_enabled": False,
                "account_validated": False,
                "startup_checks_passed": False,
                "startup_checks_passed_at": None,
                "startup_checks_age": None,
                "manual_disable_active": None,
                "failure_hard_stop_active": None,
                "consecutive_live_failures": None,
                "expected_environment": trading_config.execution.live.expected_environment,
                "validated_environment": None,
                "expected_account_id": trading_config.execution.live.expected_account_id,
                "validated_account_id": None,
                "symbol_whitelist": ",".join(trading_config.execution.live.symbol_whitelist),
                "live_max_order_notional": trading_config.execution.live.max_order_notional,
                "system_health_status": None,
                "system_health_reason_code": None,
                "health_gate_status": None,
                "health_gate_reason_code": None,
                "health_gate_detail": None,
                "reconciliation_status": None,
                "reconciliation_reason_code": None,
                "broker_cash": None,
                "broker_equity": None,
                "unresolved_incident_count": None,
                "last_failure_reason": None,
                "updated_at": None,
                "updated_age": None,
            }
        ]

    return [
        {
            "execution_mode": live_safety_state.execution_mode,
            "broker_name": live_safety_state.broker_name,
            "live_enabled": live_safety_state.live_enabled,
            "account_validated": live_safety_state.account_validated,
            "startup_checks_passed": live_safety_state.startup_checks_passed,
            "startup_checks_passed_at": (
                None
                if live_safety_state.startup_checks_passed_at is None
                else to_rfc3339(live_safety_state.startup_checks_passed_at)
            ),
            "startup_checks_age": age_text(
                live_safety_state.startup_checks_passed_at,
                reference_time,
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
            "system_health_status": live_safety_state.system_health_status,
            "system_health_reason_code": live_safety_state.system_health_reason_code,
            "health_gate_status": live_safety_state.health_gate_status,
            "health_gate_reason_code": live_safety_state.health_gate_reason_code,
            "health_gate_detail": live_safety_state.health_gate_detail,
            "reconciliation_status": live_safety_state.reconciliation_status,
            "reconciliation_reason_code": live_safety_state.reconciliation_reason_code,
            "broker_cash": round_or_none(live_safety_state.broker_cash, 2),
            "broker_equity": round_or_none(live_safety_state.broker_equity, 2),
            "unresolved_incident_count": live_safety_state.unresolved_incident_count,
            "last_failure_reason": live_safety_state.last_failure_reason,
            "updated_at": to_rfc3339(live_safety_state.updated_at),
            "updated_age": age_text(live_safety_state.updated_at, reference_time),
        }
    ]


def build_trade_journal_rows(
    traces: tuple[DecisionTraceSnapshot, ...],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build an operator-facing trade journal from canonical decision traces."""
    reference_time = utc_now() if now is None else now
    rows: list[dict[str, Any]] = []
    for trace in sorted(
        traces,
        key=lambda row: (row.signal_as_of_time, row.decision_trace_id),
        reverse=True,
    ):
        reason_code = _journal_reason_code(trace)
        rows.append(
            {
                "decision_trace_id": trace.decision_trace_id,
                "signal_as_of_time": to_rfc3339(trace.signal_as_of_time),
                "signal_age": age_text(trace.signal_as_of_time, reference_time),
                "symbol": trace.symbol,
                "execution_mode": trace.execution_mode,
                "signal": trace.signal,
                "regime_label": trace.regime_label,
                "outcome_category": _journal_outcome_category(trace),
                "reason_code": reason_code,
                "reason_texts": "; ".join(trace.reason_texts),
                "blocked": trace.risk_outcome == "BLOCKED",
                "blocked_stage": trace.blocked_stage,
                "requested_notional": round_or_none(trace.requested_notional, 6),
                "approved_notional": round_or_none(trace.approved_notional, 6),
                "ordered_adjustment_count": trace.ordered_adjustment_count,
                "model_version": trace.model_version,
                "signal_row_id": trace.signal_row_id,
                "json_report_path": trace.json_report_path,
                "markdown_report_path": trace.markdown_report_path,
            }
        )
    return rows


def filter_trade_journal_traces(
    traces: tuple[DecisionTraceSnapshot, ...],
    *,
    symbol: str | None = None,
    mode: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    regime: str | None = None,
    reason_code: str | None = None,
    outcome_category: str | None = None,
    only_blocked: bool = False,
) -> tuple[DecisionTraceSnapshot, ...]:
    """Filter recent decision traces for the operator trade journal."""
    filtered: list[DecisionTraceSnapshot] = []
    for trace in traces:
        if symbol not in {None, "All"} and trace.symbol != symbol:
            continue
        if mode not in {None, "All"} and trace.execution_mode != mode:
            continue
        if start_time is not None and trace.signal_as_of_time < start_time:
            continue
        if end_time is not None and trace.signal_as_of_time > end_time:
            continue
        if regime not in {None, "All"} and (trace.regime_label or "UNKNOWN") != regime:
            continue
        resolved_reason_code = _journal_reason_code(trace)
        if reason_code not in {None, "All"} and resolved_reason_code != reason_code:
            continue
        resolved_outcome = _journal_outcome_category(trace)
        if outcome_category not in {None, "All"} and resolved_outcome != outcome_category:
            continue
        if only_blocked and trace.risk_outcome != "BLOCKED":
            continue
        filtered.append(trace)
    return tuple(filtered)


def build_operator_incident_rows(
    *,
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Aggregate current operator-visible unsafe state into one incident list."""
    reference_time = utc_now() if now is None else now
    incidents: list[dict[str, Any]] = []
    live_state = snapshot.database.live_safety_state
    system_reliability = snapshot.system_reliability

    if trading_config.execution.mode == "live":
        if live_state is None:
            _append_incident(
                incidents,
                severity="CRITICAL",
                category="live_safety",
                scope="execution",
                reason_code="LIVE_SAFETY_STATE_MISSING",
                detail="Live mode is configured but no execution_live_safety_state row was found.",
                updated_at=snapshot.database.checked_at,
                reference_time=reference_time,
            )
        else:
            if not live_state.live_enabled:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="live_safety",
                    scope="execution",
                    reason_code="LIVE_DISABLED",
                    detail="Live execution is configured but live_enabled is false.",
                    updated_at=live_state.updated_at,
                    reference_time=reference_time,
                )
            if not live_state.startup_checks_passed:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="live_safety",
                    scope="startup",
                    reason_code="STARTUP_CHECKS_NOT_PASSED",
                    detail="Startup reconciliation or safety checks have not passed.",
                    updated_at=live_state.updated_at,
                    reference_time=reference_time,
                )
            if live_state.manual_disable_active:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="live_safety",
                    scope="execution",
                    reason_code="MANUAL_DISABLE_ACTIVE",
                    detail="Manual disable is active for guarded live trading.",
                    updated_at=live_state.updated_at,
                    reference_time=reference_time,
                )
            if live_state.failure_hard_stop_active:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="live_safety",
                    scope="execution",
                    reason_code="FAILURE_HARD_STOP_ACTIVE",
                    detail=live_state.last_failure_reason or "Live failure hard stop is active.",
                    updated_at=live_state.updated_at,
                    reference_time=reference_time,
                )
            if live_state.health_gate_status not in {None, "CLEAR"}:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="health_gate",
                    scope="execution",
                    reason_code=(
                        live_state.health_gate_reason_code or "HEALTH_GATE_BLOCKED"
                    ),
                    detail=(
                        live_state.health_gate_detail
                        or "Live submit is blocked by canonical health gating."
                    ),
                    updated_at=(
                        live_state.system_health_checked_at or live_state.updated_at
                    ),
                    reference_time=reference_time,
                )
            if live_state.reconciliation_status not in {None, "CLEAR"}:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="reconciliation",
                    scope="execution",
                    reason_code=(
                        live_state.reconciliation_reason_code
                        or "RECONCILIATION_NOT_CLEAR"
                    ),
                    detail="Broker reconciliation is not clear for live submit.",
                    updated_at=(
                        live_state.reconciliation_checked_at or live_state.updated_at
                    ),
                    reference_time=reference_time,
                )
            if live_state.unresolved_incident_count > 0:
                _append_incident(
                    incidents,
                    severity="CRITICAL",
                    category="reconciliation",
                    scope="execution",
                    reason_code="UNRESOLVED_LIVE_INCIDENTS",
                    detail=(
                        f"{live_state.unresolved_incident_count} unresolved live "
                        "reconciliation incidents remain."
                    ),
                    updated_at=(
                        live_state.reconciliation_checked_at or live_state.updated_at
                    ),
                    reference_time=reference_time,
                )

    if system_reliability is None or not system_reliability.available:
        _append_incident(
            incidents,
            severity="CRITICAL" if trading_config.execution.mode == "live" else "HIGH",
            category="health",
            scope="system",
            reason_code="SYSTEM_HEALTH_UNAVAILABLE",
            detail=(
                "Canonical reliability summary is unavailable."
                if system_reliability is None
                else system_reliability.error or "Canonical reliability summary is unavailable."
            ),
            updated_at=(
                snapshot.database.checked_at
                if system_reliability is None
                else system_reliability.checked_at
            ),
            reference_time=reference_time,
        )
    else:
        if system_reliability.health_overall_status == "UNAVAILABLE":
            _append_incident(
                incidents,
                severity="CRITICAL",
                category="health",
                scope="system",
                reason_code=_join_codes(system_reliability.reason_codes),
                detail="Canonical system health is unavailable.",
                updated_at=system_reliability.checked_at,
                reference_time=reference_time,
            )
        elif system_reliability.health_overall_status == "DEGRADED":
            _append_incident(
                incidents,
                severity="HIGH",
                category="health",
                scope="system",
                reason_code=_join_codes(system_reliability.reason_codes),
                detail="Canonical system health is degraded.",
                updated_at=system_reliability.checked_at,
                reference_time=reference_time,
            )
        if system_reliability.lag_breach_active:
            _append_incident(
                incidents,
                severity="HIGH",
                category="freshness",
                scope="features",
                reason_code="FEATURE_LAG_BREACH",
                detail=(
                    "At least one configured symbol is breaching feature "
                    "consumer lag thresholds."
                ),
                updated_at=system_reliability.checked_at,
                reference_time=reference_time,
            )

    for row in snapshot.freshness:
        if not row.available:
            _append_incident(
                incidents,
                severity="HIGH",
                category="freshness",
                scope=row.symbol,
                reason_code="FRESHNESS_UNAVAILABLE",
                detail=row.error or "Freshness endpoint returned no data.",
                updated_at=row.checked_at,
                reference_time=reference_time,
            )
            continue
        if row.freshness_status != "FRESH":
            _append_incident(
                incidents,
                severity="HIGH",
                category="freshness",
                scope=row.symbol,
                reason_code=row.reason_code or "SIGNAL_STALE",
                detail=row.detail or "Signal freshness is not fresh.",
                updated_at=row.checked_at,
                reference_time=reference_time,
            )
        if row.feature_freshness_status != "FRESH":
            _append_incident(
                incidents,
                severity="HIGH",
                category="freshness",
                scope=row.symbol,
                reason_code=row.feature_reason_code or "FEATURE_STALE",
                detail=row.detail or "Feature freshness is not fresh.",
                updated_at=row.checked_at,
                reference_time=reference_time,
            )
        if row.regime_freshness_status != "FRESH":
            _append_incident(
                incidents,
                severity="HIGH",
                category="freshness",
                scope=row.symbol,
                reason_code=row.regime_reason_code or "REGIME_STALE",
                detail=row.detail or "Regime freshness is not fresh.",
                updated_at=row.checked_at,
                reference_time=reference_time,
            )

    for state in snapshot.database.reliability_states:
        if state.breaker_state == "OPEN":
            _append_incident(
                incidents,
                severity="CRITICAL",
                category="breaker",
                scope=state.component_name,
                reason_code=state.reason_code or "BREAKER_OPEN",
                detail=state.detail or "Signal client breaker is OPEN.",
                updated_at=state.updated_at,
                reference_time=reference_time,
            )
        elif state.breaker_state not in {"CLOSED", "NONE"}:
            _append_incident(
                incidents,
                severity="HIGH",
                category="breaker",
                scope=state.component_name,
                reason_code=state.reason_code or f"BREAKER_{state.breaker_state}",
                detail=state.detail or f"Breaker is {state.breaker_state}.",
                updated_at=state.updated_at,
                reference_time=reference_time,
            )

    blocked_trade = snapshot.database.latest_blocked_trade
    if blocked_trade is not None:
        _append_incident(
            incidents,
            severity="MEDIUM",
            category="risk_block",
            scope=blocked_trade.symbol,
            reason_code=blocked_trade.primary_reason_code or "TRADE_BLOCKED",
            detail="; ".join(blocked_trade.reason_texts) or "Latest trade was blocked by risk.",
            updated_at=blocked_trade.updated_at or blocked_trade.signal_as_of_time,
            reference_time=reference_time,
        )

    incidents.sort(
        key=lambda row: (
            _SEVERITY_ORDER[row["severity"]],
            _CATEGORY_ORDER.get(row["category"], 99),
            row["_sort_timestamp"],
        )
    )
    for row in incidents:
        row.pop("_sort_timestamp", None)
    return incidents


def build_operator_banner(
    *,
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
    incidents: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the persistent top-banner payload for the operator console."""
    reference_time = utc_now() if now is None else now
    latest_evaluation_time = _latest_evaluation_time(snapshot)
    primary_incident = incidents[0] if incidents else None
    posture = "safe"
    if primary_incident is not None and primary_incident["severity"] == "CRITICAL":
        posture = "blocked"
    elif trading_config.execution.mode == "live" and _is_live_armed(snapshot):
        posture = "armed-live"
    elif incidents:
        posture = "degraded"

    execution_venue = _execution_venue(snapshot, trading_config)
    market_data_venue = trading_config.source_exchange
    venue = execution_venue
    if market_data_venue.lower() != execution_venue.lower():
        venue = f"{market_data_venue} -> {execution_venue}"

    return {
        "mode": trading_config.execution.mode,
        "safety_posture": posture,
        "severity": _POSTURE_SEVERITY[posture],
        "venue": venue,
        "environment": _execution_environment(snapshot, trading_config),
        "latest_evaluation_time": (
            None if latest_evaluation_time is None else to_rfc3339(latest_evaluation_time)
        ),
        "latest_evaluation_age": age_text(latest_evaluation_time, reference_time),
        "primary_reason_code": (
            None if primary_incident is None else primary_incident["reason_code"]
        ),
        "primary_reason_detail": (
            None if primary_incident is None else primary_incident["detail"]
        ),
    }


def build_venue_environment_rows(
    *,
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
) -> list[dict[str, Any]]:
    """Build a compact venue and environment truth summary."""
    live_state = snapshot.database.live_safety_state
    execution_venue = _execution_venue(snapshot, trading_config)
    market_data_venue = trading_config.source_exchange
    environment = _execution_environment(snapshot, trading_config)
    validated_account_id = None if live_state is None else live_state.account_id
    venue_mismatch = market_data_venue.lower() != execution_venue.lower()
    return [
        {
            "market_data_venue": market_data_venue,
            "execution_venue": execution_venue,
            "environment": environment,
            "validated_account_id": validated_account_id,
            "expected_account_id": trading_config.execution.live.expected_account_id,
            "venue_mismatch": venue_mismatch,
            "portfolio_truth_source": {
                "paper": "LOCAL_SIMULATION",
                "shadow": "SHADOW_AUDIT_ONLY",
                "live": "BROKER_RECONCILIATION_ONLY",
            }[trading_config.execution.mode],
        }
    ]


def build_config_summary_rows(
    *,
    settings: Settings,
    trading_config: PaperTradingConfig,
    snapshot: DashboardSnapshot,
) -> list[dict[str, Any]]:
    """Build an operator-friendly config summary without secrets."""
    live_state = snapshot.database.live_safety_state
    latest_trace = (
        snapshot.database.recent_decision_traces[0]
        if snapshot.database.recent_decision_traces
        else None
    )
    return [
        {"setting": "Symbols", "value": ", ".join(trading_config.symbols)},
        {
            "setting": "Canonical feature table",
            "value": settings.tables.feature_ohlc,
        },
        {
            "setting": "Inference API URL",
            "value": settings.dashboard.inference_api_base_url,
        },
        {
            "setting": "Execution mode",
            "value": trading_config.execution.mode,
        },
        {
            "setting": "Market data venue",
            "value": trading_config.source_exchange,
        },
        {
            "setting": "Execution venue / environment",
            "value": (
                f"{_execution_venue(snapshot, trading_config)} / "
                f"{_execution_environment(snapshot, trading_config)}"
            ),
        },
        {
            "setting": "Validated account",
            "value": (
                "-"
                if live_state is None or live_state.account_id is None
                else live_state.account_id
            ),
        },
        {
            "setting": "Fee assumptions",
            "value": (
                f"fees {trading_config.risk.fee_bps:.1f} bps, "
                f"slippage {trading_config.risk.slippage_bps:.1f} bps"
            ),
        },
        {
            "setting": "Initial cash",
            "value": f"{trading_config.risk.initial_cash:,.2f}",
        },
        {
            "setting": "Major risk caps",
            "value": (
                f"max open {trading_config.risk.max_open_positions}, "
                f"per asset {trading_config.risk.max_exposure_per_asset:.0%}, "
                f"total {trading_config.risk.max_total_exposure:.0%}, "
                f"daily loss {trading_config.risk.max_daily_loss_amount:,.2f}, "
                f"drawdown {trading_config.risk.max_drawdown_pct:.0%}"
            ),
        },
        {
            "setting": "Live whitelist",
            "value": ", ".join(trading_config.execution.live.symbol_whitelist) or "-",
        },
        {
            "setting": "Active model reference",
            "value": (
                latest_trace.model_version
                if latest_trace is not None
                else snapshot.api_health.model_name or "-"
            ),
        },
        {
            "setting": "Active model artifact",
            "value": snapshot.api_health.model_artifact_path or "-",
        },
        {
            "setting": "Active regime reference",
            "value": snapshot.api_health.regime_run_id or "-",
        },
        {
            "setting": "Active regime artifact",
            "value": snapshot.api_health.regime_artifact_path or "-",
        },
    ]


def build_model_reference_rows(
    *,
    snapshot: DashboardSnapshot,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build operator-facing active model and regime reference rows."""
    reference_time = utc_now() if now is None else now
    traces = snapshot.database.recent_decision_traces
    return [
        {
            "model_name": snapshot.api_health.model_name,
            "model_loaded": snapshot.api_health.model_loaded,
            "latest_model_version": (
                None if not traces else traces[0].model_version
            ),
            "regime_loaded": snapshot.api_health.regime_loaded,
            "regime_run_id": snapshot.api_health.regime_run_id,
            "model_artifact_path": snapshot.api_health.model_artifact_path,
            "regime_artifact_path": snapshot.api_health.regime_artifact_path,
            "health_checked_at": to_rfc3339(snapshot.api_health.checked_at),
            "health_checked_age": age_text(snapshot.api_health.checked_at, reference_time),
        }
    ]


def build_latest_recovery_rows(
    recovery_event: RecoveryEventSnapshot | None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build a one-row latest recovery event table."""
    if recovery_event is None:
        return []
    reference_time = utc_now() if now is None else now
    return [
        {
            "service_name": recovery_event.service_name,
            "component_name": recovery_event.component_name,
            "event_type": recovery_event.event_type,
            "reason_code": recovery_event.reason_code,
            "event_time": to_rfc3339(recovery_event.event_time),
            "event_age": age_text(recovery_event.event_time, reference_time),
            "health_overall_status": recovery_event.health_overall_status,
            "freshness_status": recovery_event.freshness_status,
            "breaker_state": recovery_event.breaker_state,
            "detail": recovery_event.detail,
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


_SEVERITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "INFO": 3,
}

_CATEGORY_ORDER = {
    "live_safety": 0,
    "health_gate": 1,
    "reconciliation": 2,
    "health": 3,
    "breaker": 4,
    "freshness": 5,
    "risk_block": 6,
}

_POSTURE_SEVERITY = {
    "safe": "success",
    "degraded": "warning",
    "blocked": "error",
    "armed-live": "error",
}


def age_from_seconds(value: float | None) -> str | None:
    """Render a compact age string from a second count."""
    if value is None:
        return None
    reference_time = utc_now()
    synthetic_timestamp = reference_time - timedelta(seconds=float(value))
    return age_text(synthetic_timestamp, reference_time)


def _journal_reason_code(trace: DecisionTraceSnapshot) -> str | None:
    if trace.primary_reason_code is not None:
        return trace.primary_reason_code
    return trace.signal_reason_code


def _journal_outcome_category(trace: DecisionTraceSnapshot) -> str:
    if trace.risk_outcome is not None:
        return trace.risk_outcome
    return "SIGNAL_ONLY"


def _append_incident(
    incidents: list[dict[str, Any]],
    *,
    severity: str,
    category: str,
    scope: str,
    reason_code: str,
    detail: str,
    updated_at: datetime | None,
    reference_time: datetime,
) -> None:
    incidents.append(
        {
            "severity": severity,
            "category": category,
            "scope": scope,
            "reason_code": reason_code,
            "detail": detail,
            "updated_at": None if updated_at is None else to_rfc3339(updated_at),
            "updated_age": age_text(updated_at, reference_time),
            "_sort_timestamp": (
                0.0
                if updated_at is None
                else -updated_at.timestamp()
            ),
        }
    )


def _latest_evaluation_time(snapshot: DashboardSnapshot) -> datetime | None:
    candidates: list[datetime] = [
        snapshot.api_health.checked_at,
        snapshot.database.checked_at,
    ]
    if snapshot.system_reliability is not None:
        candidates.append(snapshot.system_reliability.checked_at)
    candidates.extend(
        signal.as_of_time
        for signal in snapshot.signals
        if signal.as_of_time is not None
    )
    feature_as_of = latest_feature_as_of(snapshot.database.latest_features)
    if feature_as_of is not None:
        candidates.append(feature_as_of)
    return max(candidates) if candidates else None


def _execution_venue(
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
) -> str:
    if trading_config.execution.mode == "live":
        live_state = snapshot.database.live_safety_state
        return live_state.broker_name if live_state is not None else "alpaca"
    if trading_config.execution.mode == "shadow":
        return "shadow-audit-only"
    return "local-paper-simulation"


def _execution_environment(
    snapshot: DashboardSnapshot,
    trading_config: PaperTradingConfig,
) -> str:
    if trading_config.execution.mode == "live":
        live_state = snapshot.database.live_safety_state
        if live_state is not None and live_state.environment_name is not None:
            return live_state.environment_name
        return trading_config.execution.live.expected_environment
    return trading_config.execution.mode


def _is_live_armed(snapshot: DashboardSnapshot) -> bool:
    live_state = snapshot.database.live_safety_state
    if live_state is None:
        return False
    return (
        live_state.live_enabled
        and live_state.startup_checks_passed
        and not live_state.manual_disable_active
        and not live_state.failure_hard_stop_active
        and live_state.health_gate_status == "CLEAR"
        and live_state.reconciliation_status == "CLEAR"
        and live_state.unresolved_incident_count == 0
    )


def _join_codes(reason_codes: tuple[str, ...]) -> str:
    if not reason_codes:
        return "UNKNOWN"
    return ",".join(reason_codes)


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
