"""Focused semantic M18 metric tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

from app.evaluation.metrics import (
    compute_cost_aware_precision_by_mode,
    compute_performance_rows,
    compute_slippage_distribution,
)
from app.trading.schemas import PaperPosition, TradeLedgerEntry

from tests.test_evaluation_matching import _opportunity
from tests.test_evaluation_normalize import _ts


def _closed_buy_opportunity(
    *,
    mode: str,
    trace_id: int,
    symbol: str,
    regime_label: str,
    realized_pnl: float,
) -> tuple:
    row = _opportunity(
        mode=mode,
        trace_id=trace_id,
        row_id=f"{symbol}|{trace_id}",
        signal_action="BUY",
        signal_time=_ts(12, 5),
    )
    row = replace(
        row,
        symbol=symbol,
        regime_label=regime_label,
        position=replace(
            row.position,
            position_id=trace_id,
            position_status="CLOSED",
            opened_at=_ts(12, 10),
            closed_at=_ts(13, 0),
            realized_pnl=realized_pnl,
            realized_return=realized_pnl / 1000.0,
            entry_decision_trace_id=trace_id,
        ),
    )
    position = PaperPosition(
        service_name="paper-trader",
        execution_mode=mode,
        symbol=symbol,
        status="CLOSED",
        entry_signal_interval_begin=_ts(12, 0),
        entry_signal_as_of_time=_ts(12, 5),
        entry_signal_row_id=f"{symbol}|{trace_id}",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.70,
        entry_confidence=0.70,
        entry_fill_interval_begin=_ts(12, 10),
        entry_fill_time=_ts(12, 10),
        entry_price=100.0,
        quantity=10.0,
        entry_notional=1000.0,
        entry_fee=1.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        entry_order_request_id=trace_id,
        entry_decision_trace_id=trace_id,
        entry_regime_label=regime_label,
        position_id=trace_id,
        realized_pnl=realized_pnl,
        realized_return=realized_pnl / 1000.0,
    )
    return row, position


def test_cost_aware_precision_uses_actioned_buy_outcomes_after_costs() -> None:
    positive_row, positive_position = _closed_buy_opportunity(
        mode="paper",
        trace_id=1,
        symbol="BTC/USD",
        regime_label="TREND_UP",
        realized_pnl=25.0,
    )
    negative_row, negative_position = _closed_buy_opportunity(
        mode="paper",
        trace_id=2,
        symbol="BTC/USD",
        regime_label="TREND_UP",
        realized_pnl=-5.0,
    )
    open_row = _opportunity(
        mode="paper",
        trace_id=3,
        row_id="BTC/USD|3",
        signal_action="BUY",
        signal_time=_ts(12, 5),
    )
    hold_row = _opportunity(
        mode="paper",
        trace_id=4,
        row_id="BTC/USD|4",
        signal_action="HOLD",
        signal_time=_ts(12, 5),
    )

    summaries = compute_cost_aware_precision_by_mode(
        opportunities=[positive_row, negative_row, open_row, hold_row],
        execution_modes=("paper",),
    )

    assert summaries["paper"].comparable_buy_count == 2
    assert summaries["paper"].positive_after_cost_buy_count == 1
    assert summaries["paper"].cost_aware_precision == 0.5

    asset_rows, regime_rows = compute_performance_rows(
        opportunities=[positive_row, negative_row, open_row, hold_row],
        positions=[positive_position, negative_position],
        in_window_trace_ids_by_mode={"paper": {1, 2}},
    )

    assert asset_rows[0].cost_aware_precision == 0.5
    assert asset_rows[0].comparable_buy_count == 2
    assert asset_rows[0].positive_after_cost_buy_count == 1
    assert regime_rows[0].cost_aware_precision == 0.5


def test_mode_level_precision_is_not_an_average_of_asset_rows() -> None:
    btc_row, btc_position = _closed_buy_opportunity(
        mode="paper",
        trace_id=11,
        symbol="BTC/USD",
        regime_label="TREND_UP",
        realized_pnl=20.0,
    )
    eth_rows_positions = [
        _closed_buy_opportunity(
            mode="paper",
            trace_id=12 + index,
            symbol="ETH/USD",
            regime_label="TREND_UP",
            realized_pnl=-10.0,
        )
        for index in range(3)
    ]
    opportunities = [btc_row]
    positions = [btc_position]
    for row, position in eth_rows_positions:
        opportunities.append(row)
        positions.append(position)

    summaries = compute_cost_aware_precision_by_mode(
        opportunities=opportunities,
        execution_modes=("paper",),
    )
    asset_rows, _ = compute_performance_rows(
        opportunities=opportunities,
        positions=positions,
        in_window_trace_ids_by_mode={"paper": {11, 12, 13, 14}},
    )

    precision_by_asset = {row.key: row.cost_aware_precision for row in asset_rows}
    assert precision_by_asset == {"BTC/USD": 1.0, "ETH/USD": 0.0}
    assert summaries["paper"].cost_aware_precision == 0.25


def test_missing_shadow_fill_truth_remains_not_applicable() -> None:
    ledger_entries = [
        TradeLedgerEntry(
            service_name="paper-trader",
            execution_mode="paper",
            symbol="BTC/USD",
            action="BUY",
            reason="entry",
            fill_interval_begin=_ts(12, 10),
            fill_time=_ts(12, 10),
            fill_price=100.0,
            quantity=10.0,
            notional=1000.0,
            fee=1.0,
            slippage_bps=5.0,
            cash_flow=-1001.0,
        )
    ]

    rows = compute_slippage_distribution(ledger_entries)
    by_mode = {row.execution_mode: row for row in rows}

    assert by_mode["shadow"].truth_status == "NOT_APPLICABLE"
    assert by_mode["shadow"].count == 0
