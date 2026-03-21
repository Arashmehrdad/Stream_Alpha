"""Metrics tests for Stream Alpha M5."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime, timezone

from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.metrics import build_summary
from app.trading.schemas import PaperPosition


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD"),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir="artifacts/paper_trading",
        risk=RiskConfig(
            initial_cash=10_000.0,
            position_fraction=0.25,
            fee_bps=20.0,
            slippage_bps=5.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cooldown_candles=1,
            max_open_positions=2,
            max_exposure_per_asset=0.25,
        ),
    )


def _closed_position(
    symbol: str,
    pnl: float,
    trade_return: float,
    *,
    entry_regime_label: str,
) -> PaperPosition:
    return PaperPosition(
        service_name="paper-trader",
        symbol=symbol,
        status="CLOSED",
        entry_signal_interval_begin=datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc),
        entry_signal_as_of_time=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_signal_row_id=f"{symbol}|2026-03-20T09:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.7,
        entry_confidence=0.7,
        entry_fill_interval_begin=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_fill_time=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_price=100.0,
        quantity=1.0,
        entry_notional=100.0,
        entry_fee=0.2,
        stop_loss_price=98.0,
        take_profit_price=104.0,
        entry_regime_label=entry_regime_label,
        position_id=1,
        exit_reason="SELL_SIGNAL",
        exit_fill_interval_begin=datetime(2026, 3, 20, 9, 10, tzinfo=timezone.utc),
        exit_fill_time=datetime(2026, 3, 20, 9, 10, tzinfo=timezone.utc),
        exit_price=100.0 + pnl,
        exit_notional=100.0 + pnl + 0.2,
        exit_fee=0.2,
        realized_pnl=pnl,
        realized_return=trade_return,
    )


def _open_position(symbol: str, *, entry_regime_label: str) -> PaperPosition:
    return PaperPosition(
        service_name="paper-trader",
        symbol=symbol,
        status="OPEN",
        entry_signal_interval_begin=datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc),
        entry_signal_as_of_time=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_signal_row_id=f"{symbol}|2026-03-20T09:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.7,
        entry_confidence=0.7,
        entry_fill_interval_begin=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_fill_time=datetime(2026, 3, 20, 9, 5, tzinfo=timezone.utc),
        entry_price=100.0,
        quantity=2.0,
        entry_notional=200.0,
        entry_fee=0.4,
        stop_loss_price=98.0,
        take_profit_price=104.0,
        entry_regime_label=entry_regime_label,
        position_id=2,
    )


def test_metrics_math_is_deterministic() -> None:
    """Overall paper-trading metrics should be computed deterministically."""
    summary = build_summary(
        config=_config(),
        positions=[
            _closed_position(
                "BTC/USD",
                pnl=5.0,
                trade_return=0.05,
                entry_regime_label="TREND_UP",
            ),
            _closed_position(
                "ETH/USD",
                pnl=-2.0,
                trade_return=-0.02,
                entry_regime_label="HIGH_VOL",
            ),
            _open_position("BTC/USD", entry_regime_label="TREND_UP"),
        ],
        latest_prices={"BTC/USD": 110.0, "ETH/USD": 98.0},
        cash_balance=9_700.0,
    )

    overall = summary["overall"]
    assert overall["execution_mode"] == "paper"
    assert overall["data_source_exchange"] == "kraken"
    assert overall["execution_context"]["execution_contract"] == "LOCAL_PAPER_SIMULATION"
    assert overall["execution_context"]["portfolio_truth_source"] == "LOCAL_SIMULATION"
    assert overall["execution_context"]["cross_venue_context"]["execution_venue"] is None
    assert round(overall["cumulative_pnl_realized"], 6) == 3.0
    assert overall["win_rate"] == 0.5
    assert round(overall["turnover"], 6) > 0.0
    assert "BTC/USD" in overall["hit_rate_by_asset"]
    assert "ETH/USD" in overall["hit_rate_by_asset"]
    assert overall["hit_rate_by_regime"]["TREND_UP"] == 1.0
    assert overall["hit_rate_by_regime"]["HIGH_VOL"] == 0.0
    assert overall["realized_pnl_by_regime"]["TREND_UP"] == 5.0
    assert overall["closed_position_count_by_regime"]["HIGH_VOL"] == 1
    by_regime = {row["regime_label"]: row for row in summary["by_regime"]}
    assert round(by_regime["TREND_UP"]["realized_pnl"], 6) == 5.0
    assert by_regime["TREND_UP"]["open_positions"] == 1
    assert by_regime["HIGH_VOL"]["closed_positions"] == 1
