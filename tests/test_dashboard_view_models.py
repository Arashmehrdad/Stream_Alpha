"""Dashboard view-model tests for Stream Alpha M6."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.schemas import PaperPosition
from dashboards.data_sources import (
    ApiHealthSnapshot,
    DashboardSnapshot,
    DatabaseSnapshot,
    EngineStateSnapshot,
)
from dashboards.view_models import (
    build_drawdown_curve_rows,
    build_equity_curve_rows,
    build_overview_metrics,
    build_trader_freshness,
)


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
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
            max_open_positions=1,
            max_exposure_per_asset=0.25,
        ),
    )


def _closed_position() -> PaperPosition:
    interval_begin = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
    return PaperPosition(
        service_name="paper-trader",
        symbol="BTC/USD",
        status="CLOSED",
        entry_signal_interval_begin=interval_begin,
        entry_signal_as_of_time=interval_begin + timedelta(minutes=5),
        entry_signal_row_id="BTC/USD|2026-03-20T10:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.7,
        entry_confidence=0.7,
        entry_fill_interval_begin=interval_begin + timedelta(minutes=5),
        entry_fill_time=interval_begin + timedelta(minutes=5),
        entry_price=100.0,
        quantity=10.0,
        entry_notional=1000.0,
        entry_fee=2.0,
        stop_loss_price=98.0,
        take_profit_price=104.0,
        position_id=1,
        exit_reason="SELL_SIGNAL",
        exit_signal_interval_begin=interval_begin + timedelta(minutes=10),
        exit_signal_as_of_time=interval_begin + timedelta(minutes=15),
        exit_signal_row_id="BTC/USD|2026-03-20T10:10:00Z",
        exit_model_name="logistic_regression",
        exit_prob_up=0.3,
        exit_confidence=0.7,
        exit_fill_interval_begin=interval_begin + timedelta(minutes=15),
        exit_fill_time=interval_begin + timedelta(minutes=15),
        exit_price=105.0,
        exit_notional=1050.0,
        exit_fee=2.1,
        realized_pnl=45.9,
        realized_return=0.0459,
        opened_at=interval_begin + timedelta(minutes=5),
        closed_at=interval_begin + timedelta(minutes=15),
        updated_at=interval_begin + timedelta(minutes=15),
    )


def _open_position() -> PaperPosition:
    interval_begin = datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc)
    return PaperPosition(
        service_name="paper-trader",
        symbol="BTC/USD",
        status="OPEN",
        entry_signal_interval_begin=interval_begin,
        entry_signal_as_of_time=interval_begin + timedelta(minutes=5),
        entry_signal_row_id="BTC/USD|2026-03-20T11:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.8,
        entry_confidence=0.8,
        entry_fill_interval_begin=interval_begin + timedelta(minutes=5),
        entry_fill_time=interval_begin + timedelta(minutes=5),
        entry_price=110.0,
        quantity=5.0,
        entry_notional=550.0,
        entry_fee=1.1,
        stop_loss_price=107.8,
        take_profit_price=114.4,
        position_id=2,
        opened_at=interval_begin + timedelta(minutes=5),
        updated_at=interval_begin + timedelta(minutes=5),
    )


def test_overview_metrics_and_drawdown_are_computed_from_positions() -> None:
    """The dashboard should compute realized and unrealized PnL without artifact files."""
    positions = (_closed_position(), _open_position())
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
            status="ok",
        ),
        signals=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
            positions=positions,
            latest_prices={"BTC/USD": 120.0},
            cash_balance=9492.8,
        ),
    )

    overview = build_overview_metrics(snapshot=snapshot, trading_config=_config())
    equity_rows = build_equity_curve_rows(
        positions=positions,
        initial_cash=_config().risk.initial_cash,
        latest_prices={"BTC/USD": 120.0},
        fee_bps=_config().risk.fee_bps,
        as_of_time=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
    )
    drawdown_rows = build_drawdown_curve_rows(equity_rows)

    assert overview is not None
    assert round(overview.realized_pnl, 4) == 45.9
    assert round(overview.unrealized_pnl, 4) == 47.7
    assert round(overview.total_pnl, 4) == 93.6
    assert overview.open_position_count == 1
    assert len(equity_rows) == 3
    assert drawdown_rows[-1]["drawdown"] == 0.0


def test_trader_freshness_uses_persisted_engine_state() -> None:
    """The dashboard should summarize trading freshness from the real engine-state rows."""
    base_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    freshness = build_trader_freshness(
        (
            EngineStateSnapshot(
                service_name="paper-trader",
                symbol="BTC/USD",
                last_processed_interval_begin=base_time,
                cooldown_until_interval_begin=None,
                pending_signal_action=None,
                updated_at=base_time + timedelta(seconds=10),
            ),
            EngineStateSnapshot(
                service_name="paper-trader",
                symbol="ETH/USD",
                last_processed_interval_begin=base_time - timedelta(minutes=5),
                cooldown_until_interval_begin=base_time,
                pending_signal_action="BUY",
                updated_at=base_time + timedelta(seconds=15),
            ),
        )
    )

    assert freshness.state == "healthy"
    assert freshness.symbols_tracked == 2
    assert freshness.pending_signal_count == 1
    assert freshness.latest_processed_interval_begin == base_time
    assert freshness.slowest_processed_interval_begin == base_time - timedelta(minutes=5)
