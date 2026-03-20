"""Pure risk-control tests for Stream Alpha M5."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime, timezone

from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.risk import can_open_position, capped_entry_cash, evaluate_barrier_exit
from app.trading.risk import is_cooldown_active
from app.trading.schemas import FeatureCandle, PaperEngineState, PortfolioContext
from app.trading.schemas import PaperPosition


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
            max_open_positions=2,
            max_exposure_per_asset=0.25,
        ),
    )


def _candle() -> FeatureCandle:
    interval_begin = datetime(2026, 3, 20, 15, 0, tzinfo=timezone.utc)
    return FeatureCandle(
        id=1,
        source_exchange="kraken",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=datetime(2026, 3, 20, 15, 5, tzinfo=timezone.utc),
        as_of_time=datetime(2026, 3, 20, 15, 5, tzinfo=timezone.utc),
        raw_event_id="evt-1",
        open_price=100.0,
        high_price=105.0,
        low_price=95.0,
        close_price=101.0,
    )


def test_cooldown_blocks_entries() -> None:
    """Cooldown should remain active through the configured blocked candle."""
    state = PaperEngineState(
        service_name="paper-trader",
        symbol="BTC/USD",
        cooldown_until_interval_begin=_candle().interval_begin,
    )
    assert is_cooldown_active(state, _candle()) is True


def test_max_open_positions_blocks_entries() -> None:
    """Portfolio-wide max-open-position controls should block new longs."""
    allowed = can_open_position(
        config=_config(),
        state=PaperEngineState(service_name="paper-trader", symbol="BTC/USD"),
        open_position=None,
        candle=_candle(),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )
    blocked = can_open_position(
        config=_config(),
        state=PaperEngineState(service_name="paper-trader", symbol="BTC/USD"),
        open_position=None,
        candle=_candle(),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=2),
    )

    assert allowed is True
    assert blocked is False


def test_entry_cash_is_capped_by_asset_exposure() -> None:
    """Max exposure per asset should cap the cash budget for a new entry."""
    assert capped_entry_cash(config=_config(), available_cash=10_000.0) == 2500.0


def test_stop_loss_beats_take_profit_in_same_candle() -> None:
    """Barrier evaluation should choose STOP_LOSS first when both trigger."""
    position = PaperPosition(
        service_name="paper-trader",
        symbol="BTC/USD",
        status="OPEN",
        entry_signal_interval_begin=_candle().interval_begin,
        entry_signal_as_of_time=_candle().as_of_time,
        entry_signal_row_id="BTC/USD|2026-03-20T15:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.7,
        entry_confidence=0.7,
        entry_fill_interval_begin=_candle().interval_begin,
        entry_fill_time=_candle().interval_begin,
        entry_price=100.0,
        quantity=1.0,
        entry_notional=100.0,
        entry_fee=0.2,
        stop_loss_price=98.0,
        take_profit_price=102.0,
    )

    assert evaluate_barrier_exit(position, _candle()) == "STOP_LOSS"
