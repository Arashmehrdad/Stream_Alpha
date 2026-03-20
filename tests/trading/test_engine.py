"""Deterministic engine tests for Stream Alpha M5."""

# pylint: disable=duplicate-code,line-too-long,too-many-lines,too-many-arguments

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.engine import process_candle
from app.trading.schemas import FeatureCandle, PaperEngineState, PendingSignalState, PortfolioContext
from app.trading.schemas import SignalDecision


def _config(**risk_overrides) -> PaperTradingConfig:
    risk_values = {
        "initial_cash": 10_000.0,
        "position_fraction": 0.25,
        "fee_bps": 20.0,
        "slippage_bps": 5.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "cooldown_candles": 1,
        "max_open_positions": 3,
        "max_exposure_per_asset": 0.25,
    }
    risk_values.update(risk_overrides)
    risk = RiskConfig(**risk_values)
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir="artifacts/paper_trading",
        risk=risk,
    )


def _candle(
    *,
    interval_begin: datetime,
    open_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    close_price: float | None = None,
    symbol: str = "BTC/USD",
) -> FeatureCandle:
    return FeatureCandle(
        id=1,
        source_exchange="kraken",
        symbol=symbol,
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        as_of_time=interval_begin + timedelta(minutes=5),
        raw_event_id="evt-1",
        open_price=open_price,
        high_price=open_price if high_price is None else high_price,
        low_price=open_price if low_price is None else low_price,
        close_price=open_price if close_price is None else close_price,
    )


def _pending(
    signal: str,
    *,
    interval_begin: datetime,
    regime_label: str | None = None,
    approved_notional: float | None = None,
    risk_outcome: str | None = None,
    risk_reason_codes: tuple[str, ...] = (),
) -> PendingSignalState:
    return PendingSignalState(
        signal=signal,
        signal_interval_begin=interval_begin,
        signal_as_of_time=interval_begin + timedelta(minutes=5),
        row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
        reason=f"{signal.lower()}-reason",
        prob_up=0.7 if signal == "BUY" else 0.3,
        prob_down=0.3 if signal == "BUY" else 0.7,
        confidence=0.7,
        predicted_class="UP" if signal == "BUY" else "DOWN",
        model_name="logistic_regression",
        regime_label=regime_label,
        approved_notional=approved_notional,
        risk_outcome=risk_outcome,
        risk_reason_codes=risk_reason_codes,
    )


def _signal(
    signal: str,
    *,
    interval_begin: datetime,
    regime_label: str | None = None,
) -> SignalDecision:
    return SignalDecision(
        symbol="BTC/USD",
        signal=signal,
        reason=f"{signal.lower()}-reason",
        prob_up=0.7 if signal == "BUY" else 0.3 if signal == "SELL" else 0.5,
        prob_down=0.3 if signal == "BUY" else 0.7 if signal == "SELL" else 0.5,
        confidence=0.7 if signal != "HOLD" else 0.5,
        predicted_class="UP" if signal == "BUY" else "DOWN",
        row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
        as_of_time=interval_begin + timedelta(minutes=5),
        model_name="logistic_regression",
        regime_label=regime_label,
    )


def test_buy_entry_uses_next_open_fill_math() -> None:
    """BUY signals should open long at the next candle open plus adverse slippage."""
    config = _config()
    signal_candle = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
    fill_candle = _candle(interval_begin=datetime(2026, 3, 20, 10, 5, tzinfo=timezone.utc), open_price=100.0)
    state = PaperEngineState(
        service_name="paper-trader",
        symbol="BTC/USD",
        pending_signal=_pending("BUY", interval_begin=signal_candle, regime_label="TREND_UP"),
    )

    result = process_candle(
        config=config,
        candle=fill_candle,
        state=state,
        open_position=None,
        signal=_signal("HOLD", interval_begin=fill_candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )

    assert result.created_position is not None
    assert round(result.created_position.entry_price, 6) == 100.05
    assert round(result.created_position.entry_notional, 6) == round(2500.0 / 1.002, 6)
    assert result.created_position.entry_regime_label == "TREND_UP"
    assert round(result.ledger_entries[0].fee, 6) == round(
        result.created_position.entry_notional * 0.002,
        6,
    )
    assert result.ledger_entries[0].regime_label == "TREND_UP"


def test_sell_exit_uses_next_open_fill_math() -> None:
    """SELL signals should close an existing long at the next candle open minus slippage."""
    config = _config()
    signal_candle = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
    fill_candle = _candle(interval_begin=datetime(2026, 3, 20, 10, 5, tzinfo=timezone.utc), open_price=105.0)
    entry_result = process_candle(
        config=config,
        candle=fill_candle,
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            pending_signal=_pending("BUY", interval_begin=signal_candle, regime_label="RANGE"),
        ),
        open_position=None,
        signal=_signal("HOLD", interval_begin=fill_candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )
    open_position = entry_result.open_position
    assert open_position is not None

    exit_result = process_candle(
        config=config,
        candle=_candle(
            interval_begin=datetime(2026, 3, 20, 10, 10, tzinfo=timezone.utc),
            open_price=110.0,
        ),
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            pending_signal=_pending(
                "SELL",
                interval_begin=fill_candle.interval_begin,
                regime_label="TREND_DOWN",
            ),
        ),
        open_position=open_position,
        signal=_signal("HOLD", interval_begin=datetime(2026, 3, 20, 10, 10, tzinfo=timezone.utc)),
        portfolio=PortfolioContext(available_cash=7_495.0, open_position_count=1),
    )

    assert exit_result.closed_position is not None
    assert round(exit_result.closed_position.exit_price or 0.0, 6) == 109.945
    assert exit_result.closed_position.entry_regime_label == "RANGE"
    assert exit_result.closed_position.exit_regime_label == "TREND_DOWN"
    assert exit_result.ledger_entries[0].action == "SELL"
    assert exit_result.ledger_entries[0].regime_label == "TREND_DOWN"


def test_sell_while_flat_does_not_short() -> None:
    """SELL signals while flat should do nothing and never open a short."""
    candle = _candle(interval_begin=datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc), open_price=100.0)
    result = process_candle(
        config=_config(),
        candle=candle,
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            pending_signal=_pending("SELL", interval_begin=datetime(2026, 3, 20, 10, 55, tzinfo=timezone.utc)),
        ),
        open_position=None,
        signal=_signal("HOLD", interval_begin=candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )

    assert result.open_position is None
    assert result.closed_position is None
    assert not result.ledger_entries


def test_stop_loss_and_take_profit_respect_conservative_ordering() -> None:
    """If both barriers hit in one candle, STOP_LOSS should win conservatively."""
    config = _config()
    buy_fill_candle = _candle(interval_begin=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc), open_price=100.0)
    entry_result = process_candle(
        config=config,
        candle=buy_fill_candle,
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            pending_signal=_pending(
                "BUY",
                interval_begin=datetime(2026, 3, 20, 11, 55, tzinfo=timezone.utc),
                regime_label="TREND_UP",
            ),
        ),
        open_position=None,
        signal=_signal("HOLD", interval_begin=buy_fill_candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )
    open_position = entry_result.open_position
    assert open_position is not None

    barrier_result = process_candle(
        config=config,
        candle=_candle(
            interval_begin=datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
            open_price=100.0,
            high_price=110.0,
            low_price=95.0,
            close_price=104.0,
        ),
        state=PaperEngineState(service_name="paper-trader", symbol="BTC/USD"),
        open_position=open_position,
        signal=_signal(
            "HOLD",
            interval_begin=datetime(2026, 3, 20, 12, 5, tzinfo=timezone.utc),
            regime_label="HIGH_VOL",
        ),
        portfolio=PortfolioContext(available_cash=7_495.0, open_position_count=1),
    )

    assert barrier_result.closed_position is not None
    assert barrier_result.closed_position.exit_reason == "STOP_LOSS"
    assert barrier_result.closed_position.exit_regime_label == "HIGH_VOL"
    assert barrier_result.ledger_entries[0].regime_label == "HIGH_VOL"


def test_risk_approved_pending_buy_is_not_dropped_by_legacy_execution_inputs() -> None:
    """Execution should fill an already approved BUY instead of reapplying legacy gates."""
    candle = _candle(interval_begin=datetime(2026, 3, 20, 13, 0, tzinfo=timezone.utc), open_price=100.0)
    result = process_candle(
        config=_config(max_open_positions=1),
        candle=candle,
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            cooldown_until_interval_begin=candle.interval_begin,
            pending_signal=_pending(
                "BUY",
                interval_begin=datetime(2026, 3, 20, 12, 55, tzinfo=timezone.utc),
                regime_label="TREND_UP",
                approved_notional=900.0,
                risk_outcome="APPROVED",
                risk_reason_codes=("BUY_APPROVED",),
            ),
        ),
        open_position=None,
        signal=_signal("HOLD", interval_begin=candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=1),
    )

    assert result.created_position is not None
    assert result.state.pending_signal is None
    assert result.created_position.entry_notional == 900.0
    assert result.created_position.entry_approved_notional == 900.0
    assert result.created_position.entry_risk_outcome == "APPROVED"
    assert result.ledger_entries[0].approved_notional == 900.0
    assert result.ledger_entries[0].risk_reason_codes == ("BUY_APPROVED",)


def test_max_exposure_per_asset_caps_entry_size_and_ledger_fields() -> None:
    """Entry sizing should respect max exposure and keep ledger fields explicit."""
    config = _config(position_fraction=0.9, max_exposure_per_asset=0.1)
    candle = _candle(interval_begin=datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc), open_price=100.0)

    result = process_candle(
        config=config,
        candle=candle,
        state=PaperEngineState(
            service_name="paper-trader",
            symbol="BTC/USD",
            pending_signal=_pending("BUY", interval_begin=datetime(2026, 3, 20, 13, 55, tzinfo=timezone.utc)),
        ),
        open_position=None,
        signal=_signal("HOLD", interval_begin=candle.interval_begin),
        portfolio=PortfolioContext(available_cash=10_000.0, open_position_count=0),
    )

    assert result.created_position is not None
    assert round(result.created_position.entry_notional, 6) == round(1000.0 / 1.002, 6)
    assert result.ledger_entries[0].reason == "SIGNAL_BUY"
    assert result.ledger_entries[0].cash_flow < 0.0
