"""Pure M10 risk-engine tests for Stream Alpha."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.risk_engine import (
    BUY_APPROVED,
    HOLD_NO_OP,
    KILL_SWITCH_ENABLED,
    LOSS_STREAK_COOLDOWN_ACTIVE,
    MAX_ASSET_EXPOSURE_CLAMPED,
    MAX_DAILY_LOSS_BREACHED,
    MAX_DRAWDOWN_BREACHED,
    MAX_TOTAL_EXPOSURE_CLAMPED,
    SELL_EXIT_APPROVED,
    VOLATILITY_SIZE_ADJUSTED,
    evaluate_risk,
)
from app.trading.schemas import FeatureCandle, PaperPosition, PortfolioContext, ServiceRiskState
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
        "max_open_positions": 2,
        "max_exposure_per_asset": 0.25,
        "max_total_exposure": 0.60,
        "max_daily_loss_amount": 200.0,
        "max_drawdown_pct": 0.10,
        "loss_streak_limit": 2,
        "loss_streak_cooldown_candles": 3,
        "kill_switch_enabled": False,
        "min_trade_notional": 50.0,
        "volatility_target_realized_vol": 0.03,
        "min_volatility_size_multiplier": 0.40,
        "enable_confidence_weighted_sizing": False,
        "min_confidence_size_multiplier": 0.50,
        "regime_position_fraction_caps": {
            "TREND_UP": 0.25,
            "RANGE": 0.20,
            "TREND_DOWN": 0.15,
            "HIGH_VOL": 0.10,
        },
    }
    risk_values.update(risk_overrides)
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir="artifacts/paper_trading",
        risk=RiskConfig(**risk_values),
    )


def _candle(*, realized_vol_12: float = 0.03) -> FeatureCandle:
    interval_begin = datetime(2026, 3, 20, 15, 0, tzinfo=timezone.utc)
    return FeatureCandle(
        id=1,
        source_exchange="kraken",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        as_of_time=interval_begin + timedelta(minutes=5),
        raw_event_id="evt-1",
        open_price=100.0,
        high_price=105.0,
        low_price=95.0,
        close_price=101.0,
        realized_vol_12=realized_vol_12,
    )


def _signal(
    signal: str = "BUY",
    *,
    trade_allowed: bool = True,
    confidence: float = 0.80,
    regime_label: str = "TREND_UP",
) -> SignalDecision:
    interval_begin = _candle().interval_begin
    return SignalDecision(
        symbol="BTC/USD",
        signal=signal,
        reason=signal.lower(),
        prob_up=0.70 if signal == "BUY" else 0.30 if signal == "SELL" else 0.50,
        prob_down=0.30 if signal == "BUY" else 0.70 if signal == "SELL" else 0.50,
        confidence=confidence,
        predicted_class="UP" if signal == "BUY" else "DOWN",
        row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
        as_of_time=_candle().as_of_time,
        model_name="logistic_regression",
        regime_label=regime_label,
        regime_run_id="20260320T120000Z",
        trade_allowed=trade_allowed,
    )


def _portfolio(
    *,
    available_cash: float = 10_000.0,
    current_equity: float = 10_000.0,
    total_open_exposure_notional: float = 0.0,
    current_symbol_exposure_notional: float = 0.0,
    open_position_count: int = 0,
) -> PortfolioContext:
    return PortfolioContext(
        available_cash=available_cash,
        open_position_count=open_position_count,
        current_equity=current_equity,
        total_open_exposure_notional=total_open_exposure_notional,
        current_symbol_exposure_notional=current_symbol_exposure_notional,
    )


def _service_risk_state(**overrides) -> ServiceRiskState:
    payload = {
        "service_name": "paper-trader",
        "trading_day": date(2026, 3, 20),
        "realized_pnl_today": 0.0,
        "equity_high_watermark": 10_000.0,
        "current_equity": 10_000.0,
        "loss_streak_count": 0,
        "loss_streak_cooldown_until_interval_begin": None,
        "kill_switch_enabled": False,
    }
    payload.update(overrides)
    return ServiceRiskState(**payload)


def _open_position() -> PaperPosition:
    interval_begin = _candle().interval_begin
    return PaperPosition(
        service_name="paper-trader",
        symbol="BTC/USD",
        status="OPEN",
        entry_signal_interval_begin=interval_begin,
        entry_signal_as_of_time=_candle().as_of_time,
        entry_signal_row_id="BTC/USD|2026-03-20T15:00:00Z",
        entry_reason="buy",
        entry_model_name="logistic_regression",
        entry_prob_up=0.7,
        entry_confidence=0.7,
        entry_fill_interval_begin=interval_begin,
        entry_fill_time=interval_begin,
        entry_price=100.0,
        quantity=10.0,
        entry_notional=1000.0,
        entry_fee=2.0,
        stop_loss_price=98.0,
        take_profit_price=104.0,
    )


def test_kill_switch_blocks_buy() -> None:
    decision = evaluate_risk(
        config=_config(kill_switch_enabled=True),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "BLOCKED"
    assert decision.reason_codes == (KILL_SWITCH_ENABLED,)


def test_daily_loss_breach_blocks_buy() -> None:
    decision = evaluate_risk(
        config=_config(max_daily_loss_amount=200.0),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(realized_pnl_today=-250.0),
    )

    assert decision.outcome == "BLOCKED"
    assert decision.reason_codes == (MAX_DAILY_LOSS_BREACHED,)


def test_drawdown_breach_blocks_buy() -> None:
    decision = evaluate_risk(
        config=_config(max_drawdown_pct=0.10),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(current_equity=8_000.0),
        service_risk_state=_service_risk_state(
            equity_high_watermark=10_000.0,
            current_equity=8_000.0,
        ),
    )

    assert decision.outcome == "BLOCKED"
    assert decision.reason_codes == (MAX_DRAWDOWN_BREACHED,)


def test_active_loss_streak_cooldown_blocks_buy() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(
            loss_streak_cooldown_until_interval_begin=_candle().interval_begin,
        ),
    )

    assert decision.outcome == "BLOCKED"
    assert decision.reason_codes == (LOSS_STREAK_COOLDOWN_ACTIVE,)


def test_per_asset_cap_modifies_buy() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(current_symbol_exposure_notional=2_200.0),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "MODIFIED"
    assert MAX_ASSET_EXPOSURE_CLAMPED in decision.reason_codes
    assert round(decision.approved_notional, 6) == 300.0


def test_total_exposure_cap_modifies_buy() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(total_open_exposure_notional=5_800.0),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "MODIFIED"
    assert MAX_TOTAL_EXPOSURE_CLAMPED in decision.reason_codes
    assert round(decision.approved_notional, 6) == 200.0


def test_high_volatility_reduces_approved_notional() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(realized_vol_12=0.12),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "MODIFIED"
    assert VOLATILITY_SIZE_ADJUSTED in decision.reason_codes
    assert round(decision.approved_notional, 6) == round(decision.requested_notional * 0.40, 6)


def test_sell_with_open_position_stays_approved() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal("SELL"),
        open_position=_open_position(),
        portfolio=_portfolio(current_symbol_exposure_notional=1_000.0),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "APPROVED"
    assert decision.reason_codes == (SELL_EXIT_APPROVED,)
    assert decision.approved_notional == 1_000.0


def test_hold_becomes_approved_no_op() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal("HOLD"),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "APPROVED"
    assert decision.reason_codes == (HOLD_NO_OP,)
    assert decision.approved_notional == 0.0


def test_every_evaluation_emits_explicit_outcome_and_reasons() -> None:
    decision = evaluate_risk(
        config=_config(),
        candle=_candle(),
        signal=_signal(),
        open_position=None,
        portfolio=_portfolio(),
        service_risk_state=_service_risk_state(),
    )

    assert decision.outcome == "APPROVED"
    assert decision.reason_codes == (BUY_APPROVED,)
