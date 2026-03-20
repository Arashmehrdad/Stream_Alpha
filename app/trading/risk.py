"""Pure risk and execution helpers for the Stream Alpha M5 engine."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.trading.config import PaperTradingConfig
from app.trading.schemas import FeatureCandle, PaperEngineState, PaperPosition, PortfolioContext
from app.trading.schemas import ExitReason


def calculate_entry_fill_price(next_open: float, slippage_bps: float) -> float:
    """Apply adverse slippage to a long entry fill."""
    return next_open * (1.0 + (slippage_bps / 10_000.0))


def calculate_exit_fill_price(next_open: float, slippage_bps: float) -> float:
    """Apply adverse slippage to a long exit fill."""
    return next_open * (1.0 - (slippage_bps / 10_000.0))


def calculate_fee(notional: float, fee_bps: float) -> float:
    """Return the configured trading fee for a notional amount."""
    return notional * (fee_bps / 10_000.0)


def is_cooldown_active(
    state: PaperEngineState,
    candle: FeatureCandle,
) -> bool:
    """Return whether an entry is blocked by the configured cooldown window."""
    return (
        state.cooldown_until_interval_begin is not None
        and candle.interval_begin <= state.cooldown_until_interval_begin
    )


def next_cooldown_boundary(
    candle: FeatureCandle,
    cooldown_candles: int,
) -> datetime | None:
    """Return the timestamp through which entries should stay blocked."""
    if cooldown_candles <= 0:
        return None
    return candle.interval_begin + timedelta(minutes=candle.interval_minutes * cooldown_candles)


def evaluate_barrier_exit(
    position: PaperPosition | None,
    candle: FeatureCandle,
) -> ExitReason | None:
    """Evaluate stop-loss and take-profit against the finalized candle range."""
    if position is None or position.status != "OPEN":
        return None
    if candle.low_price <= position.stop_loss_price:
        return "STOP_LOSS"
    if candle.high_price >= position.take_profit_price:
        return "TAKE_PROFIT"
    return None


def can_open_position(
    *,
    config: PaperTradingConfig,
    state: PaperEngineState,
    open_position: PaperPosition | None,
    candle: FeatureCandle,
    portfolio: PortfolioContext,
) -> bool:
    """Return whether the engine is allowed to open a new long position."""
    if open_position is not None and open_position.status == "OPEN":
        return False
    if is_cooldown_active(state, candle):
        return False
    if portfolio.open_position_count >= config.risk.max_open_positions:
        return False
    return portfolio.available_cash > 0


def capped_entry_cash(
    *,
    config: PaperTradingConfig,
    available_cash: float,
) -> float:
    """Return the cash budget available for one new long entry."""
    fraction = min(config.risk.position_fraction, config.risk.max_exposure_per_asset)
    return max(0.0, available_cash * fraction)
