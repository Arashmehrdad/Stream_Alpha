"""Pure feature calculations for finalized OHLC candles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

from app.common.models import OhlcEvent
from app.features.models import FeatureOhlcRow


MIN_FINALIZED_CANDLES = 26
_RETURN_WINDOW = 12
_RSI_PERIOD = 14
_MACD_FAST_PERIOD = 12
_MACD_SLOW_PERIOD = 26


class ReturnMetrics(NamedTuple):
    """Return-derived feature values for one finalized candle."""

    current_log_return: float
    log_return_3: float
    momentum_3: float
    return_mean_12: float
    return_std_12: float
    realized_vol_12: float
    lag_log_return_1: float
    lag_log_return_2: float
    lag_log_return_3: float


class LevelMetrics(NamedTuple):
    """Rolling level statistics for closes or volumes."""

    mean_value: float
    stddev_value: float
    zscore_value: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_stddev(values: Sequence[float]) -> float:
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _safe_zscore(value: float, mean_value: float, stddev_value: float) -> float:
    if stddev_value == 0.0:
        return 0.0
    return (value - mean_value) / stddev_value


def _ema_series(values: Sequence[float], period: int) -> list[float | None]:
    """Return an EMA series seeded with the standard SMA of the first window."""
    series: list[float | None] = [None] * len(values)
    if len(values) < period:
        return series

    smoothing = 2.0 / (period + 1)
    seed = _mean(values[:period])
    series[period - 1] = seed
    previous = seed

    for index in range(period, len(values)):
        current = ((values[index] - previous) * smoothing) + previous
        series[index] = current
        previous = current

    return series


def _rsi_series(closes: Sequence[float], period: int) -> list[float | None]:
    """Return RSI values using Wilder smoothing after the initial seed window."""
    series: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return series

    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = _mean(gains[:period])
    average_loss = _mean(losses[:period])
    series[period] = _rsi_from_averages(average_gain, average_loss)

    for index in range(period, len(changes)):
        average_gain = ((average_gain * (period - 1)) + gains[index]) / period
        average_loss = ((average_loss * (period - 1)) + losses[index]) / period
        series[index + 1] = _rsi_from_averages(average_gain, average_loss)

    return series


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    if average_loss == 0.0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _require_value(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Feature value {label} is not available yet")
    return value


def _log_return_series(closes: Sequence[float]) -> list[float | None]:
    series: list[float | None] = [None]
    for index in range(1, len(closes)):
        series.append(math.log(closes[index] / closes[index - 1]))
    return series


def _return_metrics(
    closes: Sequence[float],
    log_returns_1: Sequence[float | None],
) -> ReturnMetrics:
    current_log_return = _require_value(log_returns_1[-1], "log_return_1")
    log_return_3 = math.log(closes[-1] / closes[-4])
    momentum_3 = (closes[-1] / closes[-4]) - 1.0
    return_window = [
        _require_value(value, f"log_return_1[{index}]")
        for index, value in enumerate(log_returns_1[-_RETURN_WINDOW:], start=1)
    ]
    return_mean_12 = _mean(return_window)
    return_std_12 = _population_stddev(return_window)
    realized_vol_12 = math.sqrt(sum(value * value for value in return_window))
    lag_log_return_1 = _require_value(log_returns_1[-2], "lag_log_return_1")
    lag_log_return_2 = _require_value(log_returns_1[-3], "lag_log_return_2")
    lag_log_return_3 = _require_value(log_returns_1[-4], "lag_log_return_3")
    return ReturnMetrics(
        current_log_return=current_log_return,
        log_return_3=log_return_3,
        momentum_3=momentum_3,
        return_mean_12=return_mean_12,
        return_std_12=return_std_12,
        realized_vol_12=realized_vol_12,
        lag_log_return_1=lag_log_return_1,
        lag_log_return_2=lag_log_return_2,
        lag_log_return_3=lag_log_return_3,
    )


def _rolling_level_metrics(values: Sequence[float]) -> LevelMetrics:
    window = list(values[-_RETURN_WINDOW:])
    mean_value = _mean(window)
    stddev_value = _population_stddev(window)
    zscore_value = _safe_zscore(values[-1], mean_value, stddev_value)
    return LevelMetrics(
        mean_value=mean_value,
        stddev_value=stddev_value,
        zscore_value=zscore_value,
    )


def _macd_line(closes: Sequence[float]) -> float:
    fast_ema = _ema_series(closes, _MACD_FAST_PERIOD)
    slow_ema = _ema_series(closes, _MACD_SLOW_PERIOD)
    return _require_value(fast_ema[-1], "ema_12") - _require_value(
        slow_ema[-1],
        "ema_26",
    )


def compute_feature_row(
    finalized_candles: Sequence[OhlcEvent],
    *,
    computed_at: datetime,
) -> FeatureOhlcRow | None:
    """Compute one finalized feature row from an ordered finalized candle history."""
    if len(finalized_candles) < MIN_FINALIZED_CANDLES:
        return None

    candles = list(finalized_candles)
    closes = [candle.close_price for candle in candles]
    volumes = [candle.volume for candle in candles]
    current = candles[-1]

    log_returns_1 = _log_return_series(closes)
    return_metrics = _return_metrics(closes, log_returns_1)
    volume_metrics = _rolling_level_metrics(volumes)
    close_metrics = _rolling_level_metrics(closes)
    rsi_14 = _require_value(_rsi_series(closes, _RSI_PERIOD)[-1], "rsi_14")
    macd_line_12_26 = _macd_line(closes)

    return FeatureOhlcRow(
        source_exchange=current.source_exchange,
        symbol=current.symbol,
        interval_minutes=current.interval_minutes,
        interval_begin=current.interval_begin,
        interval_end=current.interval_end,
        as_of_time=current.interval_end,
        computed_at=computed_at,
        raw_event_id=current.event_id,
        open_price=current.open_price,
        high_price=current.high_price,
        low_price=current.low_price,
        close_price=current.close_price,
        vwap=current.vwap,
        trade_count=current.trade_count,
        volume=current.volume,
        log_return_1=return_metrics.current_log_return,
        log_return_3=return_metrics.log_return_3,
        momentum_3=return_metrics.momentum_3,
        return_mean_12=return_metrics.return_mean_12,
        return_std_12=return_metrics.return_std_12,
        realized_vol_12=return_metrics.realized_vol_12,
        rsi_14=rsi_14,
        macd_line_12_26=macd_line_12_26,
        volume_mean_12=volume_metrics.mean_value,
        volume_std_12=volume_metrics.stddev_value,
        volume_zscore_12=volume_metrics.zscore_value,
        close_zscore_12=close_metrics.zscore_value,
        lag_log_return_1=return_metrics.lag_log_return_1,
        lag_log_return_2=return_metrics.lag_log_return_2,
        lag_log_return_3=return_metrics.lag_log_return_3,
    )
