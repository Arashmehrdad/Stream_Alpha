"""Deterministic tests for finalized OHLC feature calculations."""

# pylint: disable=duplicate-code

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.common.models import OhlcEvent
from app.features.engine import compute_feature_row


def _build_candle(index: int) -> OhlcEvent:
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    interval_begin = base_time + timedelta(minutes=5 * index)
    close_price = 100.0 + (index * 1.75) + ((index % 3) * 0.2)
    volume = 20.0 + (index * 0.9) + ((index % 4) * 0.35)
    return OhlcEvent(
        event_id=f"evt-{index}",
        app_name="streamalpha",
        source_exchange="kraken",
        channel="ohlc",
        message_type="update",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        open_price=close_price - 0.75,
        high_price=close_price + 0.9,
        low_price=close_price - 1.1,
        close_price=close_price,
        vwap=close_price - 0.15,
        trade_count=10 + index,
        volume=volume,
        received_at=interval_begin + timedelta(minutes=5, seconds=1),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_stddev(values: list[float]) -> float:
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _ema_last(values: list[float], period: int) -> float:
    seed = _mean(values[:period])
    previous = seed
    smoothing = 2.0 / (period + 1)
    for value in values[period:]:
        previous = ((value - previous) * smoothing) + previous
    return previous


def _rsi_last(closes: list[float], period: int) -> float:
    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = _mean(gains[:period])
    average_loss = _mean(losses[:period])
    for index in range(period, len(changes)):
        average_gain = ((average_gain * (period - 1)) + gains[index]) / period
        average_loss = ((average_loss * (period - 1)) + losses[index]) / period

    if average_loss == 0.0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def test_compute_feature_row_matches_expected_windows() -> None:
    """Feature math should match the expected rolling-window formulas."""
    candles = [_build_candle(index) for index in range(30)]
    computed_at = candles[-1].interval_end + timedelta(seconds=10)

    row = compute_feature_row(candles, computed_at=computed_at)

    assert row is not None

    closes = [candle.close_price for candle in candles]
    volumes = [candle.volume for candle in candles]
    log_returns = [
        math.log(closes[index] / closes[index - 1])
        for index in range(1, len(closes))
    ]
    return_window = log_returns[-12:]
    volume_window = volumes[-12:]
    close_window = closes[-12:]

    assert row.as_of_time == candles[-1].interval_end
    assert row.computed_at == computed_at
    assert row.raw_event_id == candles[-1].event_id
    assert row.log_return_1 == pytest.approx(log_returns[-1])
    assert row.log_return_3 == pytest.approx(math.log(closes[-1] / closes[-4]))
    assert row.momentum_3 == pytest.approx((closes[-1] / closes[-4]) - 1.0)
    assert row.return_mean_12 == pytest.approx(_mean(return_window))
    assert row.return_std_12 == pytest.approx(_population_stddev(return_window))
    assert row.realized_vol_12 == pytest.approx(
        math.sqrt(sum(value * value for value in return_window))
    )
    assert row.rsi_14 == pytest.approx(_rsi_last(closes, 14))
    assert row.macd_line_12_26 == pytest.approx(
        _ema_last(closes, 12) - _ema_last(closes, 26)
    )
    assert row.volume_mean_12 == pytest.approx(_mean(volume_window))
    assert row.volume_std_12 == pytest.approx(_population_stddev(volume_window))
    assert row.volume_zscore_12 == pytest.approx(
        (volumes[-1] - _mean(volume_window)) / _population_stddev(volume_window)
    )
    assert row.close_zscore_12 == pytest.approx(
        (closes[-1] - _mean(close_window)) / _population_stddev(close_window)
    )
    assert row.lag_log_return_1 == pytest.approx(log_returns[-2])
    assert row.lag_log_return_2 == pytest.approx(log_returns[-3])
    assert row.lag_log_return_3 == pytest.approx(log_returns[-4])


def test_compute_feature_row_requires_warmup_history() -> None:
    """No finalized feature row should be emitted before the warmup window is ready."""
    candles = [_build_candle(index) for index in range(25)]

    row = compute_feature_row(
        candles,
        computed_at=candles[-1].interval_end + timedelta(seconds=10),
    )

    assert row is None
