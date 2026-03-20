"""Tests for the M8 deterministic threshold-based regime workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.regime.config import RegimeConfig, ThresholdConfig
from app.regime.dataset import (
    REQUIRED_SOURCE_COLUMNS,
    RegimeSourceRow,
    _validate_source_columns,
    _validate_symbol_row_counts,
)
from app.regime.service import (
    HIGH_VOL,
    RANGE,
    TREND_DOWN,
    TREND_UP,
    classify_row,
    classify_rows,
    fit_symbol_thresholds,
)


def _make_config(*, min_rows_per_symbol: int = 5) -> RegimeConfig:
    return RegimeConfig(
        source_table="feature_ohlc",
        source_exchange="kraken",
        interval_minutes=5,
        symbols=("BTC/USD", "ETH/USD"),
        artifact_dir="artifacts/regime/m8",
        min_rows_per_symbol=min_rows_per_symbol,
        thresholds=ThresholdConfig(
            high_vol_percentile=75.0,
            trend_abs_momentum_percentile=60.0,
        ),
    )


def _make_row(
    *,
    symbol: str,
    index: int,
    realized_vol_12: float,
    momentum_3: float,
    macd_line_12_26: float,
) -> RegimeSourceRow:
    base_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    interval_begin = base_time + timedelta(minutes=5 * index)
    return RegimeSourceRow(
        symbol=symbol,
        interval_begin=interval_begin,
        as_of_time=interval_begin + timedelta(minutes=5),
        realized_vol_12=realized_vol_12,
        momentum_3=momentum_3,
        macd_line_12_26=macd_line_12_26,
    )


def test_fit_symbol_thresholds_is_deterministic() -> None:
    """The same canonical rows should always fit the same per-symbol thresholds."""
    config = _make_config()
    rows = [
        _make_row(
            symbol="BTC/USD",
            index=index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (1.0, -4.0, -1.0),
                (2.0, -3.0, -1.0),
                (3.0, 0.0, 1.0),
                (4.0, 3.0, 1.0),
                (5.0, 4.0, 1.0),
            ]
        )
    ] + [
        _make_row(
            symbol="ETH/USD",
            index=100 + index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (10.0, -10.0, -1.0),
                (20.0, -8.0, -1.0),
                (30.0, 0.0, 1.0),
                (40.0, 8.0, 1.0),
                (50.0, 10.0, 1.0),
            ]
        )
    ]

    first = fit_symbol_thresholds(rows, config)
    second = fit_symbol_thresholds(rows, config)

    assert first == second
    assert first["BTC/USD"].high_vol_threshold == 4.0
    assert first["BTC/USD"].trend_abs_threshold == pytest.approx(3.4)
    assert first["ETH/USD"].high_vol_threshold == 40.0
    assert first["ETH/USD"].trend_abs_threshold == pytest.approx(8.8)


def test_classification_is_deterministic_and_rule_order_is_explicit() -> None:
    """HIGH_VOL should take precedence over trend rules, then trend up/down, then range."""
    config = _make_config()
    rows = [
        _make_row(
            symbol="BTC/USD",
            index=index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (1.0, -4.0, -1.0),
                (2.0, 4.0, 1.0),
                (3.0, 0.0, 1.0),
                (4.0, 4.0, 1.0),
                (5.0, -5.0, -1.0),
            ]
        )
    ] + [
        _make_row(
            symbol="ETH/USD",
            index=100 + index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (10.0, -10.0, -1.0),
                (20.0, -8.0, -1.0),
                (30.0, 0.0, 1.0),
                (40.0, 8.0, 1.0),
                (50.0, 10.0, 1.0),
            ]
        )
    ]

    thresholds = fit_symbol_thresholds(rows, config)

    first_predictions = classify_rows(rows, thresholds)
    second_predictions = classify_rows(rows, thresholds)

    assert [prediction.regime for prediction in first_predictions] == [
        TREND_DOWN,
        TREND_UP,
        RANGE,
        HIGH_VOL,
        HIGH_VOL,
        TREND_DOWN,
        RANGE,
        RANGE,
        HIGH_VOL,
        HIGH_VOL,
    ]
    assert first_predictions == second_predictions


def test_thresholds_are_applied_per_symbol_not_globally() -> None:
    """Identical feature values should classify differently when symbol thresholds differ."""
    config = _make_config()
    rows = [
        _make_row(
            symbol="BTC/USD",
            index=index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (1.0, -4.0, -1.0),
                (2.0, -3.0, -1.0),
                (3.0, 0.0, 1.0),
                (4.0, 3.0, 1.0),
                (5.0, 4.0, 1.0),
            ]
        )
    ] + [
        _make_row(
            symbol="ETH/USD",
            index=100 + index,
            realized_vol_12=realized_vol,
            momentum_3=momentum,
            macd_line_12_26=macd,
        )
        for index, (realized_vol, momentum, macd) in enumerate(
            [
                (10.0, -10.0, -1.0),
                (20.0, -8.0, -1.0),
                (30.0, 0.0, 1.0),
                (40.0, 8.0, 1.0),
                (50.0, 10.0, 1.0),
            ]
        )
    ]
    thresholds = fit_symbol_thresholds(rows, config)

    shared_features_btc = _make_row(
        symbol="BTC/USD",
        index=999,
        realized_vol_12=5.0,
        momentum_3=2.0,
        macd_line_12_26=1.0,
    )
    shared_features_eth = _make_row(
        symbol="ETH/USD",
        index=999,
        realized_vol_12=5.0,
        momentum_3=2.0,
        macd_line_12_26=1.0,
    )

    assert classify_row(shared_features_btc, thresholds) == HIGH_VOL
    assert classify_row(shared_features_eth, thresholds) == RANGE


def test_validate_source_columns_fails_clearly_when_canonical_columns_are_missing() -> None:
    """Missing required feature columns should stop the workflow with an explicit error."""
    source_schema = tuple(
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in {"momentum_3", "macd_line_12_26"}
    )

    with pytest.raises(
        ValueError,
        match="Regime source schema does not include the required canonical columns",
    ):
        _validate_source_columns(source_schema, REQUIRED_SOURCE_COLUMNS)


def test_validate_symbol_row_counts_fails_clearly_when_rows_are_insufficient() -> None:
    """Per-symbol minimum row enforcement should name the missing symbols and counts."""
    with pytest.raises(
        ValueError,
        match="Required at least 5 rows for each symbol; BTC/USD \\(4 found\\)",
    ):
        _validate_symbol_row_counts(
            {"BTC/USD": 4, "ETH/USD": 5},
            min_rows_per_symbol=5,
            symbols=("BTC/USD", "ETH/USD"),
        )
