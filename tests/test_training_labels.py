"""Tests for M3 next-3-candle label construction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.training.dataset import TrainingConfig, _build_symbol_samples, ModelHyperparameters


def _make_config() -> TrainingConfig:
    return TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=("close_price", "log_return_1"),
        label_horizon_candles=3,
        purge_gap_candles=3,
        test_folds=5,
        first_train_fraction=0.5,
        test_fraction=0.1,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m3",
        models=ModelHyperparameters(
            logistic_regression={"max_iter": 100},
            hist_gradient_boosting={"max_iter": 10},
        ),
    )


def _make_row(index: int, close_price: float) -> dict[str, object]:
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    interval_begin = base_time + timedelta(minutes=5 * index)
    return {
        "symbol": "BTC/USD",
        "interval_begin": interval_begin,
        "as_of_time": interval_begin + timedelta(minutes=5),
        "close_price": close_price,
        "log_return_1": 0.01 * index,
    }


def test_label_logic_matches_the_next_three_candle_rule() -> None:
    """Labels should use the close exactly 3 candles ahead for the same symbol."""
    config = _make_config()
    rows = [
        _make_row(0, 10.0),
        _make_row(1, 11.0),
        _make_row(2, 12.0),
        _make_row(3, 13.0),
        _make_row(4, 14.0),
        _make_row(5, 15.0),
        _make_row(6, 12.0),
        _make_row(7, 16.0),
    ]

    built = _build_symbol_samples(rows, config)

    assert len(built.samples) == 2
    assert built.samples[0].future_return_3 == (12.0 / 13.0) - 1.0
    assert built.samples[0].label == 0
    assert built.samples[1].future_return_3 == (16.0 / 14.0) - 1.0
    assert built.samples[1].label == 1


def test_zero_future_return_rows_are_dropped() -> None:
    """Rows with a flat 3-candle future return should be excluded from training."""
    config = _make_config()
    rows = [
        _make_row(0, 10.0),
        _make_row(1, 11.0),
        _make_row(2, 12.0),
        _make_row(3, 13.0),
        _make_row(4, 14.0),
        _make_row(5, 15.0),
        _make_row(6, 13.0),
    ]

    built = _build_symbol_samples(rows, config)

    assert len(built.samples) == 0
    assert built.dropped_flat_returns == 1
