"""Focused tests for historical data sufficiency and readiness reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.common.config import PostgresSettings
from app.training import data_readiness as readiness_module
from app.training.dataset import DatasetSample, TrainingConfig


def _config() -> TrainingConfig:
    return TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD", "ETH/USD"),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=("realized_vol_12", "momentum_3", "macd_line_12_26"),
        label_horizon_candles=3,
        purge_gap_candles=3,
        test_folds=5,
        first_train_fraction=0.5,
        test_fraction=0.1,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m7",
        models={"autogluon_tabular": {"time_limit": 900}},
    )


def _sample(
    *,
    symbol: str,
    as_of_time: datetime,
    label: int,
    future_return_3: float,
) -> DatasetSample:
    return DatasetSample(
        row_id=f"{symbol}|{as_of_time.isoformat()}",
        symbol=symbol,
        interval_begin=as_of_time - timedelta(minutes=5),
        as_of_time=as_of_time,
        close_price=100.0,
        future_close_price=101.0,
        future_return_3=future_return_3,
        label=label,
        persistence_prediction=label,
        features={
            "symbol": symbol,
            "realized_vol_12": 0.02,
            "momentum_3": 0.01,
            "macd_line_12_26": 0.005,
        },
    )


def _fake_settings():
    return SimpleNamespace(
        postgres=PostgresSettings(
            host="127.0.0.1",
            port=5432,
            database="streamalpha",
            user="streamalpha",
            password="change-me-local-only",
        ),
        kraken=SimpleNamespace(ohlc_interval_minutes=5),
        tables=SimpleNamespace(raw_ohlc="raw_ohlc", feature_ohlc="feature_ohlc"),
    )


def test_compute_gap_windows_counts_missing_intervals() -> None:
    """Gap detection should count contiguous missing candles in 5-minute space."""
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    intervals = (
        start,
        start + timedelta(minutes=5),
        start + timedelta(minutes=20),
    )

    gap_count, gaps = readiness_module._compute_gap_windows(  # pylint: disable=protected-access
        intervals,
        interval_minutes=5,
    )

    assert gap_count == 2
    assert gaps[0].start == start + timedelta(minutes=10)
    assert gaps[0].end == start + timedelta(minutes=15)


def test_build_data_readiness_report_writes_symbol_and_gap_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persisted readiness artifact should capture coverage, gaps, and walk-forward truth."""
    base_time = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    samples = (
        _sample(symbol="BTC/USD", as_of_time=base_time, label=1, future_return_3=0.01),
        _sample(
            symbol="ETH/USD",
            as_of_time=base_time + timedelta(minutes=5),
            label=0,
            future_return_3=-0.02,
        ),
    )
    dataset = SimpleNamespace(
        samples=samples,
        manifest={
            "loaded_rows": 200,
            "eligible_rows": 2,
            "unique_timestamps": 80,
            "label_counts": {"0": 1, "1": 1},
        },
    )
    raw_start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    raw_intervals = tuple(
        raw_start + timedelta(minutes=offset)
        for offset in (0, 5, 10, 20)
    )
    feature_intervals = tuple(
        raw_start + timedelta(minutes=offset)
        for offset in (0, 5, 10, 15)
    )

    async def _coverage(**kwargs):
        return {
            "raw": readiness_module._TableCoverage(  # pylint: disable=protected-access
                exists=True,
                intervals_by_symbol={
                    "BTC/USD": raw_intervals,
                    "ETH/USD": raw_intervals,
                },
            ),
            "feature": readiness_module._TableCoverage(  # pylint: disable=protected-access
                exists=True,
                intervals_by_symbol={
                    "BTC/USD": feature_intervals,
                    "ETH/USD": feature_intervals,
                },
            ),
        }

    monkeypatch.setattr(
        readiness_module,
        "Settings",
        SimpleNamespace(from_env=lambda: _fake_settings()),
    )
    monkeypatch.setattr(readiness_module, "load_training_dataset_preview", lambda config: dataset)
    monkeypatch.setattr(readiness_module, "_load_table_coverage_with_fallback", _coverage)
    monkeypatch.setattr(
        readiness_module,
        "_resolve_regime_distribution",
        lambda samples: (
            {"RANGE": 2},
            {"BTC/USD": {"RANGE": 1}, "ETH/USD": {"RANGE": 1}},
            None,
        ),
    )

    report = readiness_module.build_data_readiness_report(
        _config(),
        config_path=tmp_path / "training.m7.json",
    )
    artifact_dir = readiness_module.write_data_readiness_artifacts(
        report,
        artifact_root=tmp_path / "artifacts",
    )

    assert report.raw_rows_total == 8
    assert report.feature_rows_total == 8
    assert report.labeled_rows_total == 2
    assert report.ready_for_training is True
    assert report.symbol_summaries[0].raw_missing_interval_count == 1
    assert report.regime_distribution == {"RANGE": 2}
    assert (artifact_dir / "readiness_report.json").is_file()
    assert (artifact_dir / "symbol_coverage.csv").is_file()
    assert (artifact_dir / "gap_summary.csv").is_file()
    assert (artifact_dir / "summary.md").is_file()


def test_build_data_readiness_report_marks_insufficient_walk_forward_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The readiness gate should fail clearly when unique timestamps are still too low."""
    sample_time = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    dataset = SimpleNamespace(
        samples=(
            _sample(symbol="BTC/USD", as_of_time=sample_time, label=1, future_return_3=0.01),
        ),
        manifest={
            "loaded_rows": 40,
            "eligible_rows": 1,
            "unique_timestamps": 4,
            "label_counts": {"0": 0, "1": 1},
        },
    )

    async def _coverage(**kwargs):
        interval = sample_time - timedelta(minutes=5)
        intervals = (interval,)
        return {
            "raw": readiness_module._TableCoverage(  # pylint: disable=protected-access
                exists=True,
                intervals_by_symbol={"BTC/USD": intervals, "ETH/USD": intervals},
            ),
            "feature": readiness_module._TableCoverage(  # pylint: disable=protected-access
                exists=True,
                intervals_by_symbol={"BTC/USD": intervals, "ETH/USD": intervals},
            ),
        }

    monkeypatch.setattr(
        readiness_module,
        "Settings",
        SimpleNamespace(from_env=lambda: _fake_settings()),
    )
    monkeypatch.setattr(readiness_module, "load_training_dataset_preview", lambda config: dataset)
    monkeypatch.setattr(readiness_module, "_load_table_coverage_with_fallback", _coverage)
    monkeypatch.setattr(
        readiness_module,
        "_resolve_regime_distribution",
        lambda samples: ({}, {}, "No labeled rows are available yet for regime distribution."),
    )

    report = readiness_module.build_data_readiness_report(
        _config(),
        config_path=tmp_path / "training.m7.json",
    )

    assert report.ready_for_training is False
    assert "configured walk-forward timestamp requirement" in report.readiness_detail
