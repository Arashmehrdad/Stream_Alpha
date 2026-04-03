"""Focused tests for local M7 training readiness checks."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.training import readiness as readiness_module
from app.training.data_readiness import DataReadinessReport, SymbolReadinessSnapshot
from app.training.dataset import TrainingConfig


def _config() -> TrainingConfig:
    return TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD", "ETH/USD"),
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
        artifact_root="artifacts/training/m7",
        models={"autogluon_tabular": {"time_limit": 900}},
    )


def _readiness_report(*, ready: bool) -> DataReadinessReport:
    generated_at = datetime(2026, 4, 2, tzinfo=timezone.utc)
    return DataReadinessReport(
        config_path="D:/Github/Stream_Alpha/configs/training.m7.json",
        generated_at=generated_at,
        artifact_root="artifacts/training/m7",
        source_table="feature_ohlc",
        raw_table="raw_ohlc",
        feature_table="feature_ohlc",
        source_exchange="kraken",
        interval_minutes=5,
        postgres_reachable=True,
        postgres_error=None,
        raw_table_exists=True,
        feature_table_exists=True,
        raw_rows_total=600,
        feature_rows_total=400,
        labeled_rows_total=240 if ready else 12,
        unique_timestamps=80 if ready else 4,
        required_unique_timestamps=9,
        ready_for_training=ready,
        readiness_detail=(
            "feature_ohlc satisfies the configured walk-forward timestamp requirement"
            if ready
            else "feature_ohlc does not yet satisfy the configured walk-forward timestamp requirement (4/9)."
        ),
        earliest_usable_timestamp=generated_at,
        latest_usable_timestamp=generated_at,
        overall_positive_label_rate=0.5,
        label_counts={"0": 120, "1": 120},
        regime_distribution_available=True,
        regime_distribution_detail=None,
        regime_distribution={"RANGE": 240},
        opportunity_density={},
        warnings=tuple(),
        symbol_summaries=(
            SymbolReadinessSnapshot(
                symbol="BTC/USD",
                raw_row_count=200,
                feature_row_count=200,
                labeled_row_count=120,
                raw_earliest_interval_begin=generated_at,
                raw_latest_interval_begin=generated_at,
                feature_earliest_interval_begin=generated_at,
                feature_latest_interval_begin=generated_at,
                labeled_earliest_as_of_time=generated_at,
                labeled_latest_as_of_time=generated_at,
                positive_label_rate=0.5,
                raw_missing_interval_count=0,
                feature_missing_interval_count=0,
                raw_gap_windows=tuple(),
                feature_gap_windows=tuple(),
                regime_distribution={"RANGE": 120},
            ),
        ),
    )


def test_build_training_readiness_report_marks_ready_from_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should report ready when the source probe and dataset manifest are sufficient."""

    monkeypatch.setattr(readiness_module, "load_training_config", lambda path: _config())
    monkeypatch.setattr(readiness_module, "_resolve_autogluon_version", lambda: "1.4.0")
    monkeypatch.setattr(
        readiness_module,
        "_resolve_fastai_status",
        lambda: readiness_module._OptionalBreadthStatus(  # pylint: disable=protected-access
            installed=True,
            version="2.7.18",
            usable=True,
            detail="optional breadth available",
        ),
    )
    monkeypatch.setattr(
        readiness_module,
        "build_data_readiness_report",
        lambda config, config_path=None: _readiness_report(ready=True),
    )

    report = readiness_module.build_training_readiness_report(
        tmp_path / "training.m7.json"
    )

    assert report.config_ok is True
    assert report.autogluon_version == "1.4.0"
    assert report.fastai_installed is True
    assert report.fastai_version == "2.7.18"
    assert report.fastai_usable is True
    assert report.fastai_detail == "optional breadth available"
    assert report.postgres_reachable is True
    assert report.feature_table_exists is True
    assert report.row_count == 400
    assert report.unique_timestamps == 80
    assert report.ready_for_training is True


def test_build_training_readiness_report_surfaces_unreachable_postgres(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should stay honest when PostgreSQL cannot be reached."""

    monkeypatch.setattr(readiness_module, "load_training_config", lambda path: _config())
    monkeypatch.setattr(readiness_module, "_resolve_autogluon_version", lambda: "1.4.0")
    monkeypatch.setattr(
        readiness_module,
        "_resolve_fastai_status",
        lambda: readiness_module._OptionalBreadthStatus(  # pylint: disable=protected-access
            installed=False,
            version=None,
            usable=False,
            detail="missing optional breadth only, not a blocker",
        ),
    )
    monkeypatch.setattr(
        readiness_module,
        "build_data_readiness_report",
        lambda config, config_path=None: replace(
            _readiness_report(ready=False),
            postgres_reachable=False,
            postgres_error="connection refused",
            feature_table_exists=None,
            feature_rows_total=0,
            labeled_rows_total=0,
            unique_timestamps=0,
            ready_for_training=False,
            readiness_detail="PostgreSQL is not reachable for training readiness checks",
        ),
    )

    report = readiness_module.build_training_readiness_report(
        tmp_path / "training.m7.json"
    )

    assert report.postgres_reachable is False
    assert report.postgres_error == "connection refused"
    assert report.fastai_installed is False
    assert report.fastai_version is None
    assert report.fastai_usable is False
    assert report.fastai_detail == "missing optional breadth only, not a blocker"
    assert report.ready_for_training is False


def test_resolve_fastai_status_marks_installed_but_unusable_when_ipython_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should surface the real FastAI blocker instead of only checking installation."""

    def _version(name: str) -> str | None:
        if name == "fastai":
            return "2.8.7"
        if name == "IPython":
            return None
        return None

    monkeypatch.setattr(readiness_module, "_resolve_package_version", _version)

    def _fail_fastai_import() -> None:
        import_error = ImportError("Import fastai failed.")
        import_error.__cause__ = ModuleNotFoundError("No module named 'IPython'")
        import_error.__cause__.name = "IPython"  # type: ignore[attr-defined]
        raise import_error

    monkeypatch.setitem(
        sys.modules,
        "autogluon.common.utils.try_import",
        SimpleNamespace(try_import_fastai=_fail_fastai_import),
    )

    status = readiness_module._resolve_fastai_status()  # pylint: disable=protected-access

    assert status.installed is True
    assert status.version == "2.8.7"
    assert status.usable is False
    assert status.detail == (
        "installed but unusable for AutoGluon because IPython is missing"
    )
