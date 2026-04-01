"""Focused tests for local M7 training readiness checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.common.config import PostgresSettings
from app.training import readiness as readiness_module
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


def _fake_settings():
    return SimpleNamespace(
        postgres=PostgresSettings(
            host="127.0.0.1",
            port=5432,
            database="streamalpha",
            user="streamalpha",
            password="change-me-local-only",
        )
    )


def test_build_training_readiness_report_marks_ready_from_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should report ready when the source probe and dataset manifest are sufficient."""

    async def _probe(**kwargs):
        return readiness_module._TrainingSourceProbe(  # pylint: disable=protected-access
            postgres_reachable=True,
            postgres_error=None,
            feature_table_exists=True,
            row_count=400,
        )

    monkeypatch.setattr(readiness_module, "load_training_config", lambda path: _config())
    monkeypatch.setattr(readiness_module, "_resolve_autogluon_version", lambda: "1.4.0")
    monkeypatch.setattr(
        readiness_module,
        "Settings",
        SimpleNamespace(from_env=lambda: _fake_settings()),
    )
    monkeypatch.setattr(
        readiness_module,
        "_probe_training_source_with_fallback",
        _probe,
    )
    monkeypatch.setattr(
        readiness_module,
        "load_training_dataset",
        lambda config: SimpleNamespace(
            manifest={"eligible_rows": 240, "unique_timestamps": 80}
        ),
    )

    report = readiness_module.build_training_readiness_report(
        tmp_path / "training.m7.json"
    )

    assert report.config_ok is True
    assert report.autogluon_version == "1.4.0"
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

    async def _probe(**kwargs):
        return readiness_module._TrainingSourceProbe(  # pylint: disable=protected-access
            postgres_reachable=False,
            postgres_error="connection refused",
            feature_table_exists=None,
            row_count=None,
        )

    monkeypatch.setattr(readiness_module, "load_training_config", lambda path: _config())
    monkeypatch.setattr(readiness_module, "_resolve_autogluon_version", lambda: "1.4.0")
    monkeypatch.setattr(
        readiness_module,
        "Settings",
        SimpleNamespace(from_env=lambda: _fake_settings()),
    )
    monkeypatch.setattr(
        readiness_module,
        "_probe_training_source_with_fallback",
        _probe,
    )

    report = readiness_module.build_training_readiness_report(
        tmp_path / "training.m7.json"
    )

    assert report.postgres_reachable is False
    assert report.postgres_error == "connection refused"
    assert report.ready_for_training is False
