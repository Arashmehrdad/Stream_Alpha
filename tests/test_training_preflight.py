"""Focused tests for M20 specialist training preflight checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.preflight_m20 import build_m20_preflight_report


def _write_m20_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "artifact_root": "artifacts/training/m20",
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": [
                    "close_price",
                    "realized_vol_12",
                    "momentum_3",
                    "macd_line_12_26",
                ],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 5,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20,
                "models": {
                    "neuralforecast_nhits": {
                        "batch_size": 1,
                        "candidate_role": "TREND_SPECIALIST",
                        "dataset_mode": "local_files_partitioned",
                        "input_size_candles": 96,
                        "max_steps": 10,
                        "model_kwargs": {
                            "accelerator": "auto",
                            "devices": 1,
                            "inference_windows_batch_size": 8,
                            "precision": "16-mixed",
                            "step_size": 128,
                            "valid_batch_size": 1,
                            "windows_batch_size": 8,
                        },
                    },
                    "neuralforecast_patchtst": {
                        "batch_size": 1,
                        "candidate_role": "RANGE_SPECIALIST",
                        "dataset_mode": "local_files_partitioned",
                        "input_size_candles": 96,
                        "max_steps": 10,
                        "model_kwargs": {
                            "accelerator": "auto",
                            "devices": 1,
                            "inference_windows_batch_size": 4,
                            "precision": "16-mixed",
                            "step_size": 128,
                            "valid_batch_size": 1,
                            "windows_batch_size": 4,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_m20_preflight_reports_missing_optional_runtime_deps_honestly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "training.m20.json"
    _write_m20_config(config_path)
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_optional_module",
        lambda module_name: {
            "installed": False,
            "version": None,
            "detail": f"missing {module_name}",
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_torch_runtime",
        lambda: {
            "installed": True,
            "version": "2.9.1+cpu",
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cuda_version": None,
            "detail": None,
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20.list_registry_entries",
        lambda registry_root=None: [],
    )

    report = build_m20_preflight_report(config_path=config_path)

    assert report["ready_for_manual_training"] is False
    assert report["preferred_execution_device"] == "cpu"
    assert report["source_table"] == "feature_ohlc"
    assert report["symbols"] == ["BTC/USD", "ETH/USD", "SOL/USD"]
    assert (
        report["model_preflight"]["neuralforecast_nhits"]["accelerator"] == "auto"
    )
    assert (
        report["model_preflight"]["neuralforecast_nhits"]["dataset_mode"]
        == "local_files_partitioned"
    )
    assert report["model_preflight"]["neuralforecast_nhits"]["batch_size"] == 1
    assert report["model_preflight"]["neuralforecast_nhits"]["windows_batch_size"] == 8
    assert report["model_preflight"]["neuralforecast_nhits"]["precision"] == "16-mixed"
    assert (
        report["model_preflight"]["neuralforecast_patchtst"]["devices"] == 1
    )
    assert report["model_preflight"]["neuralforecast_patchtst"]["step_size"] == 128
    assert any("lightning" in blocker for blocker in report["blockers"])
    assert any("neuralforecast" in blocker for blocker in report["blockers"])
    assert any("CUDA is not currently available" in warning for warning in report["warnings"])
    assert any("No real registry-backed NHITS or PatchTST" in warning for warning in report["warnings"])


def test_m20_preflight_require_gpu_adds_explicit_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "training.m20.json"
    _write_m20_config(config_path)
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_optional_module",
        lambda module_name: {
            "installed": True,
            "version": "ok",
            "detail": None,
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_torch_runtime",
        lambda: {
            "installed": True,
            "version": "2.9.1+cpu",
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cuda_version": None,
            "detail": None,
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20.list_registry_entries",
        lambda registry_root=None: [],
    )

    report = build_m20_preflight_report(config_path=config_path, require_gpu=True)

    assert report["ready_for_manual_training"] is False
    assert any("GPU was required" in blocker for blocker in report["blockers"])


def test_m20_preflight_reports_existing_specialist_registry_entries_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "training.m20.json"
    _write_m20_config(config_path)
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_optional_module",
        lambda module_name: {
            "installed": True,
            "version": "ok",
            "detail": None,
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20._probe_torch_runtime",
        lambda: {
            "installed": True,
            "version": "2.9.1+cu128",
            "cuda_available": True,
            "device_count": 1,
            "device_names": ["NVIDIA Test GPU"],
            "cuda_version": "12.8",
            "detail": None,
        },
    )
    monkeypatch.setattr(
        "app.training.preflight_m20.list_registry_entries",
        lambda registry_root=None: [
            {
                "model_version": "m20-20260403T120000Z",
                "model_name": "neuralforecast_nhits",
                "metadata": {
                    "model_family": "NEURALFORECAST_NHITS",
                    "candidate_role": "TREND_SPECIALIST",
                },
            }
        ],
    )

    report = build_m20_preflight_report(config_path=config_path)

    assert report["ready_for_manual_training"] is True
    assert report["preferred_execution_device"] == "gpu"
    assert report["registry"]["specialist_entry_count"] == 1
    assert report["registry"]["specialist_entries"][0]["model_family"] == "NEURALFORECAST_NHITS"
