"""Focused tests for M20 specialist prediction export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_specialist_prediction_export import (
    export_m20_specialist_prediction_records,
    export_existing_m20_specialist_predictions,
)
from app.training.service import run_training

# pylint: disable=missing-function-docstring


def _write_oof(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model_name": "neuralforecast_nhits",
            "fold_index": "4",
            "row_id": "BTC/USD|2025-01-01T00:00:00Z",
            "symbol": "BTC/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "1",
            "y_pred": "1",
            "prob_up": "0.8",
            "confidence": "0.6",
            "regime_label": "TREND_UP",
            "future_return_3": "0.1",
            "long_only_net_value_proxy": "0.09",
        },
        {
            "model_name": "neuralforecast_patchtst",
            "fold_index": "4",
            "row_id": "ETH/USD|2025-01-01T00:00:00Z",
            "symbol": "ETH/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "0",
            "y_pred": "1",
            "prob_up": "0.7",
            "confidence": "0.4",
            "regime_label": "RANGE",
            "future_return_3": "-0.1",
            "long_only_net_value_proxy": "-0.12",
        },
        {
            "model_name": "persistence_3",
            "fold_index": "4",
            "row_id": "SOL/USD|2025-01-01T00:00:00Z",
            "symbol": "SOL/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": "0",
            "y_pred": "0",
            "prob_up": "0.2",
            "confidence": "0.8",
            "regime_label": "RANGE",
            "future_return_3": "0.0",
            "long_only_net_value_proxy": "0.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_export_writes_per_specialist_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    result = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )

    assert result["exported_row_count"] == 2
    assert Path(
        result["output_files"]["predictions_neuralforecast_nhits_oof_csv"]
    ).exists()
    assert Path(
        result["output_files"]["predictions_neuralforecast_patchtst_oof_csv"]
    ).exists()


def test_export_quarantines_future_and_net_columns(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    result = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    nhits_rows = _read_csv(
        Path(result["output_files"]["predictions_neuralforecast_nhits_oof_csv"])
    )
    audit_rows = _read_csv(Path(result["output_files"]["schema_audit_csv"]))
    audit_by_column = {row["column"]: row for row in audit_rows}

    assert "future_return_3" not in nhits_rows[0]
    assert "long_only_net_value_proxy" not in nhits_rows[0]
    assert audit_by_column["future_return_3"]["exported"] == "False"
    assert "quarantined" in audit_by_column["long_only_net_value_proxy"]["reason"]


def test_export_is_research_only_and_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "base"
    previous = tmp_path / "previous"
    _write_oof(previous / "oof_predictions.csv")

    first = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    second = export_existing_m20_specialist_predictions(
        base_run_dir=base,
        previous_run_dir=previous,
    )
    manifest = json.loads(
        Path(second["output_files"]["manifest_json"]).read_text(encoding="utf-8")
    )

    assert first["output_files"] == second["output_files"]
    assert "NO_RUNTIME_EFFECT" in manifest["honesty_flags"]
    assert manifest["promotion_status"] == "NOT_PROMOTABLE"


def test_export_score_only_records_writes_confirmation_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        {
            "model_name": "neuralforecast_patchtst",
            "fold_index": 4,
            "row_id": "BTC/USD|2025-01-01T00:00:00Z",
            "symbol": "BTC/USD",
            "interval_begin": "2025-01-01T00:00:00Z",
            "as_of_time": "2025-01-01T00:05:00Z",
            "y_true": 1,
            "y_pred": 1,
            "prob_up": 0.8,
            "confidence": 0.6,
            "regime_label": "RANGE",
            "future_return_3": 0.1,
            "long_only_net_value_proxy": 0.09,
        }
    ]

    result = export_m20_specialist_prediction_records(
        run_dir=run_dir,
        prediction_rows=rows,
        prediction_source="score_only_confirmation",
        confirmation_window={
            "candidate_run_id": "confirm_run",
            "confirmation_window_start": "2024-01-01T00:00:00Z",
            "confirmation_window_end": "2025-01-01T00:00:00Z",
            "confirmation_tag": "confirm",
        },
    )
    prediction_path = Path(
        result["output_files"][
            "predictions_neuralforecast_patchtst_score_only_confirmation_csv"
        ]
    )
    exported_rows = _read_csv(prediction_path)

    assert result["exported_row_count"] == 1
    assert exported_rows[0]["candidate_id"] == "confirm_run:neuralforecast_patchtst"
    assert exported_rows[0]["confirmation_tag"] == "confirm"
    assert "future_return_3" not in exported_rows[0]


def test_export_specialist_predictions_requires_score_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --score-only"):
        run_training(
            tmp_path / "missing.json",
            export_specialist_predictions_only=True,
        )


def test_export_specialist_only_scores_only_specialist_models_and_returns_early(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class DummyConfig:
        def __init__(self) -> None:
            self.models = {
                "neuralforecast_nhits": {},
                "neuralforecast_patchtst": {},
            }
            self.recent_scoring_window_days = None
            self.export_training_frame = False
            self.label_horizon_candles = 3
            self.source_table = "feature_ohlc"
            self.symbols = ["BTC/USD"]
            self.first_train_fraction = 0.5
            self.test_fraction = 0.1
            self.test_folds = 1
            self.purge_gap_candles = 3
            self.round_trip_fee_rate = 0.0
            self.time_column = "as_of_time"
            self.categorical_feature_columns = []
            self.numeric_feature_columns = []
            self.artifact_root = "artifacts/training/m20"
            self.artifact_root = "artifacts/training/m20"

        def to_dict(self) -> dict[str, object]:
            return {
                "artifact_root": "artifacts/training/m20",
                "source_table": self.source_table,
                "symbols": self.symbols,
                "first_train_fraction": self.first_train_fraction,
                "test_fraction": self.test_fraction,
                "test_folds": self.test_folds,
                "purge_gap_candles": self.purge_gap_candles,
                "recent_scoring_window_days": self.recent_scoring_window_days,
                "export_training_frame": self.export_training_frame,
                "label_horizon_candles": self.label_horizon_candles,
                "round_trip_fee_rate": self.round_trip_fee_rate,
                "time_column": self.time_column,
                "models": self.models,
                "categorical_feature_columns": self.categorical_feature_columns,
                "numeric_feature_columns": self.numeric_feature_columns,
            }

    config = DummyConfig()
    dataset = type(
        "Dataset",
        (),
        {
            "source_rows": [],
            "samples": [],
            "manifest": {"unique_timestamps": 0},
            "source_schema": [],
            "feature_columns": [],
            "categorical_feature_columns": [],
            "numeric_feature_columns": [],
            "frequency_minutes": 1,
            "timestamps": [],
        },
    )()

    fake_fold = type("Fold", (), {"fold_index": 0})()
    captured_models: list[str] = []

    def fake_load_config(path: Path) -> DummyConfig:
        return config

    def fake_load_dataset(config_arg: object, parquet_dir: Path | None = None) -> object:
        return dataset

    def fake_partition_samples(samples, fold):
        return [], []

    def fake_partition_source_rows(source_rows, fold, horizon_candles, frequency_minutes):
        return [], []

    def fake_build_walk_forward_splits(timestamps, first_train_fraction, test_fraction, test_folds, purge_gap_candles):
        return [fake_fold]

    def fake_evaluate_fold(*args, **kwargs):
        captured_models.extend(kwargs["model_factories"].keys())
        return [], []

    def fake_export_records(*args, **kwargs):
        return {"exported_row_count": 0}

    monkeypatch.setattr("app.training.service.load_training_config", fake_load_config)
    monkeypatch.setattr("app.training.service.assert_training_data_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.service.load_training_dataset", fake_load_dataset)
    monkeypatch.setattr("app.training.service._validate_split_readiness", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.service._build_training_regime_context", lambda *args, **kwargs: type("Context", (), {"labels_by_row_id": {}})())
    monkeypatch.setattr("app.training.service.build_walk_forward_splits", fake_build_walk_forward_splits)
    monkeypatch.setattr("app.training.service._partition_samples", fake_partition_samples)
    monkeypatch.setattr("app.training.service._partition_source_rows", fake_partition_source_rows)
    monkeypatch.setattr("app.training.service._evaluate_fold", fake_evaluate_fold)
    monkeypatch.setattr("app.training.service.export_m20_specialist_prediction_records", fake_export_records)
    monkeypatch.setattr("app.training.service._save_fold_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.service._write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.service._write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.service._load_full_fit_models", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_load_full_fit_models should not be called")))
    monkeypatch.setattr("app.training.service._build_aggregate_summary", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_build_aggregate_summary should not be called")))
    monkeypatch.setattr("app.training.service._select_winner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_select_winner should not be called")))

    run_training(
        tmp_path / "config.json",
        score_only_dir=tmp_path / "fitted_models",
        export_specialist_predictions_only=True,
    )

    assert set(captured_models) == {
        "neuralforecast_nhits",
        "neuralforecast_patchtst",
    }
