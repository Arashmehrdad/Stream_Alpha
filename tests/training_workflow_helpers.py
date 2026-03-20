"""Shared helpers for deterministic M7 training-workflow tests."""

# pylint: disable=too-few-public-methods,too-many-arguments

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class SerializableProbabilityModel:
    """Tiny serializable classifier stub with binary probabilities."""

    def __init__(self, prob_up: float = 0.6) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        """Return a fixed binary probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


def write_workflow_config(config_path: Path) -> Path:
    """Write a minimal checked-in style M7 config for tests."""
    payload = {
        "artifact_root": "artifacts/training/m7",
        "categorical_feature_columns": ["symbol"],
        "close_column": "close_price",
        "comparison_policy": {
            "primary_metric": "mean_net_value_proxy",
            "max_directional_accuracy_regression": 0.01,
            "max_brier_score_worsening": 0.01,
        },
        "first_train_fraction": 0.5,
        "interval_column": "interval_begin",
        "label_horizon_candles": 3,
        "models": {
            "logistic_regression": {"max_iter": 100},
            "hist_gradient_boosting": {"max_iter": 10},
        },
        "numeric_feature_columns": ["close_price", "log_return_1"],
        "purge_gap_candles": 3,
        "round_trip_fee_bps": 20,
        "source_table": "feature_ohlc",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "test_folds": 5,
        "test_fraction": 0.1,
        "time_column": "as_of_time",
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def write_run_dir(
    base_dir: Path,
    run_name: str,
    *,
    model_name: str = "logistic_regression",
    mean_net_value_proxy: float,
    directional_accuracy: float,
    brier_score: float,
    feature_columns: list[str] | None = None,
    expanded_feature_names: list[str] | None = None,
    protocol_overrides: dict[str, Any] | None = None,
) -> Path:
    """Create a minimal but valid training artifact directory for tests."""
    feature_columns = (
        ["symbol", "close_price", "log_return_1"]
        if feature_columns is None
        else feature_columns
    )
    expanded_feature_names = (
        ["symbol=BTC/USD", "close_price", "log_return_1"]
        if expanded_feature_names is None
        else expanded_feature_names
    )
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "artifact_root": str(base_dir.parent),
        "categorical_feature_columns": ["symbol"],
        "close_column": "close_price",
        "first_train_fraction": 0.5,
        "interval_column": "interval_begin",
        "label_horizon_candles": 3,
        "models": {
            "logistic_regression": {"max_iter": 100},
            "hist_gradient_boosting": {"max_iter": 10},
        },
        "numeric_feature_columns": ["close_price", "log_return_1"],
        "purge_gap_candles": 3,
        "round_trip_fee_bps": 20,
        "source_table": "feature_ohlc",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "test_folds": 5,
        "test_fraction": 0.1,
        "time_column": "as_of_time",
    }
    if protocol_overrides:
        run_config.update(protocol_overrides)

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "loaded_rows": 300,
                "eligible_rows": 240,
                "unique_timestamps": 80,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "configured_feature_columns": feature_columns,
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": [
                    column for column in feature_columns if column != "symbol"
                ],
                "expanded_feature_names": expanded_feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "fold_metrics.csv").write_text(
        "model_name,fold_index,directional_accuracy,brier_score,mean_net_value_proxy\n"
        f"{model_name},0,{directional_accuracy},{brier_score},{mean_net_value_proxy}\n",
        encoding="utf-8",
    )
    (run_dir / "oof_predictions.csv").write_text(
        "model_name,row_id,prob_up\n"
        f"{model_name},BTC/USD|2026-03-19T00:00:00Z,0.6\n",
        encoding="utf-8",
    )
    summary = {
        "models": {
            model_name: {
                "directional_accuracy": directional_accuracy,
                "brier_score": brier_score,
                "mean_net_value_proxy": mean_net_value_proxy,
            },
            "hist_gradient_boosting": {
                "directional_accuracy": directional_accuracy - 0.02,
                "brier_score": brier_score + 0.01,
                "mean_net_value_proxy": mean_net_value_proxy - 0.001,
            },
        },
        "winner": {
            "model_name": model_name,
            "selection_rule": {
                "primary": "mean_net_value_proxy",
                "tie_break_1": "directional_accuracy",
                "tie_break_2": "lower_brier_score",
            },
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model_name": model_name,
            "trained_at": "2026-03-20T10:00:00Z",
            "feature_columns": feature_columns,
            "expanded_feature_names": expanded_feature_names,
            "model": SerializableProbabilityModel(),
        },
        run_dir / "model.joblib",
    )
    return run_dir
