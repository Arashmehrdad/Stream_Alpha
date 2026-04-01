"""Shared helpers for deterministic M7 training-workflow tests."""

# pylint: disable=too-few-public-methods,too-many-arguments

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


AUTHORITATIVE_TEST_MODEL_NAME = "authoritative_candidate_fixture"
SECONDARY_TEST_MODEL_NAME = "secondary_candidate_fixture"


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
            "primary_metric": "mean_long_only_net_value_proxy",
            "max_directional_accuracy_regression": 0.01,
            "max_brier_score_worsening": 0.01,
        },
        "first_train_fraction": 0.5,
        "interval_column": "interval_begin",
        "label_horizon_candles": 3,
        "models": {},
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


# pylint: disable=too-many-locals
def write_run_dir(
    base_dir: Path,
    run_name: str,
    *,
    model_name: str = AUTHORITATIVE_TEST_MODEL_NAME,
    mean_long_only_net_value_proxy: float,
    directional_accuracy: float,
    brier_score: float,
    feature_columns: list[str] | None = None,
    expanded_feature_names: list[str] | None = None,
    protocol_overrides: dict[str, Any] | None = None,
    persistence_mean_long_only_net_value_proxy: float | None = None,
    dummy_mean_long_only_net_value_proxy: float | None = None,
    training_model_config: dict[str, Any] | None = None,
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
    secondary_model_name = (
        SECONDARY_TEST_MODEL_NAME
        if model_name != SECONDARY_TEST_MODEL_NAME
        else "tertiary_candidate_fixture"
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
        "models": {},
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
        "model_name,fold_index,directional_accuracy,brier_score,"
        "trade_count,trade_rate,mean_long_only_gross_value_proxy,"
        "mean_long_only_net_value_proxy\n"
        f"{model_name},0,{directional_accuracy},{brier_score},1,1.0,"
        f"{mean_long_only_net_value_proxy + 0.002},{mean_long_only_net_value_proxy}\n",
        encoding="utf-8",
    )
    (run_dir / "oof_predictions.csv").write_text(
        "model_name,row_id,prob_up,regime_label,long_trade_taken,"
        "long_only_gross_value_proxy,long_only_net_value_proxy\n"
        f"{model_name},BTC/USD|2026-03-19T00:00:00Z,0.6,TREND_UP,1,"
        f"{mean_long_only_net_value_proxy + 0.002},{mean_long_only_net_value_proxy}\n",
        encoding="utf-8",
    )
    persistence_metric = (
        mean_long_only_net_value_proxy - 0.0005
        if persistence_mean_long_only_net_value_proxy is None
        else persistence_mean_long_only_net_value_proxy
    )
    dummy_metric = (
        mean_long_only_net_value_proxy - 0.001
        if dummy_mean_long_only_net_value_proxy is None
        else dummy_mean_long_only_net_value_proxy
    )

    def _metrics_payload(
        metric_value: float,
        *,
        accuracy: float,
        brier: float,
    ) -> dict[str, Any]:
        return {
            "directional_accuracy": accuracy,
            "brier_score": brier,
            "trade_count": 1,
            "trade_rate": 1.0,
            "mean_long_only_gross_value_proxy": metric_value + 0.002,
            "mean_long_only_net_value_proxy": metric_value,
            "economics_by_regime": {
                "TREND_UP": {
                    "prediction_count": 1,
                    "trade_count": 1,
                    "trade_rate": 1.0,
                    "mean_long_only_gross_value_proxy": metric_value + 0.002,
                    "mean_long_only_net_value_proxy": metric_value,
                    "after_cost_positive": metric_value > 0.0,
                }
            },
        }

    summary = {
        "economics_contract": {
            "name": "LONG_ONLY_AFTER_COST_PROXY",
            "primary_metric": "mean_long_only_net_value_proxy",
        },
        "models": {
            model_name: _metrics_payload(
                mean_long_only_net_value_proxy,
                accuracy=directional_accuracy,
                brier=brier_score,
            ),
            secondary_model_name: _metrics_payload(
                mean_long_only_net_value_proxy - 0.001,
                accuracy=directional_accuracy - 0.02,
                brier=brier_score + 0.01,
            ),
            "persistence_3": _metrics_payload(
                persistence_metric,
                accuracy=max(directional_accuracy - 0.01, 0.0),
                brier=brier_score + 0.005,
            ),
            "dummy_most_frequent": _metrics_payload(
                dummy_metric,
                accuracy=max(directional_accuracy - 0.03, 0.0),
                brier=brier_score + 0.015,
            ),
        },
        "promotion_baselines": ["persistence_3", "dummy_most_frequent"],
        "winner": {
            "model_name": model_name,
            "selection_rule": {
                "primary": "mean_long_only_net_value_proxy",
                "tie_break_1": "directional_accuracy",
                "tie_break_2": "lower_brier_score",
            },
        },
        "acceptance": {
            "winner_after_cost_positive": mean_long_only_net_value_proxy > 0.0,
            "learned_models_positive_after_costs": (
                [model_name] if mean_long_only_net_value_proxy > 0.0 else []
            ),
            "learned_models_beating_persistence_after_costs": (
                [model_name]
                if mean_long_only_net_value_proxy > persistence_metric
                else []
            ),
            "learned_models_beating_dummy_after_costs": (
                [model_name]
                if mean_long_only_net_value_proxy > dummy_metric
                else []
            ),
            "learned_models_beating_all_baselines_after_costs": (
                [model_name]
                if (
                    mean_long_only_net_value_proxy > 0.0
                    and mean_long_only_net_value_proxy > persistence_metric
                    and mean_long_only_net_value_proxy > dummy_metric
                )
                else []
            ),
            "meets_acceptance_target": (
                mean_long_only_net_value_proxy > 0.0
                and mean_long_only_net_value_proxy > persistence_metric
                and mean_long_only_net_value_proxy > dummy_metric
            ),
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
            "training_model_config": training_model_config,
            "model": SerializableProbabilityModel(),
        },
        run_dir / "model.joblib",
    )
    return run_dir
