"""Tests for truthful long-only M3/M7 training economics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.training.dataset import DatasetSample, TrainingConfig, load_training_config
from app.training.service import (
    TrainingRegimeContext,
    _build_prediction_records,
    _build_regime_economics,
    _build_summary_payload,
)


def _sample(
    *,
    row_id: str,
    future_return_3: float,
    label: int,
) -> DatasetSample:
    observed_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return DatasetSample(
        row_id=row_id,
        symbol="BTC/USD",
        interval_begin=observed_at,
        as_of_time=observed_at,
        close_price=100.0,
        future_close_price=101.0,
        future_return_3=future_return_3,
        label=label,
        persistence_prediction=label,
        features={
            "symbol": "BTC/USD",
            "realized_vol_12": 0.02,
            "momentum_3": 0.01,
            "macd_line_12_26": 0.005,
        },
    )


def _config() -> TrainingConfig:
    return TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
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
        artifact_root="artifacts/training/m3",
        models={},
    )


def test_build_prediction_records_use_long_only_trade_or_flat_contract() -> None:
    """Predicted DOWN should stay flat instead of taking a synthetic short."""
    buy_sample = _sample(
        row_id="BTC/USD|2025-01-01T00:00:00Z",
        future_return_3=0.01,
        label=1,
    )
    hold_sample = _sample(
        row_id="BTC/USD|2025-01-01T00:05:00Z",
        future_return_3=-0.02,
        label=0,
    )

    predictions = _build_prediction_records(
        model_name="authoritative_candidate_fixture",
        fold_index=0,
        test_samples=[buy_sample, hold_sample],
        predicted_labels=[1, 0],
        probabilities=[0.7, 0.3],
        fee_rate=0.002,
        regime_labels_by_row_id={
            buy_sample.row_id: "TREND_UP",
            hold_sample.row_id: "RANGE",
        },
    )

    assert predictions[0].long_trade_taken == 1
    assert predictions[0].long_only_gross_value_proxy == 0.01
    assert predictions[0].long_only_net_value_proxy == 0.008
    assert predictions[1].long_trade_taken == 0
    assert predictions[1].long_only_gross_value_proxy == 0.0
    assert predictions[1].long_only_net_value_proxy == 0.0


def test_regime_economics_and_summary_do_not_overstate_negative_winner() -> None:
    """A negative after-cost winner must not still look economically accepted."""
    buy_sample = _sample(
        row_id="BTC/USD|2025-01-01T00:00:00Z",
        future_return_3=0.01,
        label=1,
    )
    hold_sample = _sample(
        row_id="BTC/USD|2025-01-01T00:05:00Z",
        future_return_3=-0.02,
        label=0,
    )
    predictions = _build_prediction_records(
        model_name="authoritative_candidate_fixture",
        fold_index=0,
        test_samples=[buy_sample, hold_sample],
        predicted_labels=[1, 0],
        probabilities=[0.7, 0.3],
        fee_rate=0.02,
        regime_labels_by_row_id={
            buy_sample.row_id: "TREND_UP",
            hold_sample.row_id: "HIGH_VOL",
        },
    )

    regime_economics = _build_regime_economics(predictions)
    assert regime_economics["TREND_UP"]["after_cost_positive"] is False
    assert regime_economics["HIGH_VOL"]["mean_long_only_net_value_proxy"] == 0.0

    summary = _build_summary_payload(
        config=_config(),
        dataset_manifest={"eligible_rows": 2, "unique_timestamps": 2},
        aggregate_summary={
            "authoritative_candidate_fixture": {
                "directional_accuracy": 0.55,
                "brier_score": 0.24,
                "mean_long_only_net_value_proxy": -0.0005,
                "economics_by_regime": regime_economics,
            },
            "secondary_candidate_fixture": {
                "directional_accuracy": 0.54,
                "brier_score": 0.25,
                "mean_long_only_net_value_proxy": -0.0008,
                "economics_by_regime": regime_economics,
            },
            "persistence_3": {
                "directional_accuracy": 0.53,
                "brier_score": 0.26,
                "mean_long_only_net_value_proxy": -0.0007,
                "economics_by_regime": regime_economics,
            },
            "dummy_most_frequent": {
                "directional_accuracy": 0.50,
                "brier_score": 0.28,
                "mean_long_only_net_value_proxy": -0.0012,
                "economics_by_regime": regime_economics,
            },
        },
        regime_context=TrainingRegimeContext(
            config_path="configs/regime.m8.json",
            high_vol_percentile=75.0,
            trend_abs_momentum_percentile=60.0,
            thresholds_by_symbol={},
            labels_by_row_id={},
        ),
        winner_name="authoritative_candidate_fixture",
        model_path=Path("artifacts/training/m3/model.joblib"),
    )

    assert summary["acceptance"]["winner_after_cost_positive"] is False
    assert summary["acceptance"]["meets_acceptance_target"] is False
    assert summary["winner"]["selection_rule"]["primary"] == "mean_long_only_net_value_proxy"


def test_training_config_rejects_legacy_archived_sklearn_models(
    tmp_path: Path,
) -> None:
    """Checked-in authoritative configs must reject legacy sklearn models."""
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": ["close_price"],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 2,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20.0,
                "artifact_root": "artifacts/training/m3",
                "models": {"logistic_regression": {"max_iter": 100}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy archived sklearn models"):
        load_training_config(config_path)


def test_training_config_accepts_autogluon_tabular_authoritative_model(
    tmp_path: Path,
) -> None:
    """Checked-in authoritative configs should allow the real AutoGluon path."""
    config_path = tmp_path / "training.json"
    config_path.write_text(
        json.dumps(
            {
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": ["close_price"],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 2,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20.0,
                "artifact_root": "artifacts/training/m3",
                "models": {
                    "autogluon_tabular": {
                        "presets": "medium_quality",
                        "time_limit": 30,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert tuple(config.models) == ("autogluon_tabular",)


def test_training_config_accepts_utf8_bom_json(
    tmp_path: Path,
) -> None:
    """PowerShell-saved UTF-8 BOM JSON should still load in the training path."""
    config_path = tmp_path / "training-with-bom.json"
    config_path.write_text(
        json.dumps(
            {
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": ["close_price"],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 2,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20.0,
                "artifact_root": "artifacts/training/m7",
                "models": {
                    "autogluon_tabular": {
                        "presets": "high_quality",
                        "time_limit": 900,
                    }
                },
            }
        ),
        encoding="utf-8-sig",
    )

    config = load_training_config(config_path)

    assert tuple(config.models) == ("autogluon_tabular",)


def test_summary_records_winner_autogluon_training_config() -> None:
    """Evaluation summary artifacts should expose the winner fit config for auditability."""
    regime_economics = {
        "TREND_UP": {
            "after_cost_positive": True,
            "mean_long_only_gross_value_proxy": 0.004,
            "mean_long_only_net_value_proxy": 0.002,
            "prediction_count": 2,
            "trade_count": 1,
            "trade_rate": 0.5,
        }
    }
    summary = _build_summary_payload(
        config=_config(),
        dataset_manifest={"eligible_rows": 2, "unique_timestamps": 2},
        aggregate_summary={
            "autogluon_tabular": {
                "directional_accuracy": 0.58,
                "brier_score": 0.22,
                "mean_long_only_net_value_proxy": 0.002,
                "economics_by_regime": regime_economics,
            },
            "persistence_3": {
                "directional_accuracy": 0.53,
                "brier_score": 0.26,
                "mean_long_only_net_value_proxy": 0.001,
                "economics_by_regime": regime_economics,
            },
            "dummy_most_frequent": {
                "directional_accuracy": 0.50,
                "brier_score": 0.28,
                "mean_long_only_net_value_proxy": 0.0005,
                "economics_by_regime": regime_economics,
            },
        },
        regime_context=TrainingRegimeContext(
            config_path="configs/regime.m8.json",
            high_vol_percentile=75.0,
            trend_abs_momentum_percentile=60.0,
            thresholds_by_symbol={},
            labels_by_row_id={},
        ),
        winner_name="autogluon_tabular",
        model_path=Path("artifacts/training/m3/model.joblib"),
        winner_training_config={
            "presets": "high",
            "time_limit": 900,
            "eval_metric": "log_loss",
            "hyperparameters": None,
            "fit_weighted_ensemble": True,
            "num_bag_folds": 5,
            "num_stack_levels": 1,
            "num_bag_sets": 1,
            "fold_fitting_strategy": "sequential_local",
            "dynamic_stacking": False,
            "calibrate_decision_threshold": False,
            "verbosity": 0,
        },
    )

    assert summary["winner"]["training_config"]["presets"] == "high"
    assert summary["winner"]["training_config"]["hyperparameters"] is None
    assert summary["winner"]["training_config"]["num_bag_sets"] == 1
    assert (
        summary["winner"]["training_config"]["fold_fitting_strategy"]
        == "sequential_local"
    )
    assert summary["winner"]["training_config"]["dynamic_stacking"] is False
    assert summary["winner"]["training_config"]["calibrate_decision_threshold"] is False
