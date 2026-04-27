"""Tests for truthful long-only M3/M7 training economics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.training.dataset import (
    DatasetSample,
    SourceFeatureRow,
    TrainingConfig,
    load_training_config,
)
from app.training.pretrained_forecasters import (
    Chronos2Forecaster,
    Moirai1RBaseForecaster,
    TimesFm2Forecaster,
)
from app.training.service import (
    TrainingRegimeContext,
    _TrainingProgressRecorder,
    _build_acceptance_block,
    _build_model_factories,
    _build_prediction_records,
    _build_regime_economics,
    _build_summary_payload,
    _predict_for_model,
    run_training,
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


def _source_row(sample: DatasetSample) -> SourceFeatureRow:
    return SourceFeatureRow(
        row_id=sample.row_id,
        symbol=sample.symbol,
        interval_begin=sample.interval_begin,
        as_of_time=sample.as_of_time,
        close_price=sample.close_price,
        features=dict(sample.features),
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


def test_run_training_fails_early_when_readiness_gate_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training should stop at the shared readiness gate before loading the dataset."""
    monkeypatch.setattr("app.training.service.load_training_config", lambda path: _config())
    monkeypatch.setattr(
        "app.training.service._validate_authoritative_model_stack",
        lambda config: None,
    )
    monkeypatch.setattr(
        "app.training.service.assert_training_data_ready",
        lambda config, config_path=None: (_ for _ in ()).throw(
            ValueError(
                "feature_ohlc does not yet satisfy the configured "
                "walk-forward timestamp requirement (4/9)."
            )
        ),
    )
    monkeypatch.setattr(
        "app.training.service.load_training_dataset",
        lambda config: (_ for _ in ()).throw(AssertionError("dataset load should not run")),
    )

    with pytest.raises(ValueError, match="configured walk-forward timestamp requirement"):
        run_training(tmp_path / "training.m7.json")


def test_training_progress_recorder_writes_artifact_log_and_status(tmp_path: Path) -> None:
    """Long-running specialist progress should be inspectable outside the terminal."""
    recorder = _TrainingProgressRecorder(tmp_path)

    recorder.record(
        stage="setup",
        message="training artifact directory created",
        artifact_dir=str(tmp_path),
    )
    recorder.record_sequence_event(
        model_name="neuralforecast_nhits",
        fold_index=0,
        total_folds=5,
        payload={
            "event": "sequence_scoring_start",
            "row_count": 255,
            "batch_count": 2,
            "batch_size": 128,
        },
    )
    recorder.record_sequence_event(
        model_name="neuralforecast_nhits",
        fold_index=0,
        total_folds=5,
        payload={
            "event": "sequence_scoring_progress",
            "row_count": 255,
            "completed_rows": 128,
            "batch_count": 2,
            "completed_batches": 1,
            "progress": 0.5,
            "elapsed_seconds": 10.0,
            "eta_seconds": 12.0,
        },
    )
    recorder.record_sequence_event(
        model_name="neuralforecast_nhits",
        fold_index=0,
        total_folds=5,
        payload={
            "event": "backend_archive_dataset_fallback",
            "message": "NHITS backend archive fallback",
        },
    )

    log_text = (tmp_path / "progress.log").read_text(encoding="utf-8")
    status_payload = json.loads(
        (tmp_path / "progress_status.json").read_text(encoding="utf-8")
    )

    assert "training artifact directory created" in log_text
    assert "[sequence_scoring] fold 1/5 [neuralforecast_nhits] started" in log_text
    assert "[##########----------] 50.0%" in log_text
    assert "rows=128/255" in log_text
    assert "elapsed=00:10" in log_text
    assert "eta=00:12" in log_text
    assert "NHITS backend archive fallback" in log_text
    assert status_payload["event"] == "backend_archive_dataset_fallback"
    assert status_payload["fold_index"] == 1
    assert status_payload["total_folds"] == 5
    assert status_payload["model_name"] == "neuralforecast_nhits"


def test_build_model_factories_accepts_chronos2_generalist_builder() -> None:
    """The authoritative model stack should recognize the real Chronos-2 builder."""
    config = TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=("close_price", "realized_vol_12", "momentum_3"),
        label_horizon_candles=5,
        purge_gap_candles=3,
        test_folds=5,
        first_train_fraction=0.5,
        test_fraction=0.1,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m20",
        models={
            "chronos2_generalist": {
                "context_lookback_candles": 32,
                "device_map": "cpu",
            }
        },
    )

    model_factories = _build_model_factories(config)
    estimator = model_factories["chronos2_generalist"]()

    assert isinstance(estimator, Chronos2Forecaster)
    assert estimator.horizon_candles == 5
    assert estimator.context_lookback_candles == 32
    assert estimator.candidate_role == "GENERALIST"


def test_build_model_factories_accepts_timesfm_trend_builder() -> None:
    """The authoritative model stack should recognize the real TimesFM trend builder."""
    config = TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=("close_price", "realized_vol_12", "momentum_3"),
        label_horizon_candles=5,
        purge_gap_candles=3,
        test_folds=5,
        first_train_fraction=0.5,
        test_fraction=0.1,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m20",
        models={
            "timesfm_2_0_500m_pytorch_trend": {
                "context_lookback_candles": 64,
                "backend": "cpu",
            }
        },
    )

    model_factories = _build_model_factories(config)
    estimator = model_factories["timesfm_2_0_500m_pytorch_trend"]()

    assert isinstance(estimator, TimesFm2Forecaster)
    assert estimator.horizon_candles == 5
    assert estimator.context_lookback_candles == 64
    assert estimator.candidate_role == "TREND_SPECIALIST"


def test_build_model_factories_accepts_moirai_range_builder() -> None:
    """The authoritative model stack should recognize the real Moirai range builder."""
    config = TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=("close_price", "realized_vol_12", "momentum_3"),
        label_horizon_candles=5,
        purge_gap_candles=3,
        test_folds=5,
        first_train_fraction=0.5,
        test_fraction=0.1,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m20",
        models={
            "moirai_1_0_r_base_range": {
                "context_lookback_candles": 96,
                "map_location": "cpu",
            }
        },
    )

    model_factories = _build_model_factories(config)
    estimator = model_factories["moirai_1_0_r_base_range"]()

    assert isinstance(estimator, Moirai1RBaseForecaster)
    assert estimator.horizon_candles == 5
    assert estimator.context_lookback_candles == 96
    assert estimator.candidate_role == "RANGE_SPECIALIST"


def test_predict_for_model_uses_sequence_fit_and_single_probability_pass() -> None:
    """Sequence models should score once via probabilities and derive labels from that pass."""

    class _SequenceEstimator:
        def __init__(self) -> None:
            self.fitted_samples: list[DatasetSample] | None = None
            self.fitted_source_rows: list[SourceFeatureRow] | None = None
            self.predicted_source_rows: list[SourceFeatureRow] | None = None
            self.predict_samples_calls = 0
            self.predict_proba_samples_calls = 0

        def fit_samples(
            self,
            samples: list[DatasetSample],
            *,
            source_rows: list[SourceFeatureRow],
            dataset_export_root=None,
        ) -> None:
            del dataset_export_root
            self.fitted_samples = list(samples)
            self.fitted_source_rows = list(source_rows)

        def predict_samples(
            self,
            samples: list[DatasetSample],
            *,
            source_rows: list[SourceFeatureRow],
        ) -> list[int]:
            del samples, source_rows
            self.predict_samples_calls += 1
            raise AssertionError("sequence scoring should derive labels from predict_proba_samples")

        def predict_proba_samples(
            self,
            samples: list[DatasetSample],
            *,
            source_rows: list[SourceFeatureRow],
            progress_callback=None,
        ) -> list[list[float]]:
            self.predict_proba_samples_calls += 1
            self.predicted_source_rows = list(source_rows)
            return [[0.2, 0.8] for _ in samples]

    train_samples = [
        _sample(row_id="train-up", future_return_3=0.01, label=1),
        _sample(row_id="train-down", future_return_3=-0.01, label=0),
    ]
    test_samples = [_sample(row_id="test", future_return_3=0.02, label=1)]
    train_source_rows = [_source_row(sample) for sample in train_samples]
    evaluation_source_rows = [*train_source_rows, _source_row(test_samples[0])]
    estimator = _SequenceEstimator()

    predicted_labels, probabilities = _predict_for_model(
        model_name="neuralforecast_nhits",
        factory=lambda: estimator,
        train_samples=train_samples,
        test_samples=test_samples,
        train_source_rows=train_source_rows,
        evaluation_source_rows=evaluation_source_rows,
    )

    assert predicted_labels == [1]
    assert probabilities == [0.8]
    assert estimator.fitted_samples == train_samples
    assert estimator.fitted_source_rows == train_source_rows
    assert estimator.predicted_source_rows == evaluation_source_rows
    assert estimator.predict_samples_calls == 0
    assert estimator.predict_proba_samples_calls == 1


def test_predict_for_model_keeps_flat_feature_path_for_tabular_estimators() -> None:
    """Existing flat-row estimators should still use the original fit/predict_proba path."""

    class _FlatEstimator:
        def __init__(self) -> None:
            self.fit_rows: list[dict[str, object]] | None = None
            self.fit_labels: list[int] | None = None

        def fit(self, rows: list[dict[str, object]], labels: list[int]) -> None:
            self.fit_rows = list(rows)
            self.fit_labels = list(labels)

        def predict(self, rows: list[dict[str, object]]) -> list[int]:
            return [0 for _ in rows]

        def predict_proba(self, rows: list[dict[str, object]]) -> list[list[float]]:
            return [[0.75, 0.25] for _ in rows]

    train_samples = [
        _sample(row_id="train-up", future_return_3=0.01, label=1),
        _sample(row_id="train-down", future_return_3=-0.01, label=0),
    ]
    test_samples = [_sample(row_id="test", future_return_3=-0.02, label=0)]
    estimator = _FlatEstimator()

    predicted_labels, probabilities = _predict_for_model(
        model_name="autogluon_tabular",
        factory=lambda: estimator,
        train_samples=train_samples,
        test_samples=test_samples,
        train_source_rows=[_source_row(sample) for sample in train_samples],
        evaluation_source_rows=[_source_row(sample) for sample in train_samples]
        + [_source_row(test_samples[0])],
    )

    assert predicted_labels == [0]
    assert probabilities == [0.25]
    assert estimator.fit_rows == [sample.features for sample in train_samples]
    assert estimator.fit_labels == [1, 0]


def test_summary_records_winner_registry_metadata_and_candidate_artifacts() -> None:
    """Summary artifacts should carry winner metadata plus saved challenger artifact pointers."""
    regime_economics = {
        "TREND_UP": {
            "after_cost_positive": False,
            "mean_long_only_gross_value_proxy": 0.001,
            "mean_long_only_net_value_proxy": -0.0002,
            "prediction_count": 2,
            "trade_count": 1,
            "trade_rate": 0.5,
        }
    }
    summary = _build_summary_payload(
        config=_config(),
        dataset_manifest={"eligible_rows": 2, "unique_timestamps": 2},
        aggregate_summary={
            "neuralforecast_nhits": {
                "directional_accuracy": 0.56,
                "brier_score": 0.24,
                "mean_long_only_net_value_proxy": -0.0002,
                "economics_by_regime": regime_economics,
            },
            "persistence_3": {
                "directional_accuracy": 0.53,
                "brier_score": 0.26,
                "mean_long_only_net_value_proxy": -0.0003,
                "economics_by_regime": regime_economics,
            },
            "dummy_most_frequent": {
                "directional_accuracy": 0.50,
                "brier_score": 0.28,
                "mean_long_only_net_value_proxy": -0.0005,
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
        winner_name="neuralforecast_nhits",
        model_path=Path("artifacts/training/m20/model.joblib"),
        winner_training_config={
            "model_family": "NEURALFORECAST_NHITS",
            "dataset_mode": "local_files_partitioned",
        },
        winner_registry_metadata={
            "model_family": "NEURALFORECAST_NHITS",
            "candidate_role": "TREND_SPECIALIST",
            "scope_regimes": ["TREND_UP", "TREND_DOWN"],
        },
        candidate_artifacts={
            "neuralforecast_nhits": type("CandidateArtifact", (), {
                "model_path": Path(
                    "artifacts/training/m20/candidate_artifacts/"
                    "neuralforecast_nhits/model.joblib"
                ),
                "training_model_config": {"model_family": "NEURALFORECAST_NHITS"},
                "registry_metadata": {
                    "model_family": "NEURALFORECAST_NHITS",
                    "candidate_role": "TREND_SPECIALIST",
                    "scope_regimes": ["TREND_UP", "TREND_DOWN"],
                },
            })(),
        },
    )

    assert summary["winner"]["metadata"]["model_family"] == "NEURALFORECAST_NHITS"
    assert summary["winner"]["metadata"]["candidate_role"] == "TREND_SPECIALIST"
    assert summary["winner"]["training_config"]["dataset_mode"] == "local_files_partitioned"
    assert (
        summary["candidate_artifacts"]["neuralforecast_nhits"]["metadata"]["scope_regimes"]
        == ["TREND_UP", "TREND_DOWN"]
    )


def test_acceptance_block_shows_incumbent_info() -> None:
    """Acceptance metadata should expose the incumbent comparison basis."""
    verdicts = {
        "nhits": {
            "candidate_role": "TREND_SPECIALIST",
            "verdict": "accepted",
            "verdict_basis": "incumbent_comparison",
        }
    }
    block = _build_acceptance_block(
        winner_metrics={"mean_long_only_net_value_proxy": 0.005},
        learned_models_positive_after_costs=["nhits"],
        learned_models_beating_persistence=["nhits"],
        learned_models_beating_dummy=["nhits"],
        learned_models_beating_all_baselines=["nhits"],
        full_history_meets_acceptance=True,
        specialist_verdicts=verdicts,
        recent_window_meta={"window_days": 365},
        incumbent_model_version="m7-20260401T043003Z",
    )
    assert block["verdict_basis"] == "incumbent_comparison"
    assert block["incumbent_model_version"] == "m7-20260401T043003Z"
    assert block["meets_acceptance_target"] is True
