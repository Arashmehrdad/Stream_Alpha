"""Offline training orchestration for Stream Alpha M3."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.regime.config import load_regime_config
from app.regime.dataset import RegimeSourceRow
from app.regime.service import (
    REGIME_LABELS,
    SymbolThresholds,
    classify_row,
    compute_percentile,
)
from app.training.baselines import PersistenceBaseline, build_dummy_classifier
from app.training.autogluon import build_autogluon_tabular_classifier
from app.training.dataset import (
    DatasetSample,
    LEGACY_ARCHIVED_MODEL_NAMES,
    TrainingConfig,
    load_training_config,
    load_training_dataset,
)
from app.training.splits import WalkForwardFold, build_walk_forward_splits
from app.training.splits import minimum_required_unique_timestamps


_REGIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "regime.m8.json"
_REQUIRED_PROMOTION_BASELINES = ("persistence_3", "dummy_most_frequent")
_AUTHORITATIVE_MODEL_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "autogluon_tabular": build_autogluon_tabular_classifier,
}


@dataclass(frozen=True, slots=True)
class PredictionRecord:  # pylint: disable=too-many-instance-attributes
    """One out-of-fold prediction emitted by a baseline or learned model."""

    model_name: str
    fold_index: int
    row_id: str
    symbol: str
    interval_begin: str
    as_of_time: str
    y_true: int
    y_pred: int
    prob_up: float
    confidence: float
    regime_label: str
    long_trade_taken: int
    future_return_3: float
    long_only_gross_value_proxy: float
    long_only_net_value_proxy: float

    def to_csv_row(self) -> dict[str, Any]:
        """Return a CSV-friendly representation for artifact persistence."""
        return {
            "model_name": self.model_name,
            "fold_index": self.fold_index,
            "row_id": self.row_id,
            "symbol": self.symbol,
            "interval_begin": self.interval_begin,
            "as_of_time": self.as_of_time,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "prob_up": self.prob_up,
            "confidence": self.confidence,
            "regime_label": self.regime_label,
            "long_trade_taken": self.long_trade_taken,
            "future_return_3": self.future_return_3,
            "long_only_gross_value_proxy": self.long_only_gross_value_proxy,
            "long_only_net_value_proxy": self.long_only_net_value_proxy,
        }


@dataclass(frozen=True, slots=True)
class TrainingRegimeContext:
    """Deterministic M8-style regime labels used for training economics slices."""

    config_path: str
    high_vol_percentile: float
    trend_abs_momentum_percentile: float
    thresholds_by_symbol: dict[str, SymbolThresholds]
    labels_by_row_id: dict[str, str]


def run_training(config_path: Path) -> Path:  # pylint: disable=too-many-locals
    """Run the full offline M3 training flow and save artifacts to disk."""
    config = load_training_config(config_path)
    _validate_authoritative_model_stack(config)
    dataset = load_training_dataset(config)
    _validate_split_readiness(dataset, config)
    regime_context = _build_training_regime_context(dataset.samples, config)
    folds = build_walk_forward_splits(
        dataset.timestamps,
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )
    artifact_dir = _create_artifact_dir(config)
    _write_json(artifact_dir / "run_config.json", config.to_dict())
    _write_json(
        artifact_dir / "dataset_manifest.json",
        {
            **dataset.manifest,
            "source_schema": list(dataset.source_schema),
            "feature_columns": list(dataset.feature_columns),
        },
    )

    model_factories = _build_model_factories(config)
    fold_metric_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[PredictionRecord] = []

    for fold in folds:
        train_samples, test_samples = _partition_samples(dataset.samples, fold)
        fold_metrics, fold_predictions = _evaluate_fold(
            train_samples=train_samples,
            test_samples=test_samples,
            fold=fold,
            model_factories=model_factories,
            fee_rate=config.round_trip_fee_rate,
            regime_labels_by_row_id=regime_context.labels_by_row_id,
        )
        fold_metric_rows.extend(fold_metrics)
        all_prediction_rows.extend(fold_predictions)

    _write_csv(artifact_dir / "fold_metrics.csv", fold_metric_rows)
    _write_csv(
        artifact_dir / "oof_predictions.csv",
        [record.to_csv_row() for record in all_prediction_rows],
    )

    aggregate_summary = _build_aggregate_summary(
        all_prediction_rows=all_prediction_rows,
        fee_rate=config.round_trip_fee_rate,
    )
    winner_name = _select_winner(aggregate_summary)
    saved_model, expanded_features = _fit_winner_on_full_dataset(
        winner_name=winner_name,
        model_factories=model_factories,
        samples=dataset.samples,
    )
    model_path = artifact_dir / "model.joblib"
    _save_model_artifact(
        model_path=model_path,
        model_name=winner_name,
        fitted_model=saved_model,
        feature_columns=dataset.feature_columns,
        expanded_feature_names=expanded_features,
    )
    _write_json(
        artifact_dir / "feature_columns.json",
        {
            "configured_feature_columns": list(dataset.feature_columns),
            "categorical_feature_columns": list(dataset.categorical_feature_columns),
            "numeric_feature_columns": list(dataset.numeric_feature_columns),
            "expanded_feature_names": expanded_features,
        },
    )
    summary = _build_summary_payload(
        config=config,
        dataset_manifest=dataset.manifest,
        aggregate_summary=aggregate_summary,
        regime_context=regime_context,
        winner_name=winner_name,
        model_path=model_path,
    )
    _write_json(artifact_dir / "summary.json", summary)
    return artifact_dir


def _validate_split_readiness(dataset: Any, config: TrainingConfig) -> None:
    """Fail early with a clear message when live data is not yet sufficient."""
    actual_unique_timestamps = int(dataset.manifest["unique_timestamps"])
    required_unique_timestamps = minimum_required_unique_timestamps(
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )
    if actual_unique_timestamps >= required_unique_timestamps:
        return
    raise ValueError(
        "Not enough unique eligible timestamps for the configured walk-forward split. "
        f"Required at least {required_unique_timestamps}, found {actual_unique_timestamps}. "
        f"Eligible labeled rows: {dataset.manifest['eligible_rows']}."
    )


def _create_artifact_dir(config: TrainingConfig) -> Path:
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _build_model_factories(
    config: TrainingConfig,
) -> dict[str, Callable[[], Any]]:
    model_factories: dict[str, Callable[[], Any]] = {
        "persistence_3": _build_persistence_baseline,
        "dummy_most_frequent": build_dummy_classifier,
    }
    for model_name, model_config in config.models.items():
        builder = _AUTHORITATIVE_MODEL_BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(
                "No authoritative training builder exists yet for configured model(s): "
                f"{sorted(config.models)}"
            )
        model_factories[model_name] = (
            lambda builder=builder, model_config=dict(model_config): builder(model_config)
        )
    return model_factories


def _build_persistence_baseline() -> PersistenceBaseline:
    return PersistenceBaseline()


def _validate_authoritative_model_stack(config: TrainingConfig) -> None:
    configured_models = tuple(sorted(config.models))
    if not configured_models:
        raise ValueError(
            "No active authoritative trainable models are configured. "
            "Legacy sklearn models have been removed from the main training path "
            "and the intended primary stack is not implemented yet."
        )
    legacy_models = [
        model_name
        for model_name in configured_models
        if model_name in LEGACY_ARCHIVED_MODEL_NAMES
    ]
    if legacy_models:
        raise ValueError(
            "Legacy archived sklearn models are no longer allowed in the "
            f"authoritative training path: {legacy_models}"
        )
    unsupported_models = [
        model_name
        for model_name in configured_models
        if model_name not in _AUTHORITATIVE_MODEL_BUILDERS
    ]
    if unsupported_models:
        raise ValueError(
            "No authoritative training builder exists yet for configured model(s): "
            f"{unsupported_models}"
        )


def _partition_samples(
    samples: tuple[DatasetSample, ...],
    fold: WalkForwardFold,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    train_times = set(fold.train_timestamps)
    test_times = set(fold.test_timestamps)
    train_samples = [sample for sample in samples if sample.as_of_time in train_times]
    test_samples = [sample for sample in samples if sample.as_of_time in test_times]
    if not train_samples:
        raise ValueError(f"Fold {fold.fold_index} has no training samples")
    if not test_samples:
        raise ValueError(f"Fold {fold.fold_index} has no test samples")
    return train_samples, test_samples


# pylint: disable=too-many-arguments
def _evaluate_fold(
    *,
    train_samples: list[DatasetSample],
    test_samples: list[DatasetSample],
    fold: WalkForwardFold,
    model_factories: dict[str, Callable[[], Any]],
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
) -> tuple[list[dict[str, Any]], list[PredictionRecord]]:
    # The explicit inputs keep the fold evaluation contract inspectable for M3/M7.
    fold_metrics: list[dict[str, Any]] = []
    prediction_rows: list[PredictionRecord] = []
    for model_name, factory in model_factories.items():
        predicted_labels, probabilities = _predict_for_model(
            model_name=model_name,
            factory=factory,
            train_samples=train_samples,
            test_samples=test_samples,
        )
        model_predictions = _build_prediction_records(
            model_name=model_name,
            fold_index=fold.fold_index,
            test_samples=test_samples,
            predicted_labels=predicted_labels,
            probabilities=probabilities,
            fee_rate=fee_rate,
            regime_labels_by_row_id=regime_labels_by_row_id,
        )
        metrics = _compute_metrics(model_predictions)
        fold_metrics.append(
            {
                "model_name": model_name,
                "fold_index": fold.fold_index,
                "train_rows": len(train_samples),
                "test_rows": len(test_samples),
                **metrics["metrics"],
            }
        )
        prediction_rows.extend(model_predictions)
    return fold_metrics, prediction_rows


def _predict_for_model(
    *,
    model_name: str,
    factory: Callable[[], Any],
    train_samples: list[DatasetSample],
    test_samples: list[DatasetSample],
) -> tuple[list[int], list[float]]:
    if model_name == "persistence_3":
        baseline = factory().fit(train_samples)
        labels = baseline.predict(test_samples)
        probabilities = [row[1] for row in baseline.predict_proba(test_samples)]
        return labels, probabilities

    train_labels = [sample.label for sample in train_samples]
    if model_name not in _REQUIRED_PROMOTION_BASELINES and len(
        set(train_labels)
    ) < 2:
        constant_label = int(train_labels[0])
        return [constant_label] * len(test_samples), [float(constant_label)] * len(test_samples)

    estimator = factory()
    estimator.fit([sample.features for sample in train_samples], train_labels)
    test_features = [sample.features for sample in test_samples]
    predicted_labels = [int(label) for label in estimator.predict(test_features)]
    predicted_probabilities = estimator.predict_proba(test_features)
    probabilities = [float(probability[1]) for probability in predicted_probabilities]
    return predicted_labels, probabilities

# pylint: disable=too-many-arguments
def _build_prediction_records(
    *,
    model_name: str,
    fold_index: int,
    test_samples: list[DatasetSample],
    predicted_labels: list[int],
    probabilities: list[float],
    fee_rate: float,
    regime_labels_by_row_id: dict[str, str],
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for sample, label, prob_up in zip(test_samples, predicted_labels, probabilities, strict=True):
        long_trade_taken = 1 if label == 1 else 0
        long_only_gross_value_proxy = sample.future_return_3 if long_trade_taken else 0.0
        long_only_net_value_proxy = (
            long_only_gross_value_proxy - fee_rate if long_trade_taken else 0.0
        )
        records.append(
            PredictionRecord(
                model_name=model_name,
                fold_index=fold_index,
                row_id=sample.row_id,
                symbol=sample.symbol,
                interval_begin=to_rfc3339(sample.interval_begin),
                as_of_time=to_rfc3339(sample.as_of_time),
                y_true=sample.label,
                y_pred=label,
                prob_up=prob_up,
                confidence=max(prob_up, 1.0 - prob_up),
                regime_label=regime_labels_by_row_id[sample.row_id],
                long_trade_taken=long_trade_taken,
                future_return_3=sample.future_return_3,
                long_only_gross_value_proxy=long_only_gross_value_proxy,
                long_only_net_value_proxy=long_only_net_value_proxy,
            )
        )
    return records


def _compute_metrics(predictions: list[PredictionRecord]) -> dict[str, Any]:
    y_true = [prediction.y_true for prediction in predictions]
    y_pred = [prediction.y_pred for prediction in predictions]
    prob_up = [prediction.prob_up for prediction in predictions]
    directional_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    trade_count = sum(prediction.long_trade_taken for prediction in predictions)
    trade_rate = trade_count / len(predictions)
    mean_long_only_gross_value_proxy = (
        sum(prediction.long_only_gross_value_proxy for prediction in predictions)
        / len(predictions)
    )
    mean_long_only_net_value_proxy = (
        sum(prediction.long_only_net_value_proxy for prediction in predictions)
        / len(predictions)
    )
    return {
        "metrics": {
            "directional_accuracy": directional_accuracy,
            "precision_class_0": float(precision[0]),
            "precision_class_1": float(precision[1]),
            "recall_class_0": float(recall[0]),
            "recall_class_1": float(recall[1]),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
            "brier_score": brier_score_loss(y_true, prob_up),
            "trade_count": trade_count,
            "trade_rate": trade_rate,
            "mean_long_only_gross_value_proxy": mean_long_only_gross_value_proxy,
            "mean_long_only_net_value_proxy": mean_long_only_net_value_proxy,
        },
        "confidence_analysis": _confidence_analysis(predictions),
    }


def _confidence_analysis(predictions: list[PredictionRecord]) -> list[dict[str, Any]]:
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    analysis: list[dict[str, Any]] = []
    for lower_bound, upper_bound in bins:
        bin_rows = [
            prediction
            for prediction in predictions
            if lower_bound <= prediction.confidence < upper_bound
        ]
        if not bin_rows:
            continue
        accuracy = sum(
            int(prediction.y_true == prediction.y_pred) for prediction in bin_rows
        ) / len(bin_rows)
        analysis.append(
            {
                "bin": f"{lower_bound:.2f}-{min(upper_bound, 1.0):.2f}",
                "count": len(bin_rows),
                "accuracy": accuracy,
                "mean_confidence": sum(row.confidence for row in bin_rows) / len(bin_rows),
                "mean_prob_up": sum(row.prob_up for row in bin_rows) / len(bin_rows),
            }
        )
    return analysis


def _build_aggregate_summary(
    *,
    all_prediction_rows: list[PredictionRecord],
    fee_rate: float,
) -> dict[str, dict[str, Any]]:
    del fee_rate
    by_model: dict[str, list[PredictionRecord]] = {}
    for prediction in all_prediction_rows:
        by_model.setdefault(prediction.model_name, []).append(prediction)

    aggregate_summary: dict[str, dict[str, Any]] = {}
    for model_name, rows in by_model.items():
        metrics = _compute_metrics(rows)
        aggregate_summary[model_name] = {
            "prediction_count": len(rows),
            **metrics["metrics"],
            "confidence_analysis": metrics["confidence_analysis"],
            "economics_by_regime": _build_regime_economics(rows),
        }
    return aggregate_summary


def _select_winner(aggregate_summary: dict[str, dict[str, Any]]) -> str:
    learned_models = {
        name: metrics
        for name, metrics in aggregate_summary.items()
        if name not in _REQUIRED_PROMOTION_BASELINES
    }
    if not learned_models:
        raise ValueError("No learned models were available for winner selection")
    return sorted(
        learned_models.items(),
        key=lambda item: (
            -item[1]["mean_long_only_net_value_proxy"],
            -item[1]["directional_accuracy"],
            item[1]["brier_score"],
        ),
    )[0][0]


def _fit_winner_on_full_dataset(
    *,
    winner_name: str,
    model_factories: dict[str, Callable[[], Any]],
    samples: tuple[DatasetSample, ...],
) -> tuple[Any, list[str]]:
    train_labels = [sample.label for sample in samples]
    if len(set(train_labels)) < 2:
        raise ValueError("Winner cannot be fit because the full dataset has only one class")

    fitted_model = model_factories[winner_name]()
    fitted_model.fit([sample.features for sample in samples], train_labels)
    feature_names = _resolve_expanded_feature_names(
        fitted_model=fitted_model,
    )
    return fitted_model, feature_names


def _resolve_expanded_feature_names(
    *,
    fitted_model: Any,
) -> list[str]:
    """Resolve stable expanded feature names across model families."""
    if hasattr(fitted_model, "get_expanded_feature_names"):
        names = fitted_model.get_expanded_feature_names()
        return [str(name) for name in names]
    vectorizer = fitted_model.named_steps["vectorizer"]
    return [str(name) for name in vectorizer.get_feature_names_out()]


def _save_model_artifact(
    *,
    model_path: Path,
    model_name: str,
    fitted_model: Any,
    feature_columns: tuple[str, ...],
    expanded_feature_names: list[str],
) -> None:
    payload = {
        "model_name": model_name,
        "trained_at": to_rfc3339(utc_now()),
        "feature_columns": list(feature_columns),
        "expanded_feature_names": expanded_feature_names,
        "model": fitted_model,
    }
    joblib.dump(payload, model_path)
    reloaded = joblib.load(model_path)
    if reloaded["model_name"] != model_name:
        raise ValueError("Saved model artifact failed reload validation")


def _build_summary_payload(
    *,
    config: TrainingConfig,
    dataset_manifest: dict[str, Any],
    aggregate_summary: dict[str, dict[str, Any]],
    regime_context: TrainingRegimeContext,
    winner_name: str,
    model_path: Path,
) -> dict[str, Any]:
    winner_metrics = aggregate_summary[winner_name]
    learned_model_names = tuple(
        model_name
        for model_name in aggregate_summary
        if model_name not in _REQUIRED_PROMOTION_BASELINES
    )
    learned_models_positive_after_costs = [
        model_name
        for model_name in learned_model_names
        if aggregate_summary[model_name]["mean_long_only_net_value_proxy"] > 0.0
    ]
    learned_models_beating_persistence = _learned_models_beating_after_costs(
        aggregate_summary,
        baseline_name="persistence_3",
        learned_model_names=learned_model_names,
    )
    learned_models_beating_dummy = _learned_models_beating_after_costs(
        aggregate_summary,
        baseline_name="dummy_most_frequent",
        learned_model_names=learned_model_names,
    )
    learned_models_beating_all_baselines = [
        model_name
        for model_name in learned_model_names
        if all(
            aggregate_summary[model_name]["mean_long_only_net_value_proxy"]
            > aggregate_summary[baseline_name]["mean_long_only_net_value_proxy"]
            for baseline_name in _REQUIRED_PROMOTION_BASELINES
        )
    ]
    meets_acceptance_target = (
        winner_metrics["mean_long_only_net_value_proxy"] > 0.0
        and winner_name in learned_models_beating_all_baselines
    )
    return {
        "generated_at": to_rfc3339(utc_now()),
        "source_table": config.source_table,
        "dataset_manifest": dataset_manifest,
        "economics_contract": {
            "name": "LONG_ONLY_AFTER_COST_PROXY",
            "description": (
                "Predicted UP enters one long for the 3-candle horizon; "
                "predicted DOWN stays flat; the configured round-trip fee is "
                "only charged when a long trade is taken."
            ),
            "primary_metric": "mean_long_only_net_value_proxy",
            "fee_rate": config.round_trip_fee_rate,
        },
        "regime_economics": {
            "source": "M8_PERCENTILE_RULES_REFIT_ON_TRAINING_SOURCE",
            "config_path": regime_context.config_path,
            "high_vol_percentile": regime_context.high_vol_percentile,
            "trend_abs_momentum_percentile": (
                regime_context.trend_abs_momentum_percentile
            ),
            "thresholds_by_symbol": {
                symbol: threshold.to_dict()
                for symbol, threshold in sorted(regime_context.thresholds_by_symbol.items())
            },
        },
        "models": aggregate_summary,
        "official_baseline": "persistence_3",
        "promotion_baselines": list(_REQUIRED_PROMOTION_BASELINES),
        "winner": {
            "model_name": winner_name,
            "model_path": str(model_path),
            "selection_rule": {
                "primary": "mean_long_only_net_value_proxy",
                "tie_break_1": "directional_accuracy",
                "tie_break_2": "lower_brier_score",
            },
        },
        "acceptance": {
            "winner_after_cost_positive": winner_metrics["mean_long_only_net_value_proxy"]
            > 0.0,
            "learned_models_positive_after_costs": learned_models_positive_after_costs,
            "learned_models_beating_persistence_after_costs": (
                learned_models_beating_persistence
            ),
            "learned_models_beating_dummy_after_costs": learned_models_beating_dummy,
            "learned_models_beating_all_baselines_after_costs": (
                learned_models_beating_all_baselines
            ),
            "meets_acceptance_target": meets_acceptance_target,
        },
    }


def _build_training_regime_context(
    samples: tuple[DatasetSample, ...],
    config: TrainingConfig,
) -> TrainingRegimeContext:
    regime_config = load_regime_config(_REGIME_CONFIG_PATH)
    missing_symbols = sorted(set(config.symbols) - set(regime_config.symbols))
    if missing_symbols:
        raise ValueError(
            "M8 regime config is missing training symbols required for regime-sliced "
            f"economics: {missing_symbols}"
        )
    regime_rows = [_sample_to_regime_row(sample) for sample in samples]
    thresholds_by_symbol = _fit_training_regime_thresholds(
        regime_rows,
        symbols=config.symbols,
        high_vol_percentile=regime_config.thresholds.high_vol_percentile,
        trend_abs_momentum_percentile=(
            regime_config.thresholds.trend_abs_momentum_percentile
        ),
    )
    labels_by_row_id = {
        sample.row_id: classify_row(
            _sample_to_regime_row(sample),
            thresholds_by_symbol,
        )
        for sample in samples
    }
    return TrainingRegimeContext(
        config_path=str(_REGIME_CONFIG_PATH.resolve()),
        high_vol_percentile=regime_config.thresholds.high_vol_percentile,
        trend_abs_momentum_percentile=(
            regime_config.thresholds.trend_abs_momentum_percentile
        ),
        thresholds_by_symbol=thresholds_by_symbol,
        labels_by_row_id=labels_by_row_id,
    )


def _sample_to_regime_row(sample: DatasetSample) -> RegimeSourceRow:
    missing_inputs = [
        column
        for column in ("realized_vol_12", "momentum_3", "macd_line_12_26")
        if column not in sample.features or sample.features[column] is None
    ]
    if missing_inputs:
        raise ValueError(
            "Training samples are missing required regime inputs for regime-sliced "
            f"economics: {missing_inputs}"
        )
    return RegimeSourceRow(
        symbol=sample.symbol,
        interval_begin=sample.interval_begin,
        as_of_time=sample.as_of_time,
        realized_vol_12=float(sample.features["realized_vol_12"]),
        momentum_3=float(sample.features["momentum_3"]),
        macd_line_12_26=float(sample.features["macd_line_12_26"]),
    )


def _build_regime_economics(
    predictions: list[PredictionRecord],
) -> dict[str, dict[str, Any]]:
    rows_by_regime: dict[str, list[PredictionRecord]] = {}
    for prediction in predictions:
        rows_by_regime.setdefault(prediction.regime_label, []).append(prediction)
    ordered_regimes = list(REGIME_LABELS) + sorted(set(rows_by_regime) - set(REGIME_LABELS))
    return {
        regime_label: _build_regime_economics_row(rows_by_regime[regime_label])
        for regime_label in ordered_regimes
        if regime_label in rows_by_regime
    }


def _build_regime_economics_row(predictions: list[PredictionRecord]) -> dict[str, Any]:
    prediction_count = len(predictions)
    trade_count = sum(prediction.long_trade_taken for prediction in predictions)
    mean_long_only_gross_value_proxy = (
        sum(prediction.long_only_gross_value_proxy for prediction in predictions)
        / prediction_count
    )
    mean_long_only_net_value_proxy = (
        sum(prediction.long_only_net_value_proxy for prediction in predictions)
        / prediction_count
    )
    return {
        "prediction_count": prediction_count,
        "trade_count": trade_count,
        "trade_rate": trade_count / prediction_count,
        "mean_long_only_gross_value_proxy": mean_long_only_gross_value_proxy,
        "mean_long_only_net_value_proxy": mean_long_only_net_value_proxy,
        "after_cost_positive": mean_long_only_net_value_proxy > 0.0,
    }


def _fit_training_regime_thresholds(
    rows: list[RegimeSourceRow],
    *,
    symbols: tuple[str, ...],
    high_vol_percentile: float,
    trend_abs_momentum_percentile: float,
) -> dict[str, SymbolThresholds]:
    rows_by_symbol: dict[str, list[RegimeSourceRow]] = {symbol: [] for symbol in symbols}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)

    thresholds_by_symbol: dict[str, SymbolThresholds] = {}
    for symbol in symbols:
        symbol_rows = sorted(
            rows_by_symbol.get(symbol, []),
            key=lambda row: (row.interval_begin, row.as_of_time),
        )
        if not symbol_rows:
            raise ValueError(
                "Training data does not contain rows for regime-sliced economics "
                f"symbol {symbol}"
            )
        thresholds_by_symbol[symbol] = SymbolThresholds(
            symbol=symbol,
            fitted_row_count=len(symbol_rows),
            high_vol_threshold=compute_percentile(
                [row.realized_vol_12 for row in symbol_rows],
                high_vol_percentile,
            ),
            trend_abs_threshold=compute_percentile(
                [abs(row.momentum_3) for row in symbol_rows],
                trend_abs_momentum_percentile,
            ),
        )
    return thresholds_by_symbol


def _learned_models_beating_after_costs(
    aggregate_summary: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    learned_model_names: tuple[str, ...],
) -> list[str]:
    baseline_metric = aggregate_summary[baseline_name]["mean_long_only_net_value_proxy"]
    return [
        model_name
        for model_name in learned_model_names
        if aggregate_summary[model_name]["mean_long_only_net_value_proxy"] > baseline_metric
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
