"""Offline training orchestration for Stream Alpha M3."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.training.baselines import PersistenceBaseline, build_dummy_classifier
from app.training.dataset import (
    DatasetSample,
    TrainingConfig,
    load_training_config,
    load_training_dataset,
)
from app.training.splits import WalkForwardFold, build_walk_forward_splits


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
    future_return_3: float
    gross_value_proxy: float
    net_value_proxy: float

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
            "future_return_3": self.future_return_3,
            "gross_value_proxy": self.gross_value_proxy,
            "net_value_proxy": self.net_value_proxy,
        }


def run_training(config_path: Path) -> Path:  # pylint: disable=too-many-locals
    """Run the full offline M3 training flow and save artifacts to disk."""
    config = load_training_config(config_path)
    dataset = load_training_dataset(config)
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
        winner_name=winner_name,
        model_path=model_path,
    )
    _write_json(artifact_dir / "summary.json", summary)
    return artifact_dir


def _create_artifact_dir(config: TrainingConfig) -> Path:
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _build_model_factories(
    config: TrainingConfig,
) -> dict[str, Callable[[], Any]]:
    return {
        "persistence_3": _build_persistence_baseline,
        "dummy_most_frequent": build_dummy_classifier,
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("classifier", LogisticRegression(**config.models.logistic_regression)),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(**config.models.hist_gradient_boosting),
                ),
            ]
        ),
    }


def _build_persistence_baseline() -> PersistenceBaseline:
    return PersistenceBaseline()


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


def _evaluate_fold(
    *,
    train_samples: list[DatasetSample],
    test_samples: list[DatasetSample],
    fold: WalkForwardFold,
    model_factories: dict[str, Callable[[], Any]],
    fee_rate: float,
) -> tuple[list[dict[str, Any]], list[PredictionRecord]]:
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
    if model_name in {"logistic_regression", "hist_gradient_boosting"} and len(
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
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for sample, label, prob_up in zip(test_samples, predicted_labels, probabilities, strict=True):
        position = 1.0 if label == 1 else -1.0
        gross_value_proxy = position * sample.future_return_3
        net_value_proxy = gross_value_proxy - fee_rate
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
                future_return_3=sample.future_return_3,
                gross_value_proxy=gross_value_proxy,
                net_value_proxy=net_value_proxy,
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
    mean_gross_value_proxy = sum(prediction.gross_value_proxy for prediction in predictions) / len(
        predictions
    )
    mean_net_value_proxy = sum(prediction.net_value_proxy for prediction in predictions) / len(
        predictions
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
            "mean_gross_value_proxy": mean_gross_value_proxy,
            "mean_net_value_proxy": mean_net_value_proxy,
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
        }
    return aggregate_summary


def _select_winner(aggregate_summary: dict[str, dict[str, Any]]) -> str:
    learned_models = {
        name: metrics
        for name, metrics in aggregate_summary.items()
        if name in {"logistic_regression", "hist_gradient_boosting"}
    }
    if not learned_models:
        raise ValueError("No learned models were available for winner selection")
    return sorted(
        learned_models.items(),
        key=lambda item: (
            -item[1]["mean_net_value_proxy"],
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
    vectorizer = fitted_model.named_steps["vectorizer"]
    feature_names = [str(name) for name in vectorizer.get_feature_names_out()]
    return fitted_model, feature_names


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
    winner_name: str,
    model_path: Path,
) -> dict[str, Any]:
    persistence_metrics = aggregate_summary["persistence_3"]
    learned_beating_persistence = [
        model_name
        for model_name in ("logistic_regression", "hist_gradient_boosting")
        if aggregate_summary[model_name]["directional_accuracy"]
        > persistence_metrics["directional_accuracy"]
        and aggregate_summary[model_name]["mean_net_value_proxy"]
        > persistence_metrics["mean_net_value_proxy"]
    ]
    return {
        "generated_at": to_rfc3339(utc_now()),
        "source_table": config.source_table,
        "dataset_manifest": dataset_manifest,
        "models": aggregate_summary,
        "official_baseline": "persistence_3",
        "winner": {
            "model_name": winner_name,
            "model_path": str(model_path),
            "selection_rule": {
                "primary": "mean_net_value_proxy",
                "tie_break_1": "directional_accuracy",
                "tie_break_2": "lower_brier_score",
            },
        },
        "acceptance": {
            "learned_models_beating_persistence": learned_beating_persistence,
            "meets_acceptance_target": bool(learned_beating_persistence),
        },
    }


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
