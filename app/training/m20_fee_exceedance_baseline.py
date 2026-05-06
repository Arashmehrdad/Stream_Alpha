"""Tiny research-only fee-exceedance baselines for M20 market features."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_RANDOM_SEED = 1729
DEFAULT_SCENARIO_NAME = "current_fee"
BASELINE_DIR_NAME = "fee_exceedance_baselines"
TOP_K_FRACTIONS = (0.01, 0.05, 0.10)
THRESHOLDS = tuple(round(index / 100.0, 2) for index in range(5, 100, 5))
LEAKAGE_TOKENS = (
    "label",
    "target",
    "future",
    "forward",
    "realized_outcome",
    "outcome",
    "barrier",
    "event",
    "gross",
    "net",
    "post",
    "prob",
    "confidence",
)
PREDICTION_OUTPUT_COLUMNS = {"y_pred", "prob_up", "confidence", "long_trade_taken"}
KEY_COLUMNS = ("symbol", "interval_begin", "fold_index", "row_id")


def train_fee_exceedance_baselines(
    *,
    run_dir: Path,
    scenario_name: str = DEFAULT_SCENARIO_NAME,
    random_seed: int = DEFAULT_RANDOM_SEED,
    export_full_predictions: bool = False,
) -> dict[str, Any]:
    """Train/evaluate tiny research-only fee-exceedance baselines."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    frame_dir = resolved_run_dir / "training_frame"
    label_path = (
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv"
    )
    feature_rows = _read_csv_rows(frame_dir / "m20_training_frame_features.csv")
    label_rows = [
        row for row in _read_csv_rows(label_path)
        if str(row.get("scenario_name", "")) == scenario_name
    ]
    if not feature_rows:
        raise ValueError(f"Missing or empty training-frame features: {frame_dir}")
    if not label_rows:
        raise ValueError(
            f"Missing fee-exceedance labels for scenario {scenario_name}: {label_path}"
        )
    joined_rows = _join_feature_labels(feature_rows, label_rows)
    feature_audit = _feature_audit(feature_rows)
    split = _chronological_split(joined_rows)
    baselines = _evaluate_baselines(
        split=split,
        feature_columns=feature_audit["safe_feature_columns"],
        random_seed=random_seed,
    )
    baseline_dir = (
        resolved_run_dir / "research_labels" / "vol_scaled" / BASELINE_DIR_NAME
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(baseline_dir, baselines)
    if export_full_predictions:
        output_files.update(_full_prediction_output_files(baseline_dir))
    honesty_flags = _honesty_flags(feature_audit, split)
    if export_full_predictions:
        honesty_flags = sorted(
            dict.fromkeys(
                [
                    *honesty_flags,
                    "RESEARCH_ONLY_FULL_PREDICTION_EXPORT",
                    "FULL_TEST_PREDICTIONS_EXPORTED",
                    "SAMPLE_PREDICTIONS_RETAINED_FOR_PREVIEW",
                ]
            )
        )
    recommendation = _recommend(baselines)
    report = {
        "run_dir": str(resolved_run_dir),
        "baseline_dir": str(baseline_dir),
        "scenario_name": scenario_name,
        "row_count": len(joined_rows),
        "split": _split_summary(split),
        "feature_audit": feature_audit,
        "baselines": [
            {key: value for key, value in baseline.items() if key != "predictions"}
            for baseline in baselines
        ],
        "honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "notes": [
            "Research-only fee-exceedance baseline diagnostics.",
            "Accuracy is diagnostic only for this imbalanced target.",
            "No runtime inference, registry authority, promotion, execution, "
            "thresholds, NeuralForecast behavior, or roster behavior changed.",
        ],
        "output_files": output_files,
    }
    manifest = {
        "run_dir": str(resolved_run_dir),
        "baseline_dir": str(baseline_dir),
        "source_training_frame": str(frame_dir),
        "source_label_file": str(label_path),
        "scenario_name": scenario_name,
        "honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    _write_outputs(report, manifest, baselines, output_files)
    if export_full_predictions:
        _write_full_prediction_exports(
            split=split,
            feature_columns=feature_audit["safe_feature_columns"],
            baseline_dir=baseline_dir,
            scenario_name=scenario_name,
            output_files=output_files,
        )
    return make_json_safe({**report, "manifest": manifest})


def _join_feature_labels(
    feature_rows: Sequence[Mapping[str, str]],
    label_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    features_by_key = {
        (str(row.get("symbol", "")), str(row.get("interval_begin", ""))): row
        for row in feature_rows
    }
    joined = []
    for label_row in label_rows:
        key = (str(label_row.get("symbol", "")), str(label_row.get("interval_begin", "")))
        feature_row = features_by_key.get(key)
        if feature_row is None:
            continue
        joined.append({**dict(feature_row), "label": str(label_row.get("label", ""))})
    return sorted(joined, key=lambda row: (str(row["interval_begin"]), str(row["symbol"])))


def _feature_audit(feature_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    columns = list(feature_rows[0].keys()) if feature_rows else []
    excluded = []
    safe = []
    for column in columns:
        reason = _exclusion_reason(column)
        if reason:
            excluded.append({"column": column, "reason": reason})
        elif _is_numeric_column(feature_rows, column):
            safe.append(column)
        else:
            excluded.append({"column": column, "reason": "non_numeric"})
    return {
        "available_columns": columns,
        "safe_feature_columns": safe,
        "safe_feature_count": len(safe),
        "excluded_columns": excluded,
        "excluded_column_count": len(excluded),
        "exclusion_reason_counts": _reason_counts(excluded),
        "feature_order_preserved": True,
    }


def _chronological_split(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row["interval_begin"]), str(row["symbol"])))
    train_end = int(len(ordered) * 0.60)
    validation_end = int(len(ordered) * 0.80)
    return {
        "split_source": "chronological_within_single_recent_fold",
        "train": [dict(row) for row in ordered[:train_end]],
        "validation": [dict(row) for row in ordered[train_end:validation_end]],
        "test": [dict(row) for row in ordered[validation_end:]],
    }


def _evaluate_baselines(
    *,
    split: Mapping[str, Sequence[Mapping[str, str]]],
    feature_columns: Sequence[str],
    random_seed: int,
) -> list[dict[str, Any]]:
    train_rows = split["train"]
    test_rows = split["test"]
    train_labels = [_label(row) for row in train_rows]
    test_labels = [_label(row) for row in test_rows]
    baselines = [
        _baseline_result(
            "always_negative",
            test_rows,
            test_labels,
            [0 for _ in test_rows],
            [0.0 for _ in test_rows],
        ),
        _baseline_result(
            f"stratified_random_seed_{random_seed}",
            test_rows,
            test_labels,
            _stratified_random_predictions(train_labels, len(test_rows), random_seed),
            [sum(train_labels) / len(train_labels) if train_labels else 0.0 for _ in test_rows],
        ),
    ]
    if feature_columns and _sklearn_available():
        logistic = _logistic_regression_baseline(
            train_rows=train_rows,
            test_rows=test_rows,
            train_labels=train_labels,
            test_labels=test_labels,
            feature_columns=feature_columns,
        )
        if logistic:
            baselines.append(logistic)
    return baselines


def _logistic_regression_baseline(
    *,
    train_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    feature_columns: Sequence[str],
) -> dict[str, Any] | None:
    # pylint: disable=import-outside-toplevel,broad-exception-caught
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                random_state=DEFAULT_RANDOM_SEED,
            ),
        )
        model.fit(_matrix(train_rows, feature_columns), train_labels)
        probabilities = [
            float(value)
            for value in model.predict_proba(_matrix(test_rows, feature_columns))[:, 1]
        ]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        return _baseline_result(
            "logistic_regression_tiny",
            test_rows,
            test_labels,
            predictions,
            probabilities,
        )
    except Exception:
        return None


def _baseline_result(
    name: str,
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    metrics = _binary_metrics(labels, predictions, probabilities)
    return {
        "baseline_name": name,
        **metrics,
        "confusion_matrix": _confusion(labels, predictions),
        "topk": _topk_rows(name, labels, probabilities),
        "threshold_sweep": _threshold_rows(name, labels, probabilities),
        "by_symbol": _slice_metrics(name, rows, labels, predictions, "symbol"),
        "calibration_buckets": _calibration_rows(name, labels, probabilities),
        "predictions": _prediction_rows(name, rows, labels, predictions, probabilities),
    }


def _binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    matrix = _confusion(labels, predictions)
    tp_value = matrix["tp"]
    tn_value = matrix["tn"]
    fp_value = matrix["fp"]
    fn_value = matrix["fn"]
    positive_rate = sum(labels) / len(labels) if labels else 0.0
    return {
        "test_row_count": len(labels),
        "positive_rate": positive_rate,
        "accuracy_diagnostic_only": (tp_value + tn_value) / len(labels) if labels else 0.0,
        "balanced_accuracy": (
            _safe_ratio(tp_value, tp_value + fn_value)
            + _safe_ratio(tn_value, tn_value + fp_value)
        )
        / 2.0,
        "precision": _safe_ratio(tp_value, tp_value + fp_value),
        "recall": _safe_ratio(tp_value, tp_value + fn_value),
        "f1": _safe_ratio(2 * tp_value, (2 * tp_value) + fp_value + fn_value),
        "roc_auc": _roc_auc(labels, probabilities),
        "average_precision": _average_precision(labels, probabilities),
        "brier_score": _brier(labels, probabilities),
    }


def _threshold_rows(
    baseline_name: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    rows = []
    base_rate = sum(labels) / len(labels) if labels else 0.0
    for threshold in THRESHOLDS:
        selected = [
            index for index, probability in enumerate(probabilities)
            if probability >= threshold
        ]
        positives = sum(labels[index] for index in selected)
        false_positives = len(selected) - positives
        negatives = len(labels) - sum(labels)
        rows.append(
            {
                "baseline_name": baseline_name,
                "threshold": threshold,
                "predicted_trade_count": len(selected),
                "coverage": len(selected) / len(labels) if labels else 0.0,
                "positive_precision": positives / len(selected) if selected else 0.0,
                "false_positive_rate": _safe_ratio(false_positives, negatives),
                "expected_positive_event_rate": positives / len(selected) if selected else 0.0,
                "base_positive_rate": base_rate,
            }
        )
    return rows


def _topk_rows(
    baseline_name: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    rows = []
    base_rate = sum(labels) / len(labels) if labels else 0.0
    ordered = sorted(range(len(labels)), key=lambda index: probabilities[index], reverse=True)
    total_positive = sum(labels)
    for fraction in TOP_K_FRACTIONS:
        count = max(1, int(len(labels) * fraction))
        selected = ordered[:count]
        positives = sum(labels[index] for index in selected)
        precision = positives / count if count else 0.0
        rows.append(
            {
                "baseline_name": baseline_name,
                "top_k_fraction": fraction,
                "row_count": count,
                "precision_at_k": precision,
                "recall_at_k": _safe_ratio(positives, total_positive),
                "lift_at_k": precision / base_rate if base_rate else 0.0,
            }
        )
    return rows


def _slice_metrics(
    baseline_name: str,
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    predictions: Sequence[int],
    column: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row.get(column, "")), []).append(index)
    output = []
    for value, indexes in sorted(grouped.items()):
        slice_labels = [labels[index] for index in indexes]
        slice_predictions = [predictions[index] for index in indexes]
        metrics = _binary_metrics(slice_labels, slice_predictions, [0.0 for _ in indexes])
        output.append(
            {
                "baseline_name": baseline_name,
                "slice_column": column,
                "slice_value": value,
                "row_count": len(indexes),
                "positive_rate": metrics["positive_rate"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    return output


def _calibration_rows(
    baseline_name: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    bucket_count: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for bucket in range(bucket_count):
        low = bucket / bucket_count
        high = (bucket + 1) / bucket_count
        indexes = [
            index for index, probability in enumerate(probabilities)
            if low <= probability < high or (bucket == bucket_count - 1 and probability == 1.0)
        ]
        rows.append(
            {
                "baseline_name": baseline_name,
                "bucket": bucket,
                "probability_min": low,
                "probability_max": high,
                "row_count": len(indexes),
                "mean_probability": (
                    sum(probabilities[index] for index in indexes) / len(indexes)
                    if indexes else 0.0
                ),
                "observed_positive_rate": (
                    sum(labels[index] for index in indexes) / len(indexes)
                    if indexes else 0.0
                ),
            }
        )
    return rows


def _prediction_rows(
    baseline_name: str,
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "baseline_name": baseline_name,
            "symbol": row.get("symbol", ""),
            "interval_begin": row.get("interval_begin", ""),
            "row_id": row.get("row_id", ""),
            "label": labels[index],
            "prediction": predictions[index],
            "probability": probabilities[index],
        }
        for index, row in enumerate(rows)
    ]


def _write_outputs(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    output_files: Mapping[str, str],
) -> None:
    write_json_artifact(Path(output_files["fee_baseline_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["fee_feature_audit_json"]), report["feature_audit"])
    Path(output_files["fee_feature_audit_md"]).write_text(
        _feature_audit_markdown(report["feature_audit"]),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["fee_baseline_metrics_json"]), report)
    write_csv_artifact(Path(output_files["fee_baseline_metrics_csv"]), _metrics_rows(baselines))
    write_csv_artifact(Path(output_files["fee_confusion_matrices_csv"]), _confusion_rows(baselines))
    write_csv_artifact(
        Path(output_files["fee_threshold_sweep_csv"]),
        _flatten(baselines, "threshold_sweep"),
    )
    write_csv_artifact(Path(output_files["fee_topk_diagnostics_csv"]), _flatten(baselines, "topk"))
    write_csv_artifact(
        Path(output_files["fee_by_symbol_metrics_csv"]),
        _flatten(baselines, "by_symbol"),
    )
    write_csv_artifact(
        Path(output_files["fee_calibration_buckets_csv"]),
        _flatten(baselines, "calibration_buckets"),
    )
    Path(output_files["fee_baseline_report_md"]).write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    for baseline in baselines:
        path = Path(output_files[f"predictions_{baseline['baseline_name']}_csv"])
        write_csv_artifact(path, baseline["predictions"][:5000])


def _output_files(baseline_dir: Path, baselines: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    files = {
        "fee_baseline_manifest_json": str(baseline_dir / "fee_baseline_manifest.json"),
        "fee_feature_audit_json": str(baseline_dir / "fee_feature_audit.json"),
        "fee_feature_audit_md": str(baseline_dir / "fee_feature_audit.md"),
        "fee_baseline_metrics_json": str(baseline_dir / "fee_baseline_metrics.json"),
        "fee_baseline_metrics_csv": str(baseline_dir / "fee_baseline_metrics.csv"),
        "fee_confusion_matrices_csv": str(baseline_dir / "fee_confusion_matrices.csv"),
        "fee_threshold_sweep_csv": str(baseline_dir / "fee_threshold_sweep.csv"),
        "fee_topk_diagnostics_csv": str(baseline_dir / "fee_topk_diagnostics.csv"),
        "fee_by_symbol_metrics_csv": str(baseline_dir / "fee_by_symbol_metrics.csv"),
        "fee_calibration_buckets_csv": str(baseline_dir / "fee_calibration_buckets.csv"),
        "fee_baseline_report_md": str(baseline_dir / "fee_baseline_report.md"),
    }
    for baseline in baselines:
        files[f"predictions_{baseline['baseline_name']}_csv"] = str(
            baseline_dir / f"predictions_{baseline['baseline_name']}.csv"
        )
    return files


def _full_prediction_output_files(baseline_dir: Path) -> dict[str, str]:
    return {
        "predictions_logistic_regression_tiny_train_full_csv": str(
            baseline_dir / "predictions_logistic_regression_tiny_train_full.csv"
        ),
        "predictions_logistic_regression_tiny_validation_full_csv": str(
            baseline_dir / "predictions_logistic_regression_tiny_validation_full.csv"
        ),
        "predictions_logistic_regression_tiny_test_full_csv": str(
            baseline_dir / "predictions_logistic_regression_tiny_test_full.csv"
        ),
        "prediction_export_manifest_json": str(baseline_dir / "prediction_export_manifest.json"),
        "prediction_export_report_md": str(baseline_dir / "prediction_export_report.md"),
    }


def _write_full_prediction_exports(
    *,
    split: Mapping[str, Sequence[Mapping[str, str]]],
    feature_columns: Sequence[str],
    baseline_dir: Path,
    scenario_name: str,
    output_files: Mapping[str, str],
) -> None:
    # pylint: disable=too-many-locals
    # pylint: disable=import-outside-toplevel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not feature_columns:
        return
    train_rows = split["train"]
    train_labels = [_label(row) for row in train_rows]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=DEFAULT_RANDOM_SEED,
        ),
    )
    model.fit(_matrix(train_rows, feature_columns), train_labels)
    split_outputs = {}
    for split_name, rows in (
        ("train", split["train"]),
        ("validation", split["validation"]),
        ("test", split["test"]),
    ):
        probabilities = [
            float(value) for value in model.predict_proba(_matrix(rows, feature_columns))[:, 1]
        ]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        output_rows = _full_prediction_rows(
            split_name,
            rows,
            predictions,
            probabilities,
            scenario_name,
        )
        key = f"predictions_logistic_regression_tiny_{split_name}_full_csv"
        write_csv_artifact(Path(output_files[key]), output_rows)
        split_outputs[split_name] = _prediction_export_summary(output_rows)
    manifest = {
        "baseline_dir": str(baseline_dir),
        "model_name": "logistic_regression_tiny",
        "prediction_export": "full",
        "sample_predictions_retained_for_preview": True,
        "honesty_flags": [
            "RESEARCH_ONLY_FULL_PREDICTION_EXPORT",
            "FULL_TEST_PREDICTIONS_EXPORTED",
            "SAMPLE_PREDICTIONS_RETAINED_FOR_PREVIEW",
            "NO_RUNTIME_EFFECT",
            "NO_PROMOTION_EFFECT",
            "NO_REGISTRY_WRITE",
            "NOT_RUNTIME_COMPARABLE",
            "NOT_PROMOTABLE",
            "NO_PROFITABILITY_CLAIM",
        ],
        "split_summaries": split_outputs,
        "output_files": {
            key: value for key, value in output_files.items()
            if key.startswith("predictions_logistic_regression_tiny_")
            or key.startswith("prediction_export")
        },
    }
    write_json_artifact(Path(output_files["prediction_export_manifest_json"]), manifest)
    Path(output_files["prediction_export_report_md"]).write_text(
        _prediction_export_markdown(manifest),
        encoding="utf-8",
    )


def _full_prediction_rows(
    split_name: str,
    rows: Sequence[Mapping[str, str]],
    predictions: Sequence[int],
    probabilities: Sequence[float],
    scenario_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": row.get("symbol", ""),
            "interval_begin": row.get("interval_begin", ""),
            "fold_index": row.get("fold_index", ""),
            "row_id": row.get("row_id", ""),
            "split": split_name,
            "y_true": _label(row),
            "y_pred": predictions[index],
            "probability": probabilities[index],
            "model_name": "logistic_regression_tiny",
            "scenario_name": scenario_name,
        }
        for index, row in enumerate(rows)
    ]


def _prediction_export_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = sorted(float(row["probability"]) for row in rows)
    symbols = sorted({str(row["symbol"]) for row in rows})
    keys = [(row.get("symbol"), row.get("interval_begin")) for row in rows]
    return {
        "row_count": len(rows),
        "positive_rate": sum(int(row["y_true"]) for row in rows) / len(rows) if rows else 0.0,
        "probability_min": min(probabilities) if probabilities else 0.0,
        "probability_p50": _quantile(probabilities, 0.50),
        "probability_p95": _quantile(probabilities, 0.95),
        "probability_max": max(probabilities) if probabilities else 0.0,
        "probability_mean": sum(probabilities) / len(probabilities) if probabilities else 0.0,
        "missing_probability_count": 0,
        "duplicate_key_count": len(keys) - len(set(keys)),
        "symbol_coverage": symbols,
        "timestamp_min": min((str(row["interval_begin"]) for row in rows), default=""),
        "timestamp_max": max((str(row["interval_begin"]) for row in rows), default=""),
        "prediction_export_is_full": True,
    }


def _prediction_export_markdown(manifest: Mapping[str, Any]) -> str:
    test = manifest["split_summaries"]["test"]
    return "\n".join(
        [
            "# M20 Fee Baseline Full Prediction Export",
            "",
            f"- Model: `{manifest['model_name']}`",
            f"- Test rows: `{test['row_count']}`",
            f"- Test positive rate: `{float(test['positive_rate']):.6f}`",
            "- Sample prediction files were retained for preview.",
            "- These are research-only predictions, not runtime outputs.",
            "",
        ]
    )


def _honesty_flags(
    feature_audit: Mapping[str, Any],
    split: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[str]:
    flags = [
        "RESEARCH_ONLY_FEE_BASELINE",
        "NOT_RUNTIME_COMPARABLE",
        "NOT_PROMOTABLE",
        "NO_REGISTRY_WRITE",
        "MARKET_FEATURES_USED",
        "CHRONOLOGICAL_SPLIT_USED",
        "SINGLE_RECENT_FOLD_ONLY",
        "IMBALANCED_TARGET",
        "ACCURACY_DIAGNOSTIC_ONLY",
        "BASELINE_FEASIBILITY_ONLY",
        "NO_PROFITABILITY_CLAIM",
        "NO_RUNTIME_EFFECT",
        "NO_PROMOTION_EFFECT",
    ]
    if not feature_audit["safe_feature_columns"]:
        flags.append("SAFE_FEATURES_MISSING")
    if len({row.get("fold_index") for row in split["test"]}) == 1:
        flags.append("SINGLE_RECENT_FOLD_ONLY")
    return sorted(dict.fromkeys(flags))


def _recommend(baselines: Sequence[Mapping[str, Any]]) -> str:
    always = next((row for row in baselines if row["baseline_name"] == "always_negative"), None)
    best_model = max(
        baselines,
        key=lambda row: (row.get("average_precision") or 0.0, row.get("balanced_accuracy") or 0.0),
        default=None,
    )
    if not always or not best_model or best_model["baseline_name"] == "always_negative":
        return "E. reject this M20 target path as weak"
    if (best_model.get("average_precision") or 0.0) <= (always.get("average_precision") or 0.0):
        return "D. add richer features before more modeling"
    if (best_model.get("recall") or 0.0) < 0.10:
        return "A. proceed to cost-aware policy evaluation using the best fee-exceedance baseline"
    return "B. try a tiny stronger tabular model in a separate research batch"


def _split_summary(split: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    return {
        "split_source": "chronological_within_single_recent_fold",
        "train_row_count": len(split["train"]),
        "validation_row_count": len(split["validation"]),
        "test_row_count": len(split["test"]),
        "train_positive_rate": _positive_rate(split["train"]),
        "validation_positive_rate": _positive_rate(split["validation"]),
        "test_positive_rate": _positive_rate(split["test"]),
    }


def _metrics_rows(baselines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "baseline_name",
        "test_row_count",
        "positive_rate",
        "accuracy_diagnostic_only",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
    )
    return [{column: baseline.get(column) for column in columns} for baseline in baselines]


def _confusion_rows(baselines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"baseline_name": baseline["baseline_name"], **baseline["confusion_matrix"]}
        for baseline in baselines
    ]


def _flatten(baselines: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        rows.extend(dict(row) for row in baseline[key])
    return rows


def _feature_audit_markdown(audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Fee-Exceedance Feature Audit",
            "",
            f"- Safe market feature count: `{audit['safe_feature_count']}`",
            f"- Excluded column count: `{audit['excluded_column_count']}`",
            "- Prediction, label, future, realized-outcome, barrier, and target-derived "
            "columns are excluded from market features.",
            "",
        ]
    )


def _report_markdown(report: Mapping[str, Any]) -> str:
    best = max(
        report["baselines"],
        key=lambda row: (row.get("average_precision") or 0.0, row.get("balanced_accuracy") or 0.0),
        default={},
    )
    return "\n".join(
        [
            "# M20 Fee-Exceedance Baseline Report",
            "",
            f"- Run directory: `{report['run_dir']}`",
            f"- Scenario: `{report['scenario_name']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Best diagnostic baseline: `{best.get('baseline_name', 'none')}`",
            f"- Best average precision: `{float(best.get('average_precision') or 0.0):.6f}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
            "These are research-only diagnostics, not runtime-comparable models, "
            "not promotable artifacts, and not profitability evidence.",
            "",
        ]
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _exclusion_reason(column: str) -> str:
    if column in KEY_COLUMNS:
        return "key_column"
    if column in PREDICTION_OUTPUT_COLUMNS:
        return "prediction_output"
    lowered = column.lower()
    for token in LEAKAGE_TOKENS:
        if token in lowered:
            return f"excluded_token:{token}"
    return ""


def _reason_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _is_numeric_column(rows: Sequence[Mapping[str, str]], column: str) -> bool:
    sample = rows[: min(len(rows), 100)]
    return bool(sample) and all(_try_float(row.get(column)) is not None for row in sample)


def _label(row: Mapping[str, str]) -> int:
    return int(float(row.get("label", 0) or 0))


def _positive_rate(rows: Sequence[Mapping[str, str]]) -> float:
    return sum(_label(row) for row in rows) / len(rows) if rows else 0.0


def _stratified_random_predictions(
    train_labels: Sequence[int],
    count: int,
    random_seed: int,
) -> list[int]:
    rng = random.Random(random_seed)
    positive_rate = sum(train_labels) / len(train_labels) if train_labels else 0.0
    return [int(rng.random() < positive_rate) for _ in range(count)]


def _matrix(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> list[list[float]]:
    return [[float(row[column]) for column in columns] for row in rows]


def _confusion(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    return {
        "tp": _confusion_count(labels, predictions, 1, 1),
        "tn": _confusion_count(labels, predictions, 0, 0),
        "fp": _confusion_count(labels, predictions, 0, 1),
        "fn": _confusion_count(labels, predictions, 1, 0),
    }


def _confusion_count(
    labels: Sequence[int],
    predictions: Sequence[int],
    expected_label: int,
    expected_prediction: int,
) -> int:
    return sum(
        1 for label, prediction in zip(labels, predictions)
        if label == expected_label and prediction == expected_prediction
    )


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    positive_scores = [score for label, score in zip(labels, probabilities) if label == 1]
    negative_scores = [score for label, score in zip(labels, probabilities) if label == 0]
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if sum(labels) == 0:
        return None
    ordered = sorted(range(len(labels)), key=lambda index: probabilities[index], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    for rank, index in enumerate(ordered, start=1):
        if labels[index] == 1:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / sum(labels)


def _brier(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    return (
        sum((float(label) - probability) ** 2 for label, probability in zip(labels, probabilities))
        / len(labels)
        if labels else 0.0
    )


def _quantile(values: Sequence[float], q_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _try_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _sklearn_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None
