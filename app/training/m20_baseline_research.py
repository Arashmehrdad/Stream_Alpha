"""Tiny research-only baselines for M20 vol-scaled labels."""

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
DEFAULT_TEST_FRACTION = 0.20
LABEL_FILE = "triple_barrier_labels_vol_scaled.csv"
BASELINE_DIR_NAME = "baselines"
LABEL_ORDER = (-1, 0, 1)
LEAKAGE_TOKENS = (
    "label",
    "future",
    "return",
    "gross",
    "net",
    "barrier",
    "hit",
    "horizon",
    "target",
    "outcome",
)
EXACT_LEAKAGE_COLUMNS = {"y_true"}
IDENTIFIER_COLUMNS = {
    "row_id",
    "source_index",
    "event_end_index",
    "event_end_row_id",
    "model_name",
    "fold_index",
    "symbol",
    "interval_begin",
    "as_of_time",
    "regime_label",
    "label_source",
}
SCORE_COLUMNS = {"prob_up", "confidence", "y_pred"}


def train_completed_run_baselines(
    *,
    run_dir: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Train/evaluate tiny research-only baselines for a completed M20 run."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    label_path = resolved_run_dir / "research_labels" / "vol_scaled" / LABEL_FILE
    label_rows = _read_csv_rows(label_path)
    if not label_rows:
        raise ValueError(f"Missing or empty vol-scaled label file: {label_path}")
    oof_rows = _read_oof_winner_rows(resolved_run_dir, label_rows)
    merged_rows = _merge_rows(label_rows, oof_rows)
    feature_audit = _build_feature_audit(merged_rows)
    split = _split_rows(merged_rows)
    baselines = _evaluate_baselines(
        train_rows=split["train_rows"],
        test_rows=split["test_rows"],
        feature_columns=feature_audit["safe_numeric_feature_columns"],
        score_columns=feature_audit["score_feature_columns"],
        random_seed=random_seed,
    )
    honesty_flags = _build_honesty_flags(feature_audit, split, baselines)
    recommendation = _recommend(honesty_flags, baselines)
    baseline_dir = resolved_run_dir / "research_labels" / "vol_scaled" / BASELINE_DIR_NAME
    baseline_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "baseline_manifest_json": str(baseline_dir / "baseline_manifest.json"),
        "feature_audit_json": str(baseline_dir / "feature_audit.json"),
        "feature_audit_md": str(baseline_dir / "feature_audit.md"),
        "baseline_metrics_json": str(baseline_dir / "baseline_metrics.json"),
        "baseline_metrics_csv": str(baseline_dir / "baseline_metrics.csv"),
        "baseline_confusion_matrix_csv": str(baseline_dir / "baseline_confusion_matrix.csv"),
        "baseline_by_slice_csv": str(baseline_dir / "baseline_by_slice.csv"),
        "baseline_report_md": str(baseline_dir / "baseline_report.md"),
    }
    prediction_files = _write_prediction_files(baseline_dir, baselines)
    output_files.update(prediction_files)
    report = {
        "run_dir": str(resolved_run_dir),
        "baseline_dir": str(baseline_dir),
        "label_path": str(label_path),
        "row_count": len(merged_rows),
        "split": _split_summary(split),
        "feature_audit": feature_audit,
        "baselines": [
            {key: value for key, value in baseline.items() if key != "predictions"}
            for baseline in baselines
        ],
        "honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "notes": [
            "Research-only tiny baseline feasibility diagnostics.",
            "Not comparable to runtime incumbent unless target semantics and timestamps match.",
            "No runtime inference, registry authority, promotion, execution, NeuralForecast, "
            "or roster behavior changed.",
        ],
        "output_files": output_files,
    }
    manifest = {
        "run_dir": str(resolved_run_dir),
        "baseline_dir": str(baseline_dir),
        "source_label_file": str(label_path),
        "honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["baseline_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["feature_audit_json"]), feature_audit)
    Path(output_files["feature_audit_md"]).write_text(
        _build_feature_audit_markdown(feature_audit),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["baseline_metrics_json"]), report)
    write_csv_artifact(Path(output_files["baseline_metrics_csv"]), _metrics_csv_rows(baselines))
    write_csv_artifact(
        Path(output_files["baseline_confusion_matrix_csv"]),
        _confusion_csv_rows(baselines),
    )
    write_csv_artifact(Path(output_files["baseline_by_slice_csv"]), _slice_csv_rows(baselines))
    Path(output_files["baseline_report_md"]).write_text(
        _build_report_markdown(report),
        encoding="utf-8",
    )
    return make_json_safe({**report, "manifest": manifest})


def _build_feature_audit(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns = [
        column for column in columns
        if all(_try_float(row.get(column)) is not None for row in rows[: min(len(rows), 100)])
    ]
    excluded_columns = [
        column for column in numeric_columns
        if _is_leakage_column(column) or column in IDENTIFIER_COLUMNS
    ]
    safe_columns = [
        column for column in numeric_columns
        if column not in excluded_columns and column not in IDENTIFIER_COLUMNS
    ]
    score_columns = [column for column in safe_columns if column in SCORE_COLUMNS]
    safe_non_score = [column for column in safe_columns if column not in SCORE_COLUMNS]
    flags: list[str] = []
    if not safe_columns:
        flags.append("SAFE_FEATURES_MISSING")
    if safe_columns and not safe_non_score:
        flags.append("SCORE_ONLY_FEATURES")
    if safe_columns:
        flags.append("FEATURE_SOURCE_LIMITED")
    return {
        "available_columns": columns,
        "numeric_columns": numeric_columns,
        "excluded_columns": excluded_columns,
        "safe_numeric_feature_columns": safe_columns,
        "score_feature_columns": score_columns,
        "safe_non_score_feature_columns": safe_non_score,
        "feature_flags": sorted(dict.fromkeys(flags)),
    }


def _split_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    fold_values = sorted(
        {
            int(float(row["fold_index"]))
            for row in rows
            if _try_float(row.get("fold_index")) is not None
        }
    )
    if len(fold_values) >= 2:
        test_fold = fold_values[-1]
        train_rows = [
            dict(row) for row in rows if int(float(row["fold_index"])) != test_fold
        ]
        test_rows = [
            dict(row) for row in rows if int(float(row["fold_index"])) == test_fold
        ]
        return {"split_source": "fold_index", "train_rows": train_rows, "test_rows": test_rows}
    if any(row.get("interval_begin") for row in rows):
        ordered = sorted(
            rows,
            key=lambda row: (
                str(row.get("interval_begin", "")),
                str(row.get("row_id", "")),
            ),
        )
        split_index = max(1, int(len(ordered) * (1.0 - DEFAULT_TEST_FRACTION)))
        return {
            "split_source": "chronological_interval_begin",
            "train_rows": [dict(row) for row in ordered[:split_index]],
            "test_rows": [dict(row) for row in ordered[split_index:]],
        }
    return {"split_source": "missing", "train_rows": [], "test_rows": []}


def _evaluate_baselines(
    *,
    train_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
    score_columns: Sequence[str],
    random_seed: int,
) -> list[dict[str, Any]]:
    if not train_rows or not test_rows:
        return []
    train_labels = [_label(row) for row in train_rows]
    test_labels = [_label(row) for row in test_rows]
    baselines = [
        _baseline_result(
            "majority_class",
            test_rows,
            test_labels,
            [_majority_label(train_labels) for _ in test_rows],
        ),
        _baseline_result(
            f"stratified_random_seed_{random_seed}",
            test_rows,
            test_labels,
            _stratified_random_predictions(train_labels, len(test_rows), random_seed),
        ),
    ]
    if score_columns:
        baselines.append(_score_only_baseline(test_rows, test_labels, score_columns[0]))
    if feature_columns and _sklearn_available():
        logistic = _logistic_regression_baseline(
            train_rows,
            test_rows,
            train_labels,
            test_labels,
            feature_columns,
        )
        if logistic is not None:
            baselines.append(logistic)
    return baselines


def _logistic_regression_baseline(
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
    except Exception:
        return None
    train_x = [[float(row[column]) for column in feature_columns] for row in train_rows]
    test_x = [[float(row[column]) for column in feature_columns] for row in test_rows]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, random_state=0, class_weight="balanced"),
    )
    model.fit(train_x, train_labels)
    predictions = [int(value) for value in model.predict(test_x)]
    result = _baseline_result(
        "logistic_regression_tiny",
        test_rows,
        test_labels,
        predictions,
    )
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_x)
        classes = [int(value) for value in model.classes_]
        result["calibration"] = _calibration_metrics(test_labels, probabilities, classes)
    return result


def _score_only_baseline(
    test_rows: Sequence[Mapping[str, str]],
    test_labels: Sequence[int],
    score_column: str,
) -> dict[str, Any]:
    predictions = []
    for row in test_rows:
        score = float(row.get(score_column, 0.0))
        if score >= 2.0 / 3.0:
            predictions.append(1)
        elif score <= 1.0 / 3.0:
            predictions.append(-1)
        else:
            predictions.append(0)
    return _baseline_result(f"score_only_{score_column}", test_rows, test_labels, predictions)


def _baseline_result(
    name: str,
    test_rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, Any]:
    metrics = _classification_metrics(labels, predictions)
    return {
        "baseline_name": name,
        "metrics": metrics,
        "confusion_matrix": _confusion_matrix(labels, predictions),
        "by_slice": _slice_metrics(test_rows, labels, predictions),
        "predictions": [
            {
                "row_id": row.get("row_id", index),
                "label": labels[index],
                "prediction": predictions[index],
            }
            for index, row in enumerate(test_rows)
        ],
    }


def _classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    return {
        "row_count": len(labels),
        "accuracy": _safe_ratio(sum(1 for a, p in zip(labels, predictions) if a == p), len(labels)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "macro_f1": _macro_f1(labels, predictions),
        "positive_precision": _precision(labels, predictions, 1),
        "positive_recall": _recall(labels, predictions, 1),
        "negative_precision": _precision(labels, predictions, -1),
        "negative_recall": _recall(labels, predictions, -1),
        "neutral_recall": _recall(labels, predictions, 0),
        "coverage": 1.0,
    }


def _slice_metrics(
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[int],
    predictions: Sequence[int],
) -> list[dict[str, Any]]:
    output = []
    for column in ("fold_index", "symbol", "regime_label"):
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            if row.get(column):
                groups.setdefault(str(row[column]), []).append(index)
        for value, indices in sorted(groups.items()):
            slice_labels = [labels[index] for index in indices]
            slice_predictions = [predictions[index] for index in indices]
            output.append(
                {
                    "slice_column": column,
                    "slice_value": value,
                    **_classification_metrics(slice_labels, slice_predictions),
                }
            )
    return output


def _build_honesty_flags(
    feature_audit: Mapping[str, Any],
    split: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
) -> list[str]:
    flags = [
        "RESEARCH_ONLY_BASELINE",
        "NOT_RUNTIME_COMPARABLE",
        "NOT_PROMOTABLE",
        "VOLATILITY_PROXY_LABELS",
        "BASELINE_FEASIBILITY_ONLY",
    ]
    flags.extend(feature_audit["feature_flags"])
    if split["split_source"] == "missing":
        flags.append("SPLIT_SOURCE_MISSING")
    if baselines:
        majority = _find_baseline(baselines, "majority_class")
        best = max(baselines, key=lambda row: float(row["metrics"]["balanced_accuracy"]))
        if majority and best["baseline_name"] == "majority_class":
            flags.append("NO_MODEL_EDGE_OVER_MAJORITIY")
        if any(float(row["metrics"]["positive_recall"]) < 0.10 for row in baselines):
            flags.append("LOW_CLASS_RECALL_POSITIVE")
        if any(float(row["metrics"]["negative_recall"]) < 0.10 for row in baselines):
            flags.append("LOW_CLASS_RECALL_NEGATIVE")
        label_counts = _counts([_label(row) for row in split["train_rows"] + split["test_rows"]])
        if _safe_ratio(label_counts.get(0, 0), sum(label_counts.values())) > 0.50:
            flags.append("NEUTRAL_DOMINANT_TARGET")
    return sorted(dict.fromkeys(flags))


def _recommend(flags: Sequence[str], baselines: Sequence[Mapping[str, Any]]) -> str:
    if "SAFE_FEATURES_MISSING" in flags or "SCORE_ONLY_FEATURES" in flags:
        return "B. add/export row-aligned feature columns before further modeling"
    if not baselines:
        return "E. stop M20 model chase and package as negative research"
    majority = _find_baseline(baselines, "majority_class")
    best = max(baselines, key=lambda row: float(row["metrics"]["balanced_accuracy"]))
    if majority and best["baseline_name"] != "majority_class":
        delta = float(best["metrics"]["balanced_accuracy"]) - float(
            majority["metrics"]["balanced_accuracy"]
        )
        if delta > 0.02:
            return (
                "A. proceed to tiny cost-aware policy evaluation using "
                "the best research baseline"
            )
    if "NO_MODEL_EDGE_OVER_MAJORITIY" in flags:
        return "C. reject current volatility-scaled target as not learnable by tiny baselines"
    return "D. test fee-exceedance baseline next"


def _write_prediction_files(
    baseline_dir: Path,
    baselines: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_files = {}
    for baseline in baselines:
        safe_name = str(baseline["baseline_name"]).replace(" ", "_").replace("/", "_")
        path = baseline_dir / f"predictions_{safe_name}.csv"
        write_csv_artifact(path, list(baseline["predictions"]))
        output_files[f"predictions_{safe_name}_csv"] = str(path)
    return output_files


def _metrics_csv_rows(baselines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"baseline_name": baseline["baseline_name"], **baseline["metrics"]}
        for baseline in baselines
    ]


def _confusion_csv_rows(baselines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for baseline in baselines:
        for actual, predicted_counts in baseline["confusion_matrix"].items():
            for predicted, count in predicted_counts.items():
                rows.append(
                    {
                        "baseline_name": baseline["baseline_name"],
                        "actual": actual,
                        "predicted": predicted,
                        "count": count,
                    }
                )
    return rows


def _slice_csv_rows(baselines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for baseline in baselines:
        for row in baseline["by_slice"]:
            rows.append({"baseline_name": baseline["baseline_name"], **row})
    return rows


def _build_feature_audit_markdown(feature_audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Baseline Feature Audit",
            "",
            f"- Safe numeric features: `{feature_audit['safe_numeric_feature_columns']}`",
            f"- Excluded columns: `{feature_audit['excluded_columns']}`",
            f"- Feature flags: `{feature_audit['feature_flags']}`",
            "",
        ]
    )


def _build_report_markdown(report: Mapping[str, Any]) -> str:
    best = max(
        report["baselines"],
        key=lambda row: float(row["metrics"]["balanced_accuracy"]),
        default=None,
    )
    return "\n".join(
        [
            "# M20 Vol-Scaled Tiny Baseline Report",
            "",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Split source: `{report['split']['split_source']}`",
            (
                f"- Best diagnostic baseline: `{best['baseline_name']}` "
                f"(balanced_accuracy={best['metrics']['balanced_accuracy']:.6f})"
                if best
                else "- Best diagnostic baseline: `none`"
            ),
            "",
            "These baselines are research-only feasibility diagnostics. They are not "
            "runtime-comparable, not promotable, and not written to the model registry.",
            "",
        ]
    )


def _read_oof_winner_rows(
    run_dir: Path,
    label_rows: Sequence[Mapping[str, str]],
) -> dict[str, Mapping[str, str]]:
    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.exists():
        return {}
    label_ids = {str(row.get("row_id")) for row in label_rows}
    model_name = str(label_rows[0].get("model_name", "") or "") if label_rows else ""
    output: dict[str, Mapping[str, str]] = {}
    with oof_path.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            row_id = str(row.get("row_id"))
            model_matches = not model_name or str(row.get("model_name", "")) == model_name
            if row_id in label_ids and model_matches:
                output[row_id] = dict(row)
    return output


def _merge_rows(
    label_rows: Sequence[Mapping[str, str]],
    oof_by_row_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    merged_rows = []
    for label_row in label_rows:
        row_id = str(label_row.get("row_id"))
        merged = {**dict(oof_by_row_id.get(row_id, {})), **dict(label_row)}
        merged_rows.append(merged)
    return merged_rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _sklearn_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None


def _is_leakage_column(column: str) -> bool:
    lowered = column.lower()
    if lowered in EXACT_LEAKAGE_COLUMNS:
        return True
    return any(token in lowered for token in LEAKAGE_TOKENS)


def _label(row: Mapping[str, str]) -> int:
    return int(float(row["label"]))


def _try_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _majority_label(labels: Sequence[int]) -> int:
    return sorted(set(labels), key=lambda value: (-labels.count(value), value))[0]


def _stratified_random_predictions(
    train_labels: Sequence[int],
    count: int,
    random_seed: int,
) -> list[int]:
    rng = random.Random(random_seed)
    return [rng.choice(list(train_labels)) for _ in range(count)]


def _confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, dict[str, int]]:
    return {
        str(actual): {
            str(predicted): sum(
                1 for label, prediction in zip(labels, predictions)
                if label == actual and prediction == predicted
            )
            for predicted in LABEL_ORDER
        }
        for actual in LABEL_ORDER
    }


def _precision(labels: Sequence[int], predictions: Sequence[int], class_label: int) -> float:
    true_positive = sum(
        1
        for label, prediction in zip(labels, predictions)
        if label == class_label and prediction == class_label
    )
    predicted_positive = sum(1 for prediction in predictions if prediction == class_label)
    return _safe_ratio(true_positive, predicted_positive)


def _recall(labels: Sequence[int], predictions: Sequence[int], class_label: int) -> float:
    true_positive = sum(
        1
        for label, prediction in zip(labels, predictions)
        if label == class_label and prediction == class_label
    )
    actual_positive = sum(1 for label in labels if label == class_label)
    return _safe_ratio(true_positive, actual_positive)


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    scores = []
    for class_label in LABEL_ORDER:
        precision = _precision(labels, predictions, class_label)
        recall = _recall(labels, predictions, class_label)
        scores.append(_safe_ratio(2 * precision * recall, precision + recall))
    return sum(scores) / len(scores)


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    recalls = [_recall(labels, predictions, class_label) for class_label in LABEL_ORDER]
    return sum(recalls) / len(recalls)


def _calibration_metrics(
    labels: Sequence[int],
    probabilities: Any,
    classes: Sequence[int],
) -> dict[str, float]:
    brier_sum = 0.0
    log_loss_sum = 0.0
    epsilon = 1e-15
    class_index = {class_label: index for index, class_label in enumerate(classes)}
    for row_index, label in enumerate(labels):
        for class_label in LABEL_ORDER:
            probability = float(probabilities[row_index][class_index.get(class_label, 0)])
            target = 1.0 if label == class_label else 0.0
            brier_sum += (probability - target) ** 2
        true_probability = max(
            min(float(probabilities[row_index][class_index[label]]), 1.0 - epsilon),
            epsilon,
        )
        log_loss_sum += -math.log(true_probability)
    return {
        "brier_multiclass": brier_sum / len(labels),
        "log_loss": log_loss_sum / len(labels),
    }


def _find_baseline(
    baselines: Sequence[Mapping[str, Any]],
    baseline_name: str,
) -> Mapping[str, Any] | None:
    for baseline in baselines:
        if baseline["baseline_name"] == baseline_name:
            return baseline
    return None


def _counts(labels: Sequence[int]) -> dict[int, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}


def _split_summary(split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split_source": split["split_source"],
        "train_row_count": len(split["train_rows"]),
        "test_row_count": len(split["test_rows"]),
        "train_class_counts": _counts([_label(row) for row in split["train_rows"]]),
        "test_class_counts": _counts([_label(row) for row in split["test_rows"]]),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
