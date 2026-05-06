"""Focused tests for M20 conditional usefulness diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_conditional_usefulness import analyze_conditional_usefulness

# pylint: disable=missing-function-docstring


def _write_run(
    run_dir: Path,
    rows: list[dict[str, object]],
    *,
    probabilities: bool = True,
    full_predictions: bool = False,
) -> None:
    frame_dir = run_dir / "training_frame"
    label_dir = run_dir / "research_labels" / "vol_scaled"
    baseline_dir = label_dir / "fee_exceedance_baselines"
    frame_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame_dir / "m20_training_frame_features.csv", rows)
    _write_csv(
        label_dir / "fee_exceedance_labels_vol_scaled.csv",
        [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "row_id": row["row_id"],
                "label": row["label"],
                "scenario_name": "current_fee",
            }
            for row in rows
        ],
    )
    prediction_rows = []
    for row in rows:
        prediction = {
            "baseline_name": "logistic_regression_tiny",
            "symbol": row["symbol"],
            "interval_begin": row["interval_begin"],
            "row_id": row["row_id"],
            "label": row["label"],
            "prediction": int(float(row["probability"]) >= 0.5),
        }
        if probabilities:
            prediction["probability"] = row["probability"]
        prediction_rows.append(prediction)
    prediction_path = baseline_dir / "predictions_logistic_regression_tiny.csv"
    _write_csv(prediction_path, prediction_rows)
    output_files = {
        "predictions_logistic_regression_tiny_csv": str(prediction_path),
    }
    if full_predictions:
        full_path = baseline_dir / "predictions_logistic_regression_tiny_test_full.csv"
        full_rows = [
            {**row, "split": "test", "model_name": "logistic_regression_tiny"}
            for row in prediction_rows
        ]
        _write_csv(full_path, full_rows)
        output_files["predictions_logistic_regression_tiny_test_full_csv"] = str(full_path)
    manifest = {
        "output_files": output_files
    }
    (baseline_dir / "fee_baseline_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (baseline_dir / "fee_baseline_metrics.json").write_text(
        json.dumps(
            {
                "baselines": [
                    {
                        "baseline_name": "logistic_regression_tiny",
                        "positive_rate": 0.2,
                        "average_precision": 0.4,
                        "roc_auc": 0.7,
                        "balanced_accuracy": 0.6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(count: int = 1200) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        label = int(index % 5 == 0)
        probability = 0.9 if label and index < 200 else (0.6 if label else 0.1)
        rows.append(
            {
                "symbol": "BTC/USD" if index % 2 else "ETH/USD",
                "interval_begin": f"2026-04-{(index // 288) + 1:02d}T{index % 24:02d}:00:00Z",
                "fold_index": 4,
                "row_id": f"row-{index}",
                "label": label,
                "prediction": int(probability >= 0.5),
                "probability": probability,
                "realized_vol_12": 0.001 + (index % 100) / 10000.0,
                "log_return_1": -0.01 if index % 3 == 0 else 0.01,
                "rsi_14": 20 if index % 7 == 0 else (80 if index % 11 == 0 else 50),
                "macd_line_12_26": -1.0 if index % 3 == 0 else 1.0,
                "volume": 10 + index,
                "high_price": 101.0 + index,
                "low_price": 99.0 + index,
                "close_price": 100.0 + index,
            }
        )
    return rows


def test_joins_predictions_labels_features_and_focuses_test(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = analyze_conditional_usefulness(run_dir=run_dir)

    assert report["prediction_rows_analyzed"] == 1200
    assert "TEST_SPLIT_PRIMARY" in report["honesty_flags"]
    assert (Path(report["conditional_dir"]) / "conditional_usefulness_by_slice.csv").exists()


def test_creates_condition_buckets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = analyze_conditional_usefulness(run_dir=run_dir)
    definitions = report["bucket_definitions"]

    assert "volatility" in definitions
    assert "rsi" in definitions
    assert "macd" in definitions
    assert "range" in definitions


def test_slice_metrics_and_enable_classification(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = analyze_conditional_usefulness(run_dir=run_dir)

    assert report["search_breadth"]["slice_count"] > 0
    assert (
        report["search_breadth"]["enable_candidate_count"]
        + report["search_breadth"]["watchlist_candidate_count"]
        + report["search_breadth"]["disable_candidate_count"]
        > 0
    )


def test_single_class_slice_marks_metric_undefined(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = _rows(2400)
    for row in rows:
        if row["symbol"] == "BTC/USD":
            row["label"] = 0
            row["probability"] = 0.1
    _write_run(run_dir, rows)

    report = analyze_conditional_usefulness(run_dir=run_dir)
    rows_by_slice = _read_csv(Path(report["output_files"]["conditional_usefulness_by_slice_csv"]))

    assert any(row["classification"] == "METRIC_UNDEFINED" for row in rows_by_slice)


def test_low_sample_and_low_positive_classification(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = _rows(900)
    for row in rows:
        row["label"] = int(str(row["row_id"]).endswith("0"))
    _write_run(run_dir, rows)

    report = analyze_conditional_usefulness(run_dir=run_dir)
    classifications = {
        row["classification"]
        for row in _read_csv(Path(report["output_files"]["conditional_usefulness_by_slice_csv"]))
    }

    assert "INSUFFICIENT_SAMPLE" in classifications or "INSUFFICIENT_POSITIVES" in classifications


def test_blocks_when_prediction_probabilities_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows(), probabilities=False)

    with pytest.raises(ValueError, match="CONDITIONAL_ANALYSIS_BLOCKED_MISSING_PREDICTIONS"):
        analyze_conditional_usefulness(run_dir=run_dir)


def test_prefers_full_test_predictions_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows(), full_predictions=True)

    sample = analyze_conditional_usefulness(run_dir=run_dir, prediction_source="sampled-test")
    full = analyze_conditional_usefulness(run_dir=run_dir, prediction_source="full-test")

    assert sample["prediction_source"] == "sampled_test"
    assert full["prediction_source"] == "full_test"
    assert str(full["conditional_dir"]).endswith("conditional_usefulness_full_test")
    assert "FULL_TEST_PREDICTIONS_USED" in full["honesty_flags"]
    assert "full_vs_sample_conditional_comparison_json" in full["output_files"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]
