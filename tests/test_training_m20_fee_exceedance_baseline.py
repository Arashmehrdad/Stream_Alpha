"""Focused tests for M20 fee-exceedance research baselines."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_fee_exceedance_baseline import train_fee_exceedance_baselines

# pylint: disable=missing-function-docstring


def _write_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    frame_dir = run_dir / "training_frame"
    label_dir = run_dir / "research_labels" / "vol_scaled"
    frame_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = [
        {
            "symbol": row["symbol"],
            "interval_begin": row["interval_begin"],
            "fold_index": row["fold_index"],
            "row_id": row["row_id"],
            "open_price": row["open_price"],
            "close_price": row["close_price"],
            "volume": row["volume"],
            "realized_vol_12": row["realized_vol_12"],
            "future_return_3": row.get("future_return_3", 0.01),
            "prob_up": row.get("prob_up", 0.5),
        }
        for row in rows
    ]
    label_rows = [
        {
            "symbol": row["symbol"],
            "interval_begin": row["interval_begin"],
            "fold_index": row["fold_index"],
            "row_id": row["row_id"],
            "label": row["label"],
            "scenario_name": "current_fee",
        }
        for row in rows
    ]
    _write_csv(frame_dir / "m20_training_frame_features.csv", feature_rows)
    _write_csv(
        frame_dir / "m20_training_frame_keys.csv",
        [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "row_id": row["row_id"],
            }
            for row in rows
        ],
    )
    _write_csv(label_dir / "fee_exceedance_labels_vol_scaled.csv", label_rows)


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


def _rows(count: int = 60) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        symbol = "ETH/USD" if index % 2 else "BTC/USD"
        label = int(index % 5 == 0 or index > count * 0.85)
        rows.append(
            {
                "symbol": symbol,
                "interval_begin": f"2026-04-01T{index // 60:02d}:{index % 60:02d}:00Z",
                "fold_index": 4,
                "row_id": f"{symbol}|{index}",
                "open_price": 100.0 + index,
                "close_price": 100.0 + index + (2.0 if label else -0.2),
                "volume": 10.0 + index,
                "realized_vol_12": 0.001 + index / 100000.0,
                "label": label,
            }
        )
    return rows


def test_joins_feature_frame_and_fee_labels_by_symbol_interval(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = train_fee_exceedance_baselines(run_dir=run_dir)

    assert report["row_count"] == 60
    assert report["feature_audit"]["safe_feature_count"] == 4


def test_excludes_leakage_target_future_and_output_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = train_fee_exceedance_baselines(run_dir=run_dir)

    reasons = report["feature_audit"]["exclusion_reason_counts"]
    assert "excluded_token:future" in reasons
    assert "prediction_output" in reasons
    assert "key_column" in reasons


def test_chronological_split_is_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, list(reversed(_rows())))

    report = train_fee_exceedance_baselines(run_dir=run_dir)

    assert report["split"]["split_source"] == "chronological_within_single_recent_fold"
    assert report["split"]["train_row_count"] == 36
    assert report["split"]["validation_row_count"] == 12
    assert report["split"]["test_row_count"] == 12


def test_always_negative_and_random_baselines_are_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    first = train_fee_exceedance_baselines(run_dir=run_dir)
    second = train_fee_exceedance_baselines(run_dir=run_dir)

    first_csv = Path(first["output_files"]["fee_baseline_metrics_csv"]).read_text(
        encoding="utf-8"
    )
    second_csv = Path(second["output_files"]["fee_baseline_metrics_csv"]).read_text(
        encoding="utf-8"
    )
    assert first_csv == second_csv
    assert "always_negative" in first_csv
    assert "stratified_random_seed_1729" in first_csv


def test_logistic_regression_runs_when_sklearn_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows(90))

    report = train_fee_exceedance_baselines(run_dir=run_dir)
    names = {baseline["baseline_name"] for baseline in report["baselines"]}

    assert "logistic_regression_tiny" in names


def test_metrics_threshold_topk_symbol_and_reports_are_written(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows())

    report = train_fee_exceedance_baselines(run_dir=run_dir)
    baseline_dir = Path(report["baseline_dir"])

    assert (baseline_dir / "fee_baseline_manifest.json").exists()
    assert (baseline_dir / "fee_feature_audit.json").exists()
    assert (baseline_dir / "fee_threshold_sweep.csv").exists()
    assert (baseline_dir / "fee_topk_diagnostics.csv").exists()
    assert (baseline_dir / "fee_by_symbol_metrics.csv").exists()
    assert (baseline_dir / "fee_calibration_buckets.csv").exists()
    assert "average_precision" in report["baselines"][0]
    assert report["recommendation"]


def test_recommendation_handles_weak_no_edge_case(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = _rows()
    for row in rows:
        row["label"] = 0
    _write_run(run_dir, rows)

    report = train_fee_exceedance_baselines(run_dir=run_dir)

    assert report["recommendation"] == "E. reject this M20 target path as weak"


def test_full_prediction_export_writes_all_split_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, _rows(90))

    report = train_fee_exceedance_baselines(
        run_dir=run_dir,
        export_full_predictions=True,
    )
    baseline_dir = Path(report["baseline_dir"])

    assert len(
        _read_csv(baseline_dir / "predictions_logistic_regression_tiny_test_full.csv")
    ) == 18
    assert len(
        _read_csv(baseline_dir / "predictions_logistic_regression_tiny_validation_full.csv")
    ) == 18
    assert len(
        _read_csv(baseline_dir / "predictions_logistic_regression_tiny_train_full.csv")
    ) == 54
    assert (baseline_dir / "predictions_logistic_regression_tiny.csv").exists()
    manifest = (baseline_dir / "prediction_export_manifest.json").read_text(encoding="utf-8")
    assert "FULL_TEST_PREDICTIONS_EXPORTED" in manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]
