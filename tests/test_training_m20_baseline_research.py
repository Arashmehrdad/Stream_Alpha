"""Focused tests for tiny M20 research baselines."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_baseline_research import train_completed_run_baselines

# pylint: disable=missing-function-docstring,too-many-arguments


def _write_run(
    run_dir: Path,
    label_rows: list[dict[str, object]],
    oof_rows: list[dict[str, object]] | None = None,
) -> None:
    label_dir = run_dir / "research_labels" / "vol_scaled"
    label_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(label_dir / "triple_barrier_labels_vol_scaled.csv", label_rows)
    if oof_rows is not None:
        _write_csv(run_dir / "oof_predictions.csv", oof_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames or ["label"])
        writer.writeheader()
        writer.writerows(rows)


def _label_row(
    row_id: str,
    label: int,
    *,
    fold_index: int = 0,
    symbol: str = "BTC/USD",
    minute: int = 0,
    volatility: float = 0.002,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "label": label,
        "fold_index": fold_index,
        "symbol": symbol,
        "interval_begin": f"2026-04-01T00:{minute:02d}:00Z",
        "regime_label": "RANGE",
        "volatility": volatility,
        "future_return": 0.01,
        "barrier_hit": "upper",
    }


def _oof_row(row_id: str, *, prob_up: float) -> dict[str, object]:
    return {
        "model_name": "m",
        "row_id": row_id,
        "prob_up": prob_up,
        "confidence": prob_up,
        "y_pred": int(prob_up >= 0.5),
        "future_return_3": 0.01,
    }


def test_feature_audit_excludes_leakage_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(str(index), [-1, 0, 1, 1, 0, -1][index], minute=index)
        for index in range(6)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)

    excluded = report["feature_audit"]["excluded_columns"]
    assert "label" in excluded
    assert "future_return" in excluded
    assert "volatility" in report["feature_audit"]["safe_numeric_feature_columns"]


def test_feature_audit_detects_score_only_features(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        {
            k: v
            for k, v in _label_row(
                str(index),
                [-1, 0, 1, 1, 0, -1][index],
                minute=index,
            ).items()
            if k != "volatility"
        }
        for index in range(6)
    ]
    oof_rows = [_oof_row(str(index), prob_up=index / 10.0) for index in range(6)]
    _write_run(run_dir, labels, oof_rows)

    report = train_completed_run_baselines(run_dir=run_dir)

    assert "SCORE_ONLY_FEATURES" in report["honesty_flags"]


def test_feature_audit_emits_safe_features_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        {"row_id": str(index), "label": [0, 1, -1, 0][index], "interval_begin": f"t{index}"}
        for index in range(4)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)

    assert "SAFE_FEATURES_MISSING" in report["honesty_flags"]


def test_chronological_split_uses_timestamp_ordering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(str(index), [-1, 0, 1, 1, 0, -1][index], minute=5 - index)
        for index in range(6)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)

    assert report["split"]["split_source"] == "chronological_interval_begin"
    assert report["split"]["train_row_count"] == 4
    assert report["split"]["test_row_count"] == 2


def test_fold_split_uses_fold_ordering_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(str(index), [-1, 0, 1, 1, 0, -1][index], fold_index=index // 3, minute=index)
        for index in range(6)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)

    assert report["split"]["split_source"] == "fold_index"
    assert report["split"]["test_row_count"] == 3


def test_majority_and_random_baselines_are_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(str(index), [0, 0, 1, -1, 0, 1, -1, 0][index], minute=index)
        for index in range(8)
    ]
    _write_run(run_dir, labels)

    first = train_completed_run_baselines(run_dir=run_dir)
    second = train_completed_run_baselines(run_dir=run_dir)

    first_csv = Path(first["output_files"]["baseline_metrics_csv"]).read_text(encoding="utf-8")
    second_csv = Path(second["output_files"]["baseline_metrics_csv"]).read_text(encoding="utf-8")
    assert first_csv == second_csv
    assert "majority_class" in first_csv
    assert "stratified_random_seed_1729" in first_csv


def test_logistic_regression_baseline_runs_when_sklearn_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(
            str(index),
            1 if index % 3 == 0 else (-1 if index % 3 == 1 else 0),
            minute=index,
            volatility=0.001 + index / 1000.0,
        )
        for index in range(30)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)
    baseline_names = {row["baseline_name"] for row in report["baselines"]}

    assert "logistic_regression_tiny" in baseline_names


def test_outputs_confusion_slices_and_recommendation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        _label_row(
            str(index),
            [0, 0, 1, -1, 0, 1, -1, 0][index],
            symbol="ETH/USD" if index % 2 else "BTC/USD",
            minute=index,
        )
        for index in range(8)
    ]
    _write_run(run_dir, labels)

    report = train_completed_run_baselines(run_dir=run_dir)
    baseline_dir = Path(report["baseline_dir"])

    assert (baseline_dir / "baseline_manifest.json").exists()
    assert (baseline_dir / "feature_audit.json").exists()
    assert (baseline_dir / "baseline_confusion_matrix.csv").exists()
    assert (baseline_dir / "baseline_by_slice.csv").exists()
    assert (baseline_dir / "baseline_report.md").exists()
    assert report["recommendation"]
    assert "BASELINE_FEASIBILITY_ONLY" in report["honesty_flags"]
