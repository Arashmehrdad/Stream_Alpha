"""Focused tests for M20 research feature-matrix alignment."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_feature_matrix import build_m20_research_feature_matrix

# pylint: disable=missing-function-docstring


def _write_run(
    run_dir: Path,
    label_rows: list[dict[str, object]],
    source_name: str | None = "features.csv",
    source_rows: list[dict[str, object]] | None = None,
) -> None:
    label_dir = run_dir / "research_labels" / "vol_scaled"
    label_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(label_dir / "triple_barrier_labels_vol_scaled.csv", label_rows)
    if source_name and source_rows is not None:
        _write_csv(run_dir / source_name, source_rows)


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


def _label(index: int, symbol: str = "BTC/USD") -> dict[str, object]:
    return {
        "row_id": f"{symbol}|2026-04-01T00:{index:02d}:00Z",
        "symbol": symbol,
        "interval_begin": f"2026-04-01T00:{index:02d}:00Z",
        "fold_index": index // 3,
        "regime_label": "RANGE",
        "label": [-1, 0, 1][index % 3],
    }


def _feature(index: int, symbol: str = "BTC/USD") -> dict[str, object]:
    return {
        "symbol": symbol,
        "interval_begin": f"2026-04-01T00:{index:02d}:00Z",
        "feature_a": 0.1 + index,
        "prob_up": 0.4 + index / 100.0,
        "future_return_3": 0.9,
        "barrier_hit": 1,
        "label": 1,
        "y_true": 0,
    }


def test_discovers_and_selects_preferred_feature_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(4)]
    _write_run(run_dir, labels, "oof_predictions.csv", [_feature(index) for index in range(4)])
    _write_csv(run_dir / "features.csv", [_feature(index) for index in range(4)])

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    selected = report["source_audit"]["selected_source_file"]["relative_path"]
    assert selected == "features.csv"
    assert "FEATURE_SOURCE_FOUND" in report["honesty_flags"]


def test_joins_labels_and_features_by_symbol_interval_begin(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(6)]
    _write_run(run_dir, labels, "features.csv", [_feature(index) for index in range(6)])

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    alignment = report["alignment_report"]
    assert alignment["join_keys_used"] == ["symbol", "interval_begin"]
    assert alignment["matched_rows"] == 6
    assert "TIMESTAMP_ALIGNMENT_CONFIRMED" in report["honesty_flags"]
    assert (Path(report["feature_matrix_dir"]) / "research_feature_matrix.csv").exists()


def test_rejects_leakage_label_outcome_and_barrier_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(4)]
    _write_run(run_dir, labels, "features.csv", [_feature(index) for index in range(4)])

    report = build_m20_research_feature_matrix(run_dir=run_dir)
    excluded = {
        row["column"]: row["reason"]
        for row in report["alignment_report"]["leakage_screen_summary"].get("excluded", [])
    }
    exclusion_file = Path(
        report["output_files"]["feature_exclusion_reasons_csv"]
    ).read_text(encoding="utf-8")

    assert "future_return_3" in exclusion_file
    assert "barrier_hit" in exclusion_file
    assert "label" in exclusion_file
    assert "y_true" in exclusion_file
    assert excluded == {}


def test_detects_duplicate_join_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(3)]
    features = [_feature(index) for index in range(3)] + [_feature(0)]
    _write_run(run_dir, labels, "features.csv", features)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert "DUPLICATE_JOIN_KEYS" in report["honesty_flags"]
    assert report["alignment_report"]["duplicate_key_count"] == 1


def test_filters_oof_rows_to_label_model_before_alignment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        {**_label(index), "model_name": "winner"}
        for index in range(3)
    ]
    features = []
    for index in range(3):
        features.append({**_feature(index), "model_name": "winner"})
        features.append({**_feature(index), "model_name": "loser"})
    _write_run(run_dir, labels, "oof_predictions.csv", features)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert report["alignment_report"]["matched_rows"] == 3
    assert "DUPLICATE_JOIN_KEYS" not in report["honesty_flags"]


def test_detects_high_unmatched_label_rate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(10)]
    features = [_feature(index) for index in range(2)]
    _write_run(run_dir, labels, "features.csv", features)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert "HIGH_UNMATCHED_LABEL_RATE" in report["honesty_flags"]
    assert report["alignment_report"]["unmatched_label_rows"] == 8


def test_handles_missing_feature_source_with_blocker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, [_label(index) for index in range(3)], source_name=None)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert "FEATURE_SOURCE_MISSING" in report["honesty_flags"]
    assert "FEATURE_MATRIX_NOT_TRAINING_READY" in report["honesty_flags"]


def test_handles_no_safe_numeric_features_with_blocker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(3)]
    features = [
        {
            "symbol": row["symbol"],
            "interval_begin": row["interval_begin"],
            "future_return_3": 0.1,
            "label": row["label"],
        }
        for row in labels
    ]
    _write_run(run_dir, labels, "features.csv", features)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert "SAFE_FEATURES_MISSING" in report["honesty_flags"]


def test_writes_deterministic_audit_report_and_manifest_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [_label(index) for index in range(4)]
    _write_run(run_dir, labels, "features.csv", [_feature(index) for index in range(4)])

    first = build_m20_research_feature_matrix(run_dir=run_dir)
    second = build_m20_research_feature_matrix(run_dir=run_dir)

    manifest_path = Path(first["output_files"]["research_feature_matrix_manifest_json"])
    assert manifest_path.exists()
    assert Path(first["output_files"]["feature_source_audit_md"]).exists()
    assert Path(first["output_files"]["feature_alignment_report_md"]).exists()
    assert manifest_path.read_text(encoding="utf-8") == Path(
        second["output_files"]["research_feature_matrix_manifest_json"]
    ).read_text(encoding="utf-8")


def test_marks_weak_row_order_alignment_when_no_proper_keys_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels = [
        {"row_id": str(index), "label": [-1, 0, 1][index % 3]}
        for index in range(3)
    ]
    features = [{"feature_a": index + 0.1} for index in range(3)]
    _write_run(run_dir, labels, "features.csv", features)

    report = build_m20_research_feature_matrix(run_dir=run_dir)

    assert report["alignment_report"]["join_keys_used"] == ["row_order"]
    assert "WEAK_ROW_ORDER_ALIGNMENT" in report["honesty_flags"]
