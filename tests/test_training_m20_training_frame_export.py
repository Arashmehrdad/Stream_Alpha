"""Focused tests for M20 research training-frame feature export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_training_frame_export import export_m20_training_frame_features

# pylint: disable=missing-function-docstring


def _write_run(
    run_dir: Path,
    *,
    source_name: str | None,
    source_rows: list[dict[str, object]] | None,
    feature_columns: list[str] | None = None,
) -> None:
    if feature_columns is not None:
        (run_dir / "feature_columns.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "feature_columns.json").write_text(
            json.dumps({"numeric_feature_columns": feature_columns}, sort_keys=True),
            encoding="utf-8",
        )
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
        writer = csv.DictWriter(output, fieldnames=fieldnames or ["symbol"])
        writer.writeheader()
        writer.writerows(rows)


def _market_row(index: int) -> dict[str, object]:
    return {
        "symbol": "BTC/USD",
        "interval_begin": f"2026-04-01T00:{index:02d}:00Z",
        "fold_index": index // 2,
        "row_id": f"BTC/USD|{index}",
        "open_price": 100.0 + index,
        "close_price": 101.0 + index,
        "log_return_1": 0.01,
        "volume": 10.0 + index,
        "future_return_3": 0.5,
        "barrier_hit": 1,
        "label": 1,
        "prob_up": 0.7,
        "confidence": 0.8,
        "y_pred": 1,
    }


def test_discovers_and_selects_preferred_training_frame_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        source_name="oof_predictions.csv",
        source_rows=[{"symbol": "BTC/USD", "interval_begin": "t", "prob_up": 0.4}],
        feature_columns=["open_price", "close_price"],
    )
    _write_csv(run_dir / "training_frame_features.csv", [_market_row(index) for index in range(3)])

    report = export_m20_training_frame_features(run_dir=run_dir)

    selected = report["report"]["selected_source_file"]["relative_path"]
    assert selected == "training_frame_features.csv"
    assert "TRAINING_FRAME_EXPORT_READY" in report["honesty_flags"]


def test_excludes_prediction_future_label_and_barrier_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=[_market_row(index) for index in range(3)],
        feature_columns=["open_price", "close_price", "log_return_1", "volume"],
    )

    report = export_m20_training_frame_features(run_dir=run_dir)
    feature_columns_path = Path(report["output_files"]["m20_training_frame_feature_columns_json"])
    feature_payload = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    exported = Path(report["output_files"]["m20_training_frame_features_csv"]).read_text(
        encoding="utf-8"
    )

    assert feature_payload["feature_columns"] == [
        "open_price",
        "close_price",
        "log_return_1",
        "volume",
    ]
    assert "future_return_3" not in exported
    assert "barrier_hit" not in exported
    assert "prob_up" not in exported
    assert "label" not in exported.splitlines()[0]


def test_preserves_symbol_timestamp_and_fold_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=[_market_row(index) for index in range(2)],
        feature_columns=["open_price"],
    )

    report = export_m20_training_frame_features(run_dir=run_dir)
    keys_text = Path(report["output_files"]["m20_training_frame_keys_csv"]).read_text(
        encoding="utf-8"
    )

    assert "symbol" in keys_text.splitlines()[0]
    assert "interval_begin" in keys_text.splitlines()[0]
    assert "fold_index" in keys_text.splitlines()[0]
    assert "FOLD_KEYS_PRESENT" in report["honesty_flags"]


def test_emits_blocker_when_only_oof_score_columns_are_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        {
            "symbol": "BTC/USD",
            "interval_begin": f"2026-04-01T00:{index:02d}:00Z",
            "prob_up": 0.4,
            "confidence": 0.6,
            "y_pred": 0,
            "long_trade_taken": 0,
        }
        for index in range(3)
    ]
    _write_run(run_dir, source_name="oof_predictions.csv", source_rows=rows)

    report = export_m20_training_frame_features(run_dir=run_dir)

    assert "TRAINING_FRAME_EXPORT_BLOCKED" in report["honesty_flags"]
    assert "ONLY_OOF_SCORE_FEATURES_AVAILABLE" in report["honesty_flags"]
    assert Path(report["output_files"]["m20_required_future_export_schema_json"]).exists()


def test_writes_required_future_export_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, source_name=None, source_rows=None, feature_columns=["open_price"])

    report = export_m20_training_frame_features(run_dir=run_dir)
    schema_path = Path(report["output_files"]["m20_required_future_export_schema_json"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["required_keys"] == ["symbol", "interval_begin"]
    assert schema["feature_columns"] == ["open_price"]


def test_writes_deterministic_manifest_and_report_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=[_market_row(index) for index in range(3)],
        feature_columns=["open_price", "close_price"],
    )

    first = export_m20_training_frame_features(run_dir=run_dir)
    second = export_m20_training_frame_features(run_dir=run_dir)
    manifest_path = Path(first["output_files"]["m20_training_frame_export_manifest_json"])

    assert manifest_path.exists()
    assert Path(first["output_files"]["m20_training_frame_export_report_md"]).exists()
    assert manifest_path.read_text(encoding="utf-8") == Path(
        second["output_files"]["m20_training_frame_export_manifest_json"]
    ).read_text(encoding="utf-8")


def test_handles_duplicate_join_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [_market_row(0), _market_row(0), _market_row(1)]
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=rows,
        feature_columns=["open_price"],
    )

    report = export_m20_training_frame_features(run_dir=run_dir)

    assert report["report"]["duplicate_key_count"] == 1


def test_handles_missing_timestamp_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [{"symbol": "BTC/USD", "open_price": 100.0}]
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=rows,
        feature_columns=["open_price"],
    )

    report = export_m20_training_frame_features(run_dir=run_dir)

    assert "TIMESTAMP_KEYS_MISSING" in report["honesty_flags"]
    assert "TRAINING_FRAME_EXPORT_BLOCKED" in report["honesty_flags"]
    assert "ORIGINAL_TRAINING_FRAME_MISSING" in report["honesty_flags"]


def test_handles_missing_safe_market_features(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [{"symbol": "BTC/USD", "interval_begin": "2026-04-01T00:00:00Z", "label": 1}]
    _write_run(run_dir, source_name="training_frame_features.csv", source_rows=rows)

    report = export_m20_training_frame_features(run_dir=run_dir)

    assert "MARKET_FEATURES_MISSING" in report["honesty_flags"]


def test_validates_feature_column_ordering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        source_name="training_frame_features.csv",
        source_rows=[_market_row(index) for index in range(2)],
        feature_columns=["volume", "open_price", "close_price"],
    )

    report = export_m20_training_frame_features(run_dir=run_dir)
    payload = json.loads(
        Path(report["output_files"]["m20_training_frame_feature_columns_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert payload["feature_columns"] == ["volume", "open_price", "close_price"]
