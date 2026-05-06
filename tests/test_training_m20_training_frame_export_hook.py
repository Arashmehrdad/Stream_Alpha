"""Focused tests for the M20 training-frame export hook."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.m20_training_frame_export_hook import (
    export_m20_training_frame_fold,
    finalize_m20_training_frame_export,
    maybe_export_m20_training_frame,
    read_csv_rows,
)

# pylint: disable=missing-function-docstring


def _rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BTC/USD",
            "interval_begin": "2026-04-01T00:00:00Z",
            "fold_index": 4,
            "row_id": "BTC/USD|0",
            "open_price": 100.0,
            "close_price": 101.0,
            "volume": 10.0,
            "prob_up": 0.7,
            "future_return_3": 0.1,
            "label": 1,
        },
        {
            "symbol": "ETH/USD",
            "interval_begin": "2026-04-01T00:05:00Z",
            "fold_index": 4,
            "row_id": "ETH/USD|1",
            "open_price": 200.0,
            "close_price": 201.0,
            "volume": 20.0,
            "prob_up": 0.6,
            "future_return_3": -0.1,
            "label": 0,
        },
    ]


def test_export_hook_writes_deterministic_manifest_report_and_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["open_price", "close_price", "volume"],
        config_path=tmp_path / "config.json",
    )
    second = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["open_price", "close_price", "volume"],
        config_path=tmp_path / "config.json",
    )

    assert first is not None
    export_dir = run_dir / "training_frame"
    assert (export_dir / "m20_training_frame_features.csv").exists()
    assert (export_dir / "m20_training_frame_keys.csv").exists()
    assert (export_dir / "m20_training_frame_export_report.md").exists()
    manifest = export_dir / "m20_training_frame_export_manifest.json"
    assert manifest.read_text(encoding="utf-8") == (
        export_dir / "m20_training_frame_export_manifest.json"
    ).read_text(encoding="utf-8")
    assert first["manifest"] == second["manifest"]


def test_excludes_prediction_output_and_leakage_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=[
            "open_price",
            "prob_up",
            "future_return_3",
            "label",
            "close_price",
        ],
    )

    exported = read_csv_rows(run_dir / "training_frame" / "m20_training_frame_features.csv")
    assert report is not None
    assert list(exported[0]) == [
        "symbol",
        "interval_begin",
        "fold_index",
        "row_id",
        "open_price",
        "close_price",
    ]
    assert "PREDICTION_OUTPUT_COLUMNS_EXCLUDED" in report["honesty_flags"]
    assert "POSSIBLE_LEAKAGE_COLUMNS_EXCLUDED" in report["honesty_flags"]


def test_preserves_symbol_timestamp_and_fold_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["open_price"],
    )

    keys = read_csv_rows(run_dir / "training_frame" / "m20_training_frame_keys.csv")
    assert list(keys[0]) == ["symbol", "interval_begin", "fold_index", "row_id"]
    assert keys[0]["fold_index"] == "4"


def test_records_feature_column_order_and_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["volume", "open_price", "close_price"],
    )

    payload = json.loads(
        (run_dir / "training_frame" / "m20_training_frame_feature_columns.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["feature_columns"] == ["volume", "open_price", "close_price"]
    assert len(payload["feature_column_hash"]) == 64


def test_handles_duplicate_keys_with_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    duplicate_rows = [_rows()[0], _rows()[0]]
    report = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=duplicate_rows,
        feature_columns=["open_price"],
    )

    assert report is not None
    assert report["report"]["duplicate_key_count"] == 1


def test_handles_missing_timestamp_key_with_clear_blocker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [{"symbol": "BTC/USD", "fold_index": 1, "row_id": "x", "open_price": 1.0}]
    report = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=rows,
        feature_columns=["open_price"],
    )

    assert report is not None
    assert "TIMESTAMP_KEYS_PRESENT" not in report["honesty_flags"]


def test_does_nothing_when_export_flag_is_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = maybe_export_m20_training_frame(
        enabled=False,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["open_price"],
    )

    assert report is None
    assert not (run_dir / "training_frame").exists()


def test_manifest_records_no_registry_write(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = maybe_export_m20_training_frame(
        enabled=True,
        run_dir=run_dir,
        rows=_rows(),
        feature_columns=["open_price"],
    )

    assert report is not None
    assert report["manifest"]["registry_write"] is False
    assert "NO_REGISTRY_WRITE" in report["honesty_flags"]


def test_per_fold_export_writes_fold_files_manifest_and_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    export_m20_training_frame_fold(
        run_dir=run_dir,
        fold_index=4,
        rows=_rows(),
        feature_columns=["open_price", "close_price"],
        export_mode="export_only",
    )

    fold_dir = run_dir / "training_frame" / "folds"
    assert (fold_dir / "fold_4_features.csv").exists()
    assert (fold_dir / "fold_4_keys.csv").exists()
    assert (fold_dir / "fold_4_manifest.json").exists()
    assert (run_dir / "training_frame" / "m20_training_frame_export_checkpoint.json").exists()


def test_finalize_export_only_combines_folds_and_marks_scoring_skipped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    export_m20_training_frame_fold(
        run_dir=run_dir,
        fold_index=4,
        rows=_rows(),
        feature_columns=["open_price", "close_price"],
        export_mode="export_only",
        skipped_folds=[{"fold_index": 0, "reason": "outside_recent_score_only_window"}],
    )

    report = finalize_m20_training_frame_export(
        run_dir=run_dir,
        feature_columns=["open_price", "close_price"],
        export_mode="export_only",
        skipped_folds=[{"fold_index": 0, "reason": "outside_recent_score_only_window"}],
        complete=True,
    )

    assert (run_dir / "training_frame" / "m20_training_frame_features.csv").exists()
    assert "EXPORT_ONLY_MODE" in report["honesty_flags"]
    assert "MODEL_SCORING_SKIPPED_EXPORT_ONLY" in report["honesty_flags"]
    assert "FOLDS_SKIPPED_BY_RECENT_WINDOW" in report["honesty_flags"]
    assert report["report"]["export_complete"] is True


def test_partial_export_checkpoint_marks_partial_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    export_m20_training_frame_fold(
        run_dir=run_dir,
        fold_index=4,
        rows=_rows(),
        feature_columns=["open_price"],
        export_mode="early_score_only",
    )

    checkpoint = json.loads(
        (run_dir / "training_frame" / "m20_training_frame_export_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["export_complete"] is False
    assert "MARKET_FEATURE_FRAME_EXPORT_PARTIAL" in checkpoint["honesty_flags"]
