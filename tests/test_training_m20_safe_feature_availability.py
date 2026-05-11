"""Tests for the M20 safe feature availability audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_safe_feature_availability import audit_m20_safe_feature_availability

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_run(base: Path, *, include_ohlc: bool = True) -> Path:
    run_dir = base / "run"
    row = {
        "symbol": "BTC/USD",
        "interval_begin": "2026-05-01T00:00:00Z",
        "fold_index": 4,
        "row_id": "BTC/USD|0",
        "realized_vol_12": 0.01,
        "momentum_3": 0.003,
        "macd_line_12_26": 0.2,
    }
    if include_ohlc:
        row.update({"high_price": 102.0, "low_price": 99.0, "close_price": 101.0})
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", [row])
    _write_json(
        run_dir / "training_frame" / "m20_training_frame_feature_columns.json",
        {"feature_columns": [key for key in row if key not in {"symbol", "interval_begin"}]},
    )
    return run_dir


def _thresholds(path: Path, *, schema_version: str = "m8_thresholds_v1") -> Path:
    _write_json(
        path,
        {
            "schema_version": schema_version,
            "run_id": "20260320T165813Z",
            "required_inputs": ["realized_vol_12", "momentum_3", "macd_line_12_26"],
            "thresholds_by_symbol": {
                "BTC/USD": {
                    "symbol": "BTC/USD",
                    "high_vol_threshold": 0.05,
                    "trend_abs_threshold": 0.02,
                    "fitted_row_count": 100,
                }
            },
        },
    )
    return path


def test_audit_marks_regime_and_adx_safe_when_inputs_exist(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    by_feature = {row["feature_name"]: row for row in result["feature_sources"]}
    assert by_feature["regime_label"]["availability_status"] == "SAFE_COMPUTABLE"
    assert by_feature["adx_14"]["availability_status"] == "SAFE_COMPUTABLE"
    assert result["recommendation"] == "BUILD_M20_RESEARCH_FEATURE_ENRICHMENT_ARTIFACT"


def test_missing_ohlc_blocks_adx_but_not_regime(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path, include_ohlc=False)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    by_feature = {row["feature_name"]: row for row in result["feature_sources"]}
    assert by_feature["regime_label"]["availability_status"] == "SAFE_COMPUTABLE"
    assert by_feature["adx_14"]["availability_status"] == "BLOCKED"
    assert "high_price" in by_feature["adx_14"]["blockers"]


def test_bad_threshold_schema_blocks_regime(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(
            tmp_path / "thresholds.json",
            schema_version="m8_thresholds_v0",
        ),
    )

    by_feature = {row["feature_name"]: row for row in result["feature_sources"]}
    assert by_feature["regime_label"]["availability_status"] == "BLOCKED"
    assert "UNSUPPORTED_M8_THRESHOLD_SCHEMA" in by_feature["regime_label"]["blockers"]


def test_outputs_preserve_research_honesty_flags(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"


def test_leakage_audit_records_no_future_or_outcome_inputs(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    for row in result["leakage_risk_audit"]:
        assert row["uses_future_data"] == "False"
        assert row["uses_labels"] == "False"
        assert row["uses_economic_outcomes"] == "False"


def test_audit_writes_required_artifacts(tmp_path: Path) -> None:
    run_dir = _source_run(tmp_path)
    result = audit_m20_safe_feature_availability(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    output_dir = Path(result["output_files"]["manifest_json"]).parent
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "feature_sources.csv").exists()
    assert (output_dir / "blocked_features.csv").exists()
    assert (output_dir / "leakage_risk_audit.csv").exists()
    assert (output_dir / "recommendation.json").exists()
