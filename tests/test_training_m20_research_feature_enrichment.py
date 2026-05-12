"""Tests for M20 research feature enrichment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_research_feature_enrichment import (
    build_m20_research_feature_enrichment,
)

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows(*, future_spike: bool = False) -> list[dict[str, object]]:
    output = []
    for index in range(32):
        output.append(
            {
                "symbol": "BTC/USD",
                "interval_begin": f"2026-05-01T{index // 12:02d}:{(index % 12) * 5:02d}:00Z",
                "fold_index": 4,
                "row_id": f"BTC/USD|{index}",
                "open_price": 100 + index,
                "high_price": 101 + index + (1000 if future_spike and index == 31 else 0),
                "low_price": 99 + index,
                "close_price": 100 + index,
                "realized_vol_12": 0.01 if index < 20 else 0.10,
                "momentum_3": 0.03 if index % 2 == 0 else -0.03,
                "macd_line_12_26": 1.0 if index % 2 == 0 else -1.0,
            }
        )
    return output


def _run_dir(base: Path, *, rows: list[dict[str, object]] | None = None) -> Path:
    run_dir = base / "run"
    source_rows = rows or _rows()
    _write_csv(run_dir / "training_frame" / "m20_training_frame_features.csv", source_rows)
    _write_json(
        run_dir / "training_frame" / "m20_training_frame_feature_columns.json",
        {
            "feature_columns": [
                column
                for column in source_rows[0]
                if column not in {"symbol", "interval_begin", "fold_index", "row_id"}
            ]
        },
    )
    (run_dir / "research_labels" / "vol_scaled" / "safe_feature_availability").mkdir(
        parents=True,
        exist_ok=True,
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
                    "fitted_row_count": 100,
                    "high_vol_threshold": 0.05,
                    "trend_abs_threshold": 0.02,
                }
            },
        },
    )
    return path


def test_builds_research_features_with_regime_and_adx(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    result = build_m20_research_feature_enrichment(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )
    output_rows = _read_csv(Path(result["output_files"]["research_features_csv"]))

    assert result["rows_written"] == 32
    assert result["added_features"] == ["regime_label", "adx_14"]
    assert output_rows[0]["regime_label"] == "TREND_UP"
    assert output_rows[20]["regime_label"] == "HIGH_VOL"
    assert output_rows[0]["adx_14"] == ""
    assert output_rows[-1]["adx_14"] != ""


def test_adx_is_causal_and_future_row_does_not_change_prior_values(tmp_path: Path) -> None:
    first_dir = _run_dir(tmp_path / "first", rows=_rows())
    second_dir = _run_dir(tmp_path / "second", rows=_rows(future_spike=True))
    thresholds = _thresholds(tmp_path / "thresholds.json")

    first = build_m20_research_feature_enrichment(
        source_run_dir=first_dir,
        regime_thresholds_path=thresholds,
    )
    second = build_m20_research_feature_enrichment(
        source_run_dir=second_dir,
        regime_thresholds_path=thresholds,
    )

    first_rows = _read_csv(Path(first["output_files"]["research_features_csv"]))
    second_rows = _read_csv(Path(second["output_files"]["research_features_csv"]))
    assert first_rows[30]["adx_14"] == second_rows[30]["adx_14"]


def test_bad_thresholds_block_regime_but_keep_adx(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    result = build_m20_research_feature_enrichment(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(
            tmp_path / "thresholds.json",
            schema_version="m8_thresholds_v0",
        ),
    )
    output_rows = _read_csv(Path(result["output_files"]["research_features_csv"]))

    assert result["added_features"] == ["adx_14"]
    assert result["blocked_features"][0]["feature_name"] == "regime_label"
    assert "regime_label" not in output_rows[0]
    assert "adx_14" in output_rows[0]


def test_leakage_audit_preserves_research_only_status(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    result = build_m20_research_feature_enrichment(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    for row in result["leakage_audit"]:
        assert row["uses_future_data"] == "False"
        assert row["uses_labels"] == "False"
        assert row["uses_economic_outcomes"] == "False"
        assert row["runtime_effect"] == "False"
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_original_training_frame_is_not_mutated(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    source_path = run_dir / "training_frame" / "m20_training_frame_features.csv"
    before = source_path.read_text(encoding="utf-8")

    build_m20_research_feature_enrichment(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )

    assert source_path.read_text(encoding="utf-8") == before


def test_outputs_required_artifacts_and_recommendation(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    result = build_m20_research_feature_enrichment(
        source_run_dir=run_dir,
        regime_thresholds_path=_thresholds(tmp_path / "thresholds.json"),
    )
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "research_feature_columns.json").exists()
    assert (output_dir / "feature_lineage.csv").exists()
    assert (output_dir / "leakage_audit.csv").exists()
    assert result["recommendation"] == "RE_RUN_V2_STRATEGY_CANDIDATE_FACTORY_WITH_RESEARCH_FEATURES"
