"""Focused tests for M20 economic outcome artifact generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_economic_outcome_artifacts import (
    build_m20_economic_outcome_artifacts,
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


def _label_path(run_dir: Path) -> Path:
    return run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv"


def test_outcomes_from_future_return_labels(tmp_path: Path) -> None:
    _write_csv(
        _label_path(tmp_path),
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "fold_index": 4,
                "label": 1,
                "future_return": 0.03,
                "horizon": 3,
            }
        ],
    )

    result = build_m20_economic_outcome_artifacts(source_run_dir=tmp_path)
    rows = _read_csv(Path(result["output_files"]["economic_outcomes_csv"]))

    assert result["economics_computable"] is True
    assert float(rows[0]["gross_value_proxy"]) == 0.03
    assert float(rows[0]["net_value_proxy"]) == pytest.approx(0.028)


def test_fee_and_slippage_are_subtracted(tmp_path: Path) -> None:
    _write_csv(
        _label_path(tmp_path),
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "label": 1,
                "future_return": 0.05,
            }
        ],
    )

    result = build_m20_economic_outcome_artifacts(
        source_run_dir=tmp_path,
        fee_bps=25,
        slippage_bps=5,
    )
    rows = _read_csv(Path(result["output_files"]["economic_outcomes_csv"]))

    assert float(rows[0]["net_value_proxy"]) == pytest.approx(0.047)


def test_outcomes_from_training_frame_prices(tmp_path: Path) -> None:
    feature_path = tmp_path / "training_frame" / "m20_training_frame_features.csv"
    _write_csv(
        feature_path,
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "fold_index": 4,
                "close_price": 100,
            },
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:05:00Z",
                "fold_index": 4,
                "close_price": 102,
            },
        ],
    )

    result = build_m20_economic_outcome_artifacts(
        source_run_dir=tmp_path,
        fee_bps=10,
        horizon_candles=1,
    )
    rows = _read_csv(Path(result["output_files"]["economic_outcomes_csv"]))

    assert result["rows_written"] == 1
    assert float(rows[0]["future_return"]) == pytest.approx(0.02)
    assert float(rows[0]["net_value_proxy"]) == pytest.approx(0.019)


def test_missing_magnitude_source_emits_blocker(tmp_path: Path) -> None:
    _write_csv(
        _label_path(tmp_path),
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "label": 1,
            }
        ],
    )

    result = build_m20_economic_outcome_artifacts(source_run_dir=tmp_path)

    assert result["economics_computable"] is False
    assert "ECONOMIC_MAGNITUDE_NOT_AVAILABLE" in result["blockers"]


def test_binary_labels_do_not_create_fake_net_proxy(tmp_path: Path) -> None:
    _write_csv(
        _label_path(tmp_path),
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "label": 1,
            }
        ],
    )

    result = build_m20_economic_outcome_artifacts(source_run_dir=tmp_path)

    assert result["rows_written"] == 0
    assert not Path(result["output_files"]["economic_outcomes_csv"]).exists()
    assert result["recommendation"] == "ADD_RETURN_MAGNITUDE_SOURCE_FOR_ECONOMIC_OUTCOMES"


def test_missing_source_files_emit_operator_blocker(tmp_path: Path) -> None:
    result = build_m20_economic_outcome_artifacts(source_run_dir=tmp_path)

    assert result["economics_computable"] is False
    assert "MISSING_SAFE_ECONOMIC_SOURCE" in result["blockers"]
    assert result["recommendation"] == "BLOCKED_MISSING_SAFE_ECONOMIC_SOURCE"


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_csv(
        _label_path(tmp_path),
        [
            {
                "symbol": "BTC/USD",
                "interval_begin": "2025-01-01T00:00:00Z",
                "label": 1,
                "future_return": 0.01,
            }
        ],
    )

    result = build_m20_economic_outcome_artifacts(source_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["economic_outcome_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["honesty_flags"]
    assert "NOT_PROMOTABLE" in report["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in report["honesty_flags"]


def test_no_runtime_imports_or_registry_writes() -> None:
    module_path = Path("app/training/m20_economic_outcome_artifacts.py")
    source = module_path.read_text(encoding="utf-8")

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
