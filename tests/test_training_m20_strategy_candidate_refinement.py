"""Focused tests for M20 strategy candidate refinement analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_strategy_candidate_refinement import (
    analyze_m20_strategy_candidate_refinement,
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


def _write_sources(base: Path) -> None:
    factory = base / "research_labels" / "vol_scaled" / "strategy_candidate_factory"
    _write_csv(
        factory / "candidate_metrics.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "selected_rows": 4,
                "mean_net_proxy": -0.002,
                "classification": "STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE",
            },
            {
                "strategy_family": "volume_context",
                "candidate_name": "volume_high",
                "selected_rows": 2,
                "mean_net_proxy": -0.01,
                "classification": "STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE",
            },
        ],
    )
    _write_csv(
        factory / "strategy_candidates.csv",
        [
            _candidate_row("return_reversal", "return_positive", 0, 0.01),
            _candidate_row("return_reversal", "return_positive", 1, -0.02),
            _candidate_row("return_reversal", "return_positive", 2, -0.01),
            _candidate_row("return_reversal", "return_positive", 3, 0.03),
            _candidate_row("volume_context", "volume_high", 2, -0.02),
            _candidate_row("volume_context", "volume_high", 3, -0.01),
        ],
    )
    (factory / "manifest.json").write_text("{}", encoding="utf-8")
    _write_csv(
        base / "training_frame" / "m20_training_frame_features.csv",
        [
            {
                "symbol": "BTC/USD" if index < 2 else "ETH/USD",
                "interval_begin": f"2025-0{index + 1}-01T00:00:00Z",
                "fold_index": "4",
                "high_price": 102 + index,
                "low_price": 99,
                "close_price": 100 + index,
                "volume": 10 + index * 10,
                "realized_vol_12": 0.01 + index * 0.02,
            }
            for index in range(4)
        ],
    )


def _candidate_row(
    strategy_family: str,
    candidate_name: str,
    index: int,
    net_value: float,
) -> dict[str, object]:
    return {
        "strategy_family": strategy_family,
        "candidate_name": candidate_name,
        "symbol": "BTC/USD" if index < 2 else "ETH/USD",
        "interval_begin": f"2025-0{index + 1}-01T00:00:00Z",
        "fold_index": "4",
        "setup_passed": True,
        "fee_exceedance_label": "1" if net_value > 0 else "0",
        "gross_value_proxy": net_value + 0.002,
        "net_value_proxy": net_value,
    }


def test_refinement_analyzes_symbol_time_and_regime_slices(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_candidate_refinement(
        source_run_dir=tmp_path,
        min_slice_rows=1,
    )
    slices = _read_csv(Path(result["output_files"]["slice_diagnostics_csv"]))
    slice_families = {row["slice_family"] for row in slices}

    assert {"symbol", "month", "quarter", "volatility_bucket"} <= slice_families
    assert result["candidate_count"] == 2


def test_positive_slice_is_watchlist_not_promotable(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_candidate_refinement(
        source_run_dir=tmp_path,
        min_slice_rows=1,
    )
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))
    return_positive = next(row for row in decisions if row["candidate_name"] == "return_positive")

    assert return_positive["candidate_decision"] == "REFINED_SLICE_POLICY_WATCHLIST"
    assert return_positive["promotion_status"] == "NOT_PROMOTABLE"
    assert return_positive["profitability_status"] == "NO_PROFIT_CLAIM"
    assert result["next_required_action"] == "RUN_GENERIC_STRATEGY_SLICE_POLICY_EVALUATOR"


def test_tail_loss_and_sample_size_diagnostics_are_written(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_candidate_refinement(
        source_run_dir=tmp_path,
        min_slice_rows=3,
    )
    tail = _read_csv(Path(result["output_files"]["tail_loss_diagnostics_csv"]))
    samples = _read_csv(Path(result["output_files"]["sample_size_diagnostics_csv"]))

    assert any(float(row["tail_loss_rate"]) > 0.0 for row in tail)
    assert any(row["sample_size_status"] == "LOW_SAMPLE" for row in samples)


def test_missing_candidate_factory_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing strategy candidate factory artifacts"):
        analyze_m20_strategy_candidate_refinement(source_run_dir=tmp_path)


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_candidate_refinement(source_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["strategy_candidate_refinement_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_candidate_refinement.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
