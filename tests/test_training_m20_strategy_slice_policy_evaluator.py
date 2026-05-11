"""Focused tests for M20 strategy slice policy evaluator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_strategy_slice_policy_evaluator import (
    analyze_m20_strategy_slice_policy,
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
    refinement = base / "research_labels" / "vol_scaled" / "strategy_candidate_refinement"
    rows = [
        _slice_row("symbol", "BTC/USD", 200, 0.002, 0.40),
        _slice_row("month", "2025-01", 150, -0.003, 0.80),
        _slice_row("volatility_bucket", "HIGH", 80, 0.004, 0.30),
        _slice_row("range_bucket", "LOW", 120, 0.001, 0.90),
    ]
    _write_csv(refinement / "slice_diagnostics.csv", rows)
    _write_csv(
        refinement / "refined_candidate_metrics.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "refinement_decision": "REFINED_SLICE_POLICY_WATCHLIST",
            }
        ],
    )
    (refinement / "manifest.json").write_text("{}", encoding="utf-8")


def _slice_row(
    slice_family: str,
    slice_value: str,
    selected_rows: int,
    mean_net: float,
    tail_loss_rate: float,
) -> dict[str, object]:
    return {
        "strategy_family": "return_reversal",
        "candidate_name": "return_positive",
        "slice_family": slice_family,
        "slice_value": slice_value,
        "selected_rows": selected_rows,
        "selected_positive_rate": 0.5,
        "mean_net_proxy": mean_net,
        "cumulative_net_proxy": mean_net * selected_rows,
        "max_drawdown_proxy": -0.02,
        "win_rate_proxy": 0.5,
        "worst_5_net_proxy": -0.05,
        "tail_loss_rate": tail_loss_rate,
        "sample_size_status": "ADEQUATE_SAMPLE",
        "slice_decision": "REFINED_SLICE_WATCHLIST_POSITIVE_NET_PROXY",
    }


def test_evaluator_builds_generic_policy_outputs(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_slice_policy(source_run_dir=tmp_path)

    assert result["policy_count"] == 4
    assert Path(result["output_files"]["policy_metrics_csv"]).exists()
    assert Path(result["output_files"]["by_symbol_csv"]).exists()
    assert Path(result["output_files"]["by_time_csv"]).exists()
    assert Path(result["output_files"]["tail_risk_csv"]).exists()


def test_positive_slice_is_research_watchlist_not_promotable(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_slice_policy(source_run_dir=tmp_path)
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))
    btc_policy = next(row for row in decisions if row["slice_value"] == "BTC/USD")

    assert (
        btc_policy["candidate_decision"]
        == "SLICE_POLICY_RESEARCH_WATCHLIST_POSITIVE_NET_PROXY"
    )
    assert btc_policy["promotion_status"] == "NOT_PROMOTABLE"
    assert btc_policy["profitability_status"] == "NO_PROFIT_CLAIM"
    assert result["next_required_action"] == "PLAN_GENERIC_STRATEGY_CONDITIONED_MODEL_FACTORY"


def test_low_sample_and_tail_risk_are_classified(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_slice_policy(source_run_dir=tmp_path)
    metrics = _read_csv(Path(result["output_files"]["policy_metrics_csv"]))
    low_sample = next(row for row in metrics if row["slice_value"] == "HIGH")
    tail_risk = next(row for row in metrics if row["slice_value"] == "LOW")

    assert low_sample["policy_classification"] == "SLICE_POLICY_LOW_SAMPLE"
    assert tail_risk["policy_classification"] == "SLICE_POLICY_POSITIVE_NET_WITH_TAIL_RISK"


def test_missing_refinement_artifacts_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing strategy candidate refinement artifacts"):
        analyze_m20_strategy_slice_policy(source_run_dir=tmp_path)


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = analyze_m20_strategy_slice_policy(source_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["strategy_slice_policy_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_slice_policy_evaluator.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
