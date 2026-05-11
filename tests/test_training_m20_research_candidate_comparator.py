"""Focused tests for M20 generic research candidate comparator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_research_candidate_comparator import (
    compare_m20_research_candidates,
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


def _write_sources(base: Path, prediction: Path) -> None:
    vol = base / "research_labels" / "vol_scaled"
    _write_csv(
        vol / "strategy_candidate_factory" / "candidate_decisions.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "candidate_decision": "STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE",
                "selected_rows": 20,
                "coverage": 0.5,
                "mean_net_proxy": -0.002,
                "cumulative_net_proxy": -0.04,
            }
        ],
    )
    _write_csv(
        vol / "strategy_slice_policy_evaluator" / "candidate_decisions.csv",
        [
            {
                "policy_name": "return_reversal:return_positive:symbol=BTC/USD",
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "candidate_decision": "SLICE_POLICY_ECONOMICS_NEGATIVE",
                "selected_rows": 10,
                "mean_net_proxy": -0.001,
                "tail_loss_rate": 0.7,
            }
        ],
    )
    _write_csv(
        vol / "strategy_model_factory_plan" / "candidate_inputs.csv",
        [
            {
                "strategy_family": "return_reversal",
                "candidate_name": "return_positive",
                "best_slice": "symbol=BTC/USD",
                "best_slice_mean_net_proxy": -0.001,
                "slice_policy_decision": "SLICE_POLICY_ECONOMICS_NEGATIVE",
                "model_factory_status": "BLOCKED_PENDING_STRATEGY_REFINEMENT",
            }
        ],
    )
    _write_csv(
        prediction
        / "research_labels"
        / "vol_scaled"
        / "cost_aware_policy_adjudication"
        / "model_decisions.csv",
        [
            {
                "model_name": "neuralforecast_patchtst",
                "best_policy": "EDGE_SLICE_MONTH_2024-11",
                "final_decision": "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE",
                "selected_rows": 12,
                "lift_vs_base": 2.4,
                "mean_net_proxy": -0.003,
                "cumulative_net_proxy": -0.036,
            }
        ],
    )


def test_comparator_reads_strategy_policy_and_specialist_artifacts(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = compare_m20_research_candidates(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )
    scorecard = _read_csv(Path(result["output_files"]["candidate_scorecard_csv"]))
    types = {row["candidate_type"] for row in scorecard}

    assert "strategy_candidate" in types
    assert "strategy_slice_policy" in types
    assert "strategy_model_factory_input" in types
    assert "specialist_policy" in types


def test_negative_economics_remain_research_only(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = compare_m20_research_candidates(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))

    assert any(
        row["candidate_decision"] == "ECONOMICS_NEGATIVE_RESEARCH_ONLY"
        for row in decisions
    )
    assert all(row["promotion_status"] == "NOT_PROMOTABLE" for row in decisions)
    assert all(row["profitability_status"] == "NO_PROFIT_CLAIM" for row in decisions)
    assert result["recommendation"] == "PAUSE_CURRENT_M20_CANDIDATE_PATHS"


def test_missing_prediction_run_is_reported_as_blocker(tmp_path: Path) -> None:
    _write_sources(tmp_path, tmp_path / "prediction")

    result = compare_m20_research_candidates(source_run_dir=tmp_path)
    blockers = {row["blocker"] for row in result["blockers"]}

    assert "SPECIALIST_POLICY_ADJUDICATION_NOT_LINKED" in blockers


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = compare_m20_research_candidates(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )
    report = json.loads(
        Path(result["output_files"]["research_candidate_comparison_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_research_candidate_comparator.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
