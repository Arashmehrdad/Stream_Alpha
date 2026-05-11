"""Focused tests for M20 cost-aware policy adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_cost_aware_policy_adjudication import (
    write_m20_cost_aware_policy_adjudication,
)

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _policy_dir(run_dir: Path) -> Path:
    return (
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "cost_aware_specialist_policy_evaluator"
    )


def _write_policy_artifacts(run_dir: Path) -> None:
    policy_dir = _policy_dir(run_dir)
    _write_json(
        policy_dir / "cost_aware_policy_report.json",
        {
            "best_policy_candidate": "neuralforecast_patchtst:TOP_1_PERCENT",
            "economics_available": True,
            "overall_status": ["RESEARCH_ONLY", "NO_RUNTIME_EFFECT"],
        },
    )
    _write_json(
        policy_dir / "recommendation.json",
        {
            "best_policy_candidate": "neuralforecast_patchtst:TOP_1_PERCENT",
            "economics_available": True,
            "evidence_blockers": [
                "NOT_BACKTEST",
                "NOT_RUNTIME_READY",
                "NOT_PROMOTABLE",
                "NO_PROFIT_CLAIM",
            ],
            "recommendation": "REJECT_OR_WATCHLIST_SPECIALIST_POLICY",
        },
    )
    _write_json(
        policy_dir / "economics_availability.json",
        {"economics_available": True, "safe_source": "economic_outcome_artifacts"},
    )
    rows = [
        {
            "model_name": "neuralforecast_patchtst",
            "best_policy": "TOP_1_PERCENT",
            "candidate_decision": "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE",
            "selected_rows": 100,
            "coverage": 0.01,
            "precision": 0.35,
            "lift_vs_base": 2.4,
            "economics_status": "NET_PROXY_AVAILABLE",
            "mean_net_proxy": -0.001,
            "cumulative_net_proxy": -1.0,
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        },
        {
            "model_name": "neuralforecast_nhits",
            "best_policy": "TOP_1_PERCENT",
            "candidate_decision": "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE",
            "selected_rows": 100,
            "coverage": 0.01,
            "precision": 0.25,
            "lift_vs_base": 1.8,
            "economics_status": "NET_PROXY_AVAILABLE",
            "mean_net_proxy": -0.002,
            "cumulative_net_proxy": -2.0,
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        },
    ]
    _write_csv(policy_dir / "candidate_decisions.csv", rows)
    _write_csv(
        policy_dir / "model_policy_metrics.csv",
        [
            {
                "model_name": row["model_name"],
                "best_policy": row["best_policy"],
                "best_policy_classification": row["candidate_decision"],
            }
            for row in rows
        ],
    )
    _write_csv(
        policy_dir / "topk_policy_metrics.csv",
        [
            {
                "model_name": "neuralforecast_patchtst",
                "policy_name": "TOP_1_PERCENT",
                "lift_vs_base": 2.6,
            },
            {
                "model_name": "neuralforecast_nhits",
                "policy_name": "TOP_1_PERCENT",
                "lift_vs_base": 1.5,
            },
        ],
    )


def test_adjudication_reads_policy_artifacts(tmp_path: Path) -> None:
    _write_policy_artifacts(tmp_path)

    result = write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)

    assert Path(result["output_files"]["cost_aware_policy_adjudication_json"]).exists()
    assert result["overall_decision"] == "PAUSE_NEURALFORECAST_SPECIALIST_POLICY_PATH"


def test_negative_economics_produces_signal_confirmed_negative(tmp_path: Path) -> None:
    _write_policy_artifacts(tmp_path)

    result = write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)

    assert result["model_decisions"]["neuralforecast_patchtst"] == (
        "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE"
    )
    assert result["model_decisions"]["neuralforecast_nhits"] == (
        "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE"
    )


def test_statistical_signal_negative_net_is_not_promotable(tmp_path: Path) -> None:
    _write_policy_artifacts(tmp_path)

    result = write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)
    rows = _read_csv(Path(result["output_files"]["model_decisions_csv"]))
    patch = next(row for row in rows if row["model_name"] == "neuralforecast_patchtst")

    assert float(patch["lift_vs_base"]) > 1.0
    assert float(patch["mean_net_proxy"]) < 0.0
    assert patch["promotion_status"] == "NOT_PROMOTABLE"


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_policy_artifacts(tmp_path)

    result = write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["cost_aware_policy_adjudication_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert report["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert report["promotion_status"] == "NOT_PROMOTABLE"
    assert report["profitability_status"] == "NO_PROFIT_CLAIM"


def test_missing_policy_artifacts_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing cost-aware policy evaluator artifact"):
        write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)


def test_next_action_is_generic_factory_not_patchtst_rerun(tmp_path: Path) -> None:
    _write_policy_artifacts(tmp_path)

    result = write_m20_cost_aware_policy_adjudication(prediction_run_dir=tmp_path)
    next_actions = _read_csv(Path(result["output_files"]["next_actions_csv"]))

    assert result["next_required_action"] == (
        "MOVE_TO_GENERIC_STRATEGY_CONDITIONED_CANDIDATE_FACTORY"
    )
    assert "PATCHTST" not in next_actions[0]["action"]
