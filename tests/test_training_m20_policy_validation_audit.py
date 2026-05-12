"""Tests for M20 policy validation audit."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_policy_validation_audit import audit_m20_policy_validation

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _eval_dir(base: Path) -> Path:
    eval_dir = base / "research_labels" / "vol_scaled" / "decision_policy_eval"
    _write_csv(
        eval_dir / "policy_metrics.csv",
        [
            {
                "policy_name": "p1",
                "policy_family": "PROBABILITY_THRESHOLD",
                "selected_rows": 1200,
                "mean_net_value_proxy": -0.01,
                "classification": "POLICY_ECONOMICS_NEGATIVE",
            },
            {
                "policy_name": "p2",
                "policy_family": "CANDIDATE_EVENT_POLICY",
                "selected_rows": 12,
                "mean_net_value_proxy": 0.01,
                "classification": "POLICY_LOW_SAMPLE",
            },
        ],
    )
    _write_csv(
        eval_dir / "candidate_decisions.csv",
        [
            {
                "policy_name": "p1",
                "policy_decision": "POLICY_ECONOMICS_NEGATIVE",
            },
            {
                "policy_name": "p2",
                "policy_decision": "POLICY_LOW_SAMPLE",
            },
        ],
    )
    _write_csv(
        eval_dir / "baseline_comparison.csv",
        [
            {
                "policy_name": "p1",
                "baseline_policy_name": "baseline",
                "mean_net_delta_vs_baseline": -0.01,
            }
        ],
    )
    _write_csv(
        eval_dir / "search_breadth.csv",
        [
            {
                "policy_configurations_tried": 2,
                "candidate_definitions_referenced": 1,
            }
        ],
    )
    return eval_dir


def test_policy_validation_audit_reports_search_breadth_and_low_sample(
    tmp_path: Path,
) -> None:
    _eval_dir(tmp_path)

    result = audit_m20_policy_validation(source_run_dir=tmp_path)

    assert result["search_breadth_audit"][0]["warning"] == "MULTIPLE_COMPARISON_RESEARCH_ONLY"
    assert result["low_sample_warnings"][0]["policy_name"] == "p2"
    assert result["recommendation"] == "DESIGN_TRADING_AWARE_RESEARCH_LABELS"


def test_policy_validation_audit_preserves_research_statuses(tmp_path: Path) -> None:
    _eval_dir(tmp_path)

    result = audit_m20_policy_validation(source_run_dir=tmp_path)

    assert "NO_RUNTIME_EFFECT" in result["overall_status"]
    assert "NOT_PROMOTABLE" in result["overall_status"]
    assert "NO_PROFIT_CLAIM" in result["overall_status"]
    assert result["paired_comparison_readiness"][0]["comparison_rows"] == 1
