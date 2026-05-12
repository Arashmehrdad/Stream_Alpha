"""Tests for final M20 research summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.m20_final_research_summary import write_m20_final_research_summary

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(base: Path) -> Path:
    source = base / "source"
    research = source / "research_labels" / "vol_scaled"
    for name in (
        "strategy_candidate_v2_factory",
        "strategy_candidate_v2_refined_factory",
        "decision_policy_eval",
        "trading_aware_policy_eval",
        "m20_research_dashboard",
        "research_candidate_comparator",
    ):
        _write_json(
            research / name / "recommendation.json",
            {
                "recommendation": "PAUSE_OR_NEGATIVE",
                "next_required_action": "PAUSE_OR_NEGATIVE",
            },
        )
    _write_json(
        research / "m20_input_redesign_decision" / "recommendation.json",
        {
            "recommendation": "PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS",
            "next_required_action": "PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS",
        },
    )
    _write_json(
        research / "m20_input_redesign_decision" / "m20_input_redesign_decision.json",
        {
            "final_decision": "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        },
    )
    return source


def test_final_summary_writes_terminal_negative_decision(tmp_path: Path) -> None:
    source = _source(tmp_path)

    result = write_m20_final_research_summary(source_run_dir=source)

    assert result["final_m20_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert result["status"] == "RESEARCH_ONLY_NEGATIVE_RESULT"
    assert result["terminal_decision"]["promotable"] is False
    assert result["terminal_decision"]["profit_claim"] is False


def test_final_summary_marks_missing_optional_artifacts(tmp_path: Path) -> None:
    source = _source(tmp_path)

    result = write_m20_final_research_summary(source_run_dir=source)
    statuses = {row["artifact_status"] for row in result["research_sequence_rollup"]}

    assert "MISSING_OPTIONAL_ARTIFACT" in statuses


def test_final_summary_data_upgrade_plan_is_planning_only(tmp_path: Path) -> None:
    source = _source(tmp_path)

    result = write_m20_final_research_summary(source_run_dir=source)

    assert result["data_upgrade_plan"]
    assert {
        row["plan_status"] for row in result["data_upgrade_plan"]
    } == {"PLANNING_ONLY_NOT_IMPLEMENTED"}
    assert all(row["runtime_effect"] == "NO_RUNTIME_EFFECT" for row in result["data_upgrade_plan"])


def test_final_summary_preserves_honesty_flags(tmp_path: Path) -> None:
    source = _source(tmp_path)

    result = write_m20_final_research_summary(source_run_dir=source)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_BACKTEST" in result["honesty_flags"]
    assert "NOT_RUNTIME_READY" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert (
        result["project_route_recommendation"]
        == "KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY"
    )
