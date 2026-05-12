"""Tests for the research data-upgrade feasibility plan."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.research_data_upgrade_plan import write_research_data_upgrade_plan


def test_data_upgrade_plan_writes_required_families(tmp_path: Path) -> None:
    """The plan should include all required future input families."""
    _write_source_artifacts(tmp_path)

    result = write_research_data_upgrade_plan(repo_root=tmp_path)

    families = {row["data_family"] for row in result["required_data_families"]}
    assert "order_book_depth" in families
    assert "spread_liquidity" in families
    assert "trade_flow_imbalance" in families
    assert "same_venue_execution_quality" in families
    assert "lower_turnover_event_labels" in families
    assert result["recommendation"] == "PLAN_DATA_UPGRADE_IMPLEMENTATION_BATCHES"


def test_data_upgrade_plan_writes_expected_artifacts(tmp_path: Path) -> None:
    """The planner should persist its deterministic artifact contract."""
    _write_source_artifacts(tmp_path)

    result = write_research_data_upgrade_plan(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "research_data_upgrade_plan.json",
        "research_data_upgrade_plan.md",
        "required_data_families.csv",
        "source_feasibility.csv",
        "leakage_and_runtime_boundary_audit.csv",
        "implementation_batches.csv",
        "blocked_routes.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}


def test_data_upgrade_plan_preserves_runtime_boundaries(tmp_path: Path) -> None:
    """The plan must remain planning-only and non-runtime."""
    _write_source_artifacts(tmp_path)

    result = write_research_data_upgrade_plan(repo_root=tmp_path)

    assert "PLANNING_ONLY" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert {row["runtime_effect"] for row in result["required_data_families"]} == {
        "NO_RUNTIME_EFFECT"
    }
    assert {row["training_boundary"] for row in result["leakage_and_runtime_boundary_audit"]} == {
        "NO_TRAINING_IN_THIS_PLAN"
    }


def test_data_upgrade_plan_reports_source_decisions_required(tmp_path: Path) -> None:
    """Microstructure families should be blocked until data sources are selected."""
    _write_source_artifacts(tmp_path)

    result = write_research_data_upgrade_plan(repo_root=tmp_path)

    blocked = {row["blocked_route"] for row in result["blocked_routes"]}
    assert "implement_order_book_depth" in blocked
    assert "implement_spread_liquidity" in blocked
    assert "implement_trade_flow_imbalance" in blocked
    assert result["next_required_action"] == "DESIGN_MARKET_MICROSTRUCTURE_RESEARCH_INGESTION_PLAN"


def test_recommendation_json_has_no_promotion_or_profit_claim(tmp_path: Path) -> None:
    """Recommendation payload should not claim readiness or profitability."""
    _write_source_artifacts(tmp_path)

    result = write_research_data_upgrade_plan(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent
    recommendation = json.loads((output_dir / "recommendation.json").read_text())

    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def _write_source_artifacts(root: Path) -> None:
    m20_dir = (
        root
        / "artifacts/training/m20/20260506T054337Z/research_labels/vol_scaled"
        / "m20_final_research_summary"
    )
    m22_dir = root / "artifacts/platform_maturity/m22/platform_maturity_closeout"
    m20_dir.mkdir(parents=True, exist_ok=True)
    m22_dir.mkdir(parents=True, exist_ok=True)
    (root / "app/ingestion").mkdir(parents=True, exist_ok=True)
    (root / "app/features").mkdir(parents=True, exist_ok=True)
    (m20_dir / "m20_final_research_summary.json").write_text(
        json.dumps({"final_m20_decision": "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"}),
        encoding="utf-8",
    )
    (m22_dir / "m22_platform_maturity_closeout.json").write_text(
        json.dumps(
            {
                "platform_maturity_state": (
                    "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_COMPLETE"
                )
            }
        ),
        encoding="utf-8",
    )
