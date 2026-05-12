"""Tests for the isolated research microstructure capture plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.market_microstructure_capture_plan import (
    write_microstructure_capture_plan,
)


def test_capture_plan_writes_expected_artifacts(tmp_path: Path) -> None:
    """The DU6 capture plan should persist its deterministic artifact contract."""
    output_dir = tmp_path / "capture_plan"

    result = write_microstructure_capture_plan(repo_root=tmp_path, output_dir=output_dir)

    assert result["capture_plan_status"] == "ISOLATED_RESEARCH_CAPTURE_PLAN_DEFINED"
    for filename in (
        "manifest.json",
        "microstructure_capture_plan.json",
        "microstructure_capture_plan.md",
        "service_contract.csv",
        "isolation_boundary.csv",
        "storage_plan.csv",
        "operator_runbook.csv",
        "safety_gate_plan.csv",
        "implementation_batches.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_capture_plan_does_not_implement_capture_or_runtime_wiring(tmp_path: Path) -> None:
    """The plan must remain planning-only and non-runtime."""
    output_dir = tmp_path / "capture_plan"

    result = write_microstructure_capture_plan(repo_root=tmp_path, output_dir=output_dir)

    assert result["capture_implemented"] is False
    assert result["runtime_wiring_changed"] is False
    boundaries = _read_csv(output_dir / "isolation_boundary.csv")
    boundary_names = {row["boundary"] for row in boundaries}
    assert {"process", "tables", "topics", "runtime", "m20"} <= boundary_names
    assert {row["runtime_effect"] for row in boundaries} == {"NO_RUNTIME_EFFECT"}


def test_capture_plan_requires_separate_approval_before_implementation(
    tmp_path: Path,
) -> None:
    """Implementation should remain blocked until the user explicitly approves it."""
    output_dir = tmp_path / "capture_plan"

    result = write_microstructure_capture_plan(repo_root=tmp_path, output_dir=output_dir)

    assert result["recommendation"] == (
        "REQUIRE_APPROVAL_BEFORE_RESEARCH_CAPTURE_IMPLEMENTATION"
    )
    assert result["next_required_action"] == (
        "APPROVE_OR_PAUSE_ISOLATED_MICROSTRUCTURE_CAPTURE"
    )
    blocked = _read_csv(output_dir / "blocked_decisions.csv")
    assert "capture_implementation" in {row["decision"] for row in blocked}


def test_capture_plan_preserves_research_only_non_claims(tmp_path: Path) -> None:
    """Outputs should keep M20 paused and avoid runtime/promotion/profit claims."""
    output_dir = tmp_path / "capture_plan"

    write_microstructure_capture_plan(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_capture_plan.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in (
        "RESEARCH_ONLY",
        "NO_RUNTIME_EFFECT",
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert report["m20_research_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_capture_plan_keeps_existing_ingestion_contracts_unchanged(tmp_path: Path) -> None:
    """Storage plan should only mention future research tables."""
    output_dir = tmp_path / "capture_plan"

    write_microstructure_capture_plan(repo_root=tmp_path, output_dir=output_dir)

    storage_rows = _read_csv(output_dir / "storage_plan.csv")
    table_names = {row["table_name"] for row in storage_rows}
    assert "raw_trades" not in table_names
    assert "raw_ohlc" not in table_names
    assert "feature_ohlc" not in table_names
    assert {row["mutates_existing_tables"] for row in storage_rows} == {"False"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
