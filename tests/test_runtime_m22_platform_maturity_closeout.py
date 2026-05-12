"""Tests for M22 platform maturity closeout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from app.runtime.m22_platform_maturity_closeout import (
    AUDIT_SPECS,
    write_m22_platform_maturity_closeout,
)


def test_closeout_reports_complete_when_all_audits_are_consolidated(tmp_path: Path) -> None:
    """Complete M9-M21 fixtures should close the platform audit chain."""
    _write_complete_audit_fixtures(tmp_path)

    result = write_m22_platform_maturity_closeout(repo_root=tmp_path)

    assert (
        result["platform_maturity_state"]
        == "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_COMPLETE"
    )
    assert result["gap_count"] == 0
    assert result["consolidated_milestone_count"] == len(AUDIT_SPECS)
    assert result["recommendation"] == "PLAN_DATA_UPGRADE_BEFORE_REOPENING_ALPHA_RESEARCH"
    assert result["next_required_action"] == "DESIGN_RESEARCH_DATA_UPGRADE_FEASIBILITY_PLAN"


def test_closeout_writes_expected_artifacts(tmp_path: Path) -> None:
    """The closeout should persist its deterministic artifact contract."""
    _write_complete_audit_fixtures(tmp_path)

    result = write_m22_platform_maturity_closeout(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m22_platform_maturity_closeout.json",
        "m22_platform_maturity_closeout.md",
        "audit_evidence_index.csv",
        "milestone_status_rollup.csv",
        "remaining_limitations.csv",
        "non_claims.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_audit_artifact_blocks_complete_state(tmp_path: Path) -> None:
    """A missing audit artifact should prevent the complete state."""
    _write_complete_audit_fixtures(tmp_path)
    missing_report = (
        tmp_path
        / "artifacts/platform_maturity/m14/explainability_audit"
        / "m14_explainability_audit.json"
    )
    missing_report.unlink()

    result = write_m22_platform_maturity_closeout(repo_root=tmp_path)

    assert result["platform_maturity_state"] == "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_PARTIAL"
    statuses = {row["milestone"]: row["status"] for row in result["milestone_status_rollup"]}
    assert statuses["M14"] == "MISSING_AUDIT_ARTIFACT"
    assert result["recommendation"] == "RESOLVE_PLATFORM_MATURITY_AUDIT_GAPS"


def test_non_consolidated_audit_blocks_complete_state(tmp_path: Path) -> None:
    """A partial audit artifact should remain visible in the rollup."""
    _write_complete_audit_fixtures(tmp_path)
    report_path = (
        tmp_path
        / "artifacts/platform_maturity/m18/evaluation_reporting_audit"
        / "m18_evaluation_reporting_audit.json"
    )
    payload = json.loads(report_path.read_text())
    payload["m18_state"] = "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_PARTIAL"
    payload["gap_count"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = write_m22_platform_maturity_closeout(repo_root=tmp_path)

    statuses = {row["milestone"]: row["status"] for row in result["milestone_status_rollup"]}
    assert statuses["M18"] == "NOT_CONSOLIDATED"
    assert result["gap_count"] == 1


def test_closeout_preserves_research_only_non_claims(tmp_path: Path) -> None:
    """M22 closeout should not create runtime, promotion, or profit claims."""
    _write_complete_audit_fixtures(tmp_path)

    result = write_m22_platform_maturity_closeout(repo_root=tmp_path)

    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    claims = {row["claim"] for row in result["non_claims"]}
    assert {"NOT_BACKTEST", "NOT_RUNTIME_READY", "NOT_PROMOTABLE", "NO_PROFIT_CLAIM"} <= claims


def _write_complete_audit_fixtures(root: Path) -> None:
    for milestone, directory, report_name in AUDIT_SPECS:
        artifact_dir = root / "artifacts" / "platform_maturity" / milestone / directory
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state_key = f"{milestone}_state"
        report = {
            state_key: f"{milestone.upper()}_CONSOLIDATED",
            "gap_count": 0,
            "m20_research_decision": "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        }
        recommendation = {
            "recommendation": f"{milestone.upper()}_DONE",
            "next_required_action": "NEXT",
            "runtime_ready": False,
            "promotable": False,
            "profitability_claim": False,
        }
        (artifact_dir / report_name).write_text(json.dumps(report), encoding="utf-8")
        (artifact_dir / "recommendation.json").write_text(
            json.dumps(recommendation),
            encoding="utf-8",
        )
