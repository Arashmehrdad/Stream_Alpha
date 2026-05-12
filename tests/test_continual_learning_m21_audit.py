"""Tests for the M21 continual-learning guarded-workflow audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.continual_learning.m21_continual_learning_audit import (
    audit_m21_continual_learning,
)


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M21 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m21_continual_learning(repo_root=tmp_path)

    assert result["m21_state"] == "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_COMPLETE"
    assert result["next_required_action"] == "PLAN_NEXT_PLATFORM_MATURITY_OR_DATA_UPGRADE_ROUTE"
    assert result["m20_research_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m21_continual_learning(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m21_continual_learning_audit.json",
        "m21_continual_learning_audit.md",
        "continual_learning_surface_audit.csv",
        "guarded_workflow_audit.csv",
        "drift_cap_audit.csv",
        "operator_visibility_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_service_produces_partial_state(tmp_path: Path) -> None:
    """Missing continual-learning service should be reported as an M21 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/continual_learning/service.py",
        _service_fixture().replace("class ContinualLearningService", ""),
    )

    result = audit_m21_continual_learning(repo_root=tmp_path)

    assert result["m21_state"] == "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "continual_learning_service" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M21_CONTINUAL_LEARNING_GAP_FILLS"


def test_missing_operator_confirmation_block_is_reported(tmp_path: Path) -> None:
    """Operator confirmation is required for guarded M21 workflow authority."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/continual_learning/service.py",
        _service_fixture().replace("CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED", ""),
    )

    result = audit_m21_continual_learning(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "operator_confirmation_required" in gap_names


def test_missing_drift_cap_writer_is_reported(tmp_path: Path) -> None:
    """Runtime drift-cap persistence from M19 truth is an M21 requirement."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/continual_learning/service.py",
        _service_fixture().replace("async def write_runtime_drift_caps", ""),
    )

    result = audit_m21_continual_learning(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "runtime_drift_cap_writer" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M21 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m21_continual_learning(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["continual_learning_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "configs/continual_learning.yaml", _config_fixture())
    _write(root / "app/continual_learning/config.py", "class ContinualLearningConfig:\n    pass\n")
    _write(root / "app/continual_learning/schemas.py", _schemas_fixture())
    _write(root / "app/continual_learning/service.py", _service_fixture())
    _write(root / "app/trading/repository.py", _repository_fixture())
    _write(root / "app/inference/main.py", _inference_fixture())
    _write(root / "app/trading/runner.py", "write_runtime_drift_caps\n")
    _write(root / "app/trading/schemas.py", "ContinualLearningContextPayload\n")
    _write(root / "dashboards/data_sources.py", "/continual-learning/drift-caps\n")
    _write(root / "tests/test_continual_learning_service.py", "promote_profile\n")
    _write(root / "tests/test_inference_api.py", "/continual-learning/summary\n")
    _write(root / "tests/test_dashboard_data_sources.py", "/continual-learning/drift-caps\n")
    _write(
        root / "tests/trading/test_runner_idempotency.py",
        "test_runner_persists_runtime_m21_drift_caps\n",
    )
    _write(root / "tests/trading/test_decision_trace.py", "ContinualLearningContextPayload\n")
    _write(root / "README.md", _readme_fixture())
    _write(root / "docs/training.md", _docs_fixture())
    _write(root / "PLANS.md", _plans_fixture())


def _config_fixture() -> str:
    return "candidate_types:\n  - CALIBRATION_OVERLAY\n  - INCREMENTAL_SHADOW_CHALLENGER\n"


def _schemas_fixture() -> str:
    return (
        "class ContinualLearningExperimentRecord:\n    pass\n"
        "class ContinualLearningProfileRecord:\n    pass\n"
        "class ContinualLearningDriftCapRecord:\n    pass\n"
        "class ContinualLearningPromotionDecisionRecord:\n    pass\n"
        "class ContinualLearningEventRecord:\n    pass\n"
        "class ContinualLearningContextPayload:\n    pass\n"
        "class ContinualLearningPromoteProfileRequest:\n    pass\n"
        "class ContinualLearningRollbackRequest:\n    pass\n"
        "class ContinualLearningWorkflowResponse:\n    pass\n"
        "INCREMENTAL_SHADOW_CHALLENGER\n"
    )


def _service_fixture() -> str:
    return (
        "CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED\n"
        "CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED\n"
        "CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED\n"
        "CONTINUAL_LEARNING_DRIFT_CAP_BREACHED\n"
        "CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS\n"
        "CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS\n"
        "class ContinualLearningService:\n"
        "    async def resolve_runtime_context(self):\n        pass\n"
        "    async def write_runtime_drift_caps(self):\n        pass\n"
        "    async def summary(self):\n        pass\n"
        "    async def experiments(self):\n        pass\n"
        "    async def profiles(self):\n        pass\n"
        "    async def drift_caps(self):\n        pass\n"
        "    async def promotions(self):\n        pass\n"
        "    async def events(self):\n        pass\n"
        "    async def promote_profile(self):\n        pass\n"
        "    async def rollback_profile(self):\n        pass\n"
        "load_latest_adaptive_drift_state\n"
        "save_continual_learning_drift_cap\n"
        "def _drift_cap_from_adaptive_state():\n    pass\n"
        "_write_drift_caps_summary_artifact\n"
    )


def _repository_fixture() -> str:
    return (
        "save_continual_learning_drift_cap\n"
        "load_latest_continual_learning_drift_cap\n"
        "load_all_continual_learning_drift_caps\n"
    )


def _inference_fixture() -> str:
    return (
        "/continual-learning/summary\n"
        "/continual-learning/promotions/promote-profile\n"
        "/continual-learning/promotions/rollback-active-profile\n"
    )


def _readme_fixture() -> str:
    return "/continual-learning/promotions/promote-profile\n"


def _docs_fixture() -> str:
    return "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n"


def _plans_fixture() -> str:
    return (
        "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_CONSOLIDATED\n"
        "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
