"""Tests for the M20 dynamic ensemble audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.ensemble.m20_dynamic_ensemble_audit import audit_m20_dynamic_ensemble


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M20 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)

    assert result["m20_state"] == "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M21_CONTINUAL_LEARNING_AUDIT"
    assert result["next_required_action"] == "AUDIT_M21_CONTINUAL_LEARNING_AND_GUARDED_WORKFLOW"
    assert result["m20_research_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m20_dynamic_ensemble_audit.json",
        "m20_dynamic_ensemble_audit.md",
        "ensemble_surface_audit.csv",
        "research_boundary_audit.csv",
        "promotion_boundary_audit.csv",
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


def test_missing_ensemble_service_produces_partial_state(tmp_path: Path) -> None:
    """Missing ensemble service should be reported as an M20 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/ensemble/service.py",
        _service_fixture().replace("class EnsembleService", ""),
    )

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)

    assert result["m20_state"] == "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "ensemble_service" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M20_DYNAMIC_ENSEMBLE_GAP_FILLS"


def test_missing_final_negative_summary_is_reported(tmp_path: Path) -> None:
    """The paused M20 research decision must remain explicit."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/training/m20_final_research_summary.py",
        "status = 'missing'\n",
    )

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "final_research_summary_writer" in gap_names


def test_missing_rollback_boundary_is_reported(tmp_path: Path) -> None:
    """Explicit rollback is part of the M20 ensemble control boundary."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/ensemble/rollback.py",
        "async def missing():\n    pass\n",
    )

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "ensemble_rollback_helper" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M20 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m20_dynamic_ensemble(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["ensemble_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "configs/ensemble.yaml", "candidate_roles:\n")
    _write(root / "app/ensemble/config.py", "class EnsembleConfig:\n    pass\n")
    _write(root / "app/ensemble/schemas.py", _schemas_fixture())
    _write(root / "app/ensemble/service.py", _service_fixture())
    _write(root / "app/ensemble/promote.py", _promote_fixture())
    _write(root / "app/ensemble/rollback.py", "async def rollback_active_profile():\n    pass\n")
    _write(
        root / "app/ensemble/research.py",
        "def build_m20_candidate_inventory_truth():\n    pass\n",
    )
    _write(root / "app/trading/repository.py", _repository_fixture())
    _write(root / "app/inference/service.py", _inference_service_fixture())
    _write(root / "app/trading/schemas.py", "EnsembleContextPayload\n")
    _write(root / "dashboards/data_sources.py", "ensemble_roster_status\n")
    _write(root / "app/training/m20_final_research_summary.py", _final_summary_fixture())
    _write(root / "app/training/m20_input_redesign_decision.py", _final_summary_fixture())
    _write(root / "app/training/m20_research_candidate_comparator.py", _comparator_fixture())
    _write(root / "app/training/m20_cost_aware_policy_adjudication.py", _policy_fixture())
    _write(root / "app/training/m20_research_feature_enrichment.py", "NO_RUNTIME_EFFECT\n")
    _write(root / "app/training/m20_strategy_candidate_v2_factory.py", "NOT_PROMOTABLE\n")
    _write(root / "tests/test_ensemble_packet2.py", "M20\n")
    _write(root / "tests/trading/test_decision_trace.py", "dynamic_ensemble\n")
    _write(root / "tests/test_dashboard_data_sources.py", "ensemble_roster_status\n")
    _write(root / "tests/test_training_m20_final_research_summary.py", _final_summary_fixture())
    _write(root / "tests/test_inference_api.py", "dynamic_ensemble\n")
    _write(root / "README.md", _readme_fixture())
    _write(root / "docs/training.md", _docs_fixture())
    _write(root / "PLANS.md", _plans_fixture())


def _schemas_fixture() -> str:
    return (
        "class EnsembleProfileRecord:\n    pass\n"
        "class EnsembleResult:\n    pass\n"
        "class EnsembleContextPayload:\n    pass\n"
        "class EnsemblePromotionDecisionRecord:\n    pass\n"
        "class EnsembleChallengerRunRecord:\n    pass\n"
        "rollback_target_profile_id\n"
    )


def _service_fixture() -> str:
    return (
        "class EnsembleService:\n"
        "    async def resolve_ensemble(self):\n        pass\n"
        "def build_ensemble_fallback_result():\n    pass\n"
        "agreement_multiplier\n"
    )


def _promote_fixture() -> str:
    return "def build_draft_profile():\n    pass\nasync def activate_draft_profile():\n    pass\n"


def _repository_fixture() -> str:
    return (
        "save_ensemble_profile\n"
        "save_ensemble_promotion_decision\n"
        "save_ensemble_challenger_run\n"
    )


def _inference_service_fixture() -> str:
    return (
        "_resolve_runtime_ensemble_state\n"
        "dynamic_ensemble\n"
        "ensemble_roster_status\n"
        "ACTIVE_WEAK\n"
    )


def _final_summary_fixture() -> str:
    return "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n"


def _comparator_fixture() -> str:
    return "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE\n"


def _policy_fixture() -> str:
    return "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE\n"


def _readme_fixture() -> str:
    return (
        "candidate ecosystem remains narrow\n"
        "KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY\n"
    )


def _docs_fixture() -> str:
    return "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n"


def _plans_fixture() -> str:
    return (
        "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_CONSOLIDATED\n"
        "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
