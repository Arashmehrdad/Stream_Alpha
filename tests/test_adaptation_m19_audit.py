"""Tests for the M19 bounded adaptation audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.adaptation.m19_bounded_adaptation_audit import audit_m19_bounded_adaptation


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M19 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)

    assert result["m19_state"] == "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M20_DYNAMIC_ENSEMBLE_AUDIT"
    assert (
        result["next_required_action"]
        == "AUDIT_M20_DYNAMIC_ENSEMBLE_AND_RESEARCH_BOUNDARIES"
    )
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m19_bounded_adaptation_audit.json",
        "m19_bounded_adaptation_audit.md",
        "adaptation_surface_audit.csv",
        "drift_control_audit.csv",
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


def test_missing_adaptation_service_produces_partial_state(tmp_path: Path) -> None:
    """Missing adaptation service should be reported as an M19 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/adaptation/service.py",
        _service_fixture().replace("class AdaptationService", ""),
    )

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)

    assert result["m19_state"] == "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "adaptation_service" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M19_BOUNDED_ADAPTATION_GAP_FILLS"


def test_missing_runtime_truth_writer_is_reported(tmp_path: Path) -> None:
    """Runtime-generated M19 drift/performance truth is required."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/adaptation/service.py",
        _service_fixture().replace("async def write_runtime_persisted_truth", ""),
    )

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "runtime_truth_writer" in gap_names


def test_missing_rollback_boundary_is_reported(tmp_path: Path) -> None:
    """Explicit rollback is part of the bounded adaptation safety boundary."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/adaptation/service.py",
        _service_fixture().replace("async def rollback_active_profile", ""),
    )

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "rollback_active_profile" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M19 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m19_bounded_adaptation(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["adaptation_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "configs/adaptation.yaml", _config_fixture())
    _write(root / "app/adaptation/config.py", "class AdaptationConfig:\n    pass\n")
    _write(root / "app/adaptation/schemas.py", _schemas_fixture())
    _write(root / "app/adaptation/service.py", _service_fixture())
    _write(root / "app/adaptation/drift.py", _drift_fixture())
    _write(root / "app/adaptation/performance.py", _performance_fixture())
    _write(root / "app/adaptation/thresholds.py", "def bounded_effective_thresholds():\n    pass\n")
    _write(root / "app/adaptation/sizing.py", "def bounded_size_multiplier():\n    pass\n")
    _write(root / "app/adaptation/calibration.py", "def apply_calibration():\n    pass\n")
    _write(root / "app/adaptation/promotion.py", "def decide_promotion():\n    pass\n")
    _write(root / "app/trading/repository.py", _repository_fixture())
    _write(root / "app/inference/main.py", _inference_main_fixture())
    _write(root / "app/trading/runner.py", "write_runtime_persisted_truth\n")
    _write(root / "dashboards/data_sources.py", "/adaptation/performance\n")
    _write(root / "tests/test_adaptation_service.py", "M19\n")
    _write(root / "tests/test_adaptation_repository.py", "M19\n")
    _write(root / "tests/test_inference_api.py", "/adaptation/summary\n")
    _write(
        root / "tests/trading/test_runner_idempotency.py",
        "test_runner_persists_runtime_m19_drift_and_performance_truth\n",
    )
    _write(root / "tests/test_dashboard_data_sources.py", "/adaptation/drift\n")
    _write(root / "README.md", _readme_fixture())
    _write(root / "docs/training.md", _docs_fixture())
    _write(root / "PLANS.md", "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_CONSOLIDATED\n")


def _config_fixture() -> str:
    return (
        "drift:\n"
        "  warning: 0.1\n"
        "  breach: 0.2\n"
        "  minimum_reference_samples: 10\n"
        "  minimum_live_samples: 10\n"
        "freeze_on_drift_breach: true\n"
        "freeze_on_degraded_reliability: true\n"
        "promotion_thresholds:\n"
    )


def _schemas_fixture() -> str:
    return (
        "class AdaptiveDriftRecord:\n    pass\n"
        "class AdaptivePerformanceWindow:\n    pass\n"
        "class AdaptiveProfileRecord:\n    pass\n"
        "class AdaptivePromotionDecisionRecord:\n    pass\n"
        "class AppliedAdaptation:\n    pass\n"
        "bounded_by_m10\n"
    )


def _service_fixture() -> str:
    return (
        "class AdaptationService:\n"
        "    async def resolve_applied_adaptation(self):\n        pass\n"
        "    async def write_runtime_persisted_truth(self):\n        pass\n"
        "    async def rollback_active_profile(self):\n        pass\n"
        "    def _write_drift_summary_artifact(self):\n        pass\n"
        "    def _write_performance_summary_artifact(self):\n        pass\n"
    )


def _drift_fixture() -> str:
    return "population_stability_index\ndef classify_drift():\n    pass\n"


def _performance_fixture() -> str:
    return "def build_rolling_performance_windows():\n    pass\n"


def _repository_fixture() -> str:
    return (
        "load_feature_rows_for_adaptation\n"
        "save_adaptive_drift_state\n"
        "save_adaptive_performance_window\n"
        "rollback_adaptive_profile\n"
        "save_adaptive_promotion_decision\n"
    )


def _inference_main_fixture() -> str:
    return "/adaptation/summary\n/adaptation/promotions\n"


def _readme_fixture() -> str:
    return "M19 bounded adaptation\n"


def _docs_fixture() -> str:
    return "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\nM19\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
