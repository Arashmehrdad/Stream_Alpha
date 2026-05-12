"""Tests for the M18 evaluation reporting audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.evaluation.m18_evaluation_reporting_audit import audit_m18_evaluation_reporting


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M18 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)

    assert result["m18_state"] == "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M19_BOUNDED_ADAPTATION_AUDIT"
    assert result["next_required_action"] == "AUDIT_M19_BOUNDED_ADAPTATION_AND_DRIFT_CONTROLS"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m18_evaluation_reporting_audit.json",
        "m18_evaluation_reporting_audit.md",
        "evaluation_surface_audit.csv",
        "reporting_contract_audit.csv",
        "degradation_boundary_audit.csv",
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


def test_missing_evaluation_service_produces_partial_state(tmp_path: Path) -> None:
    """Missing evaluation service should be reported as an M18 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/evaluation/service.py",
        _service_fixture().replace("class EvaluationService", ""),
    )

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)

    assert result["m18_state"] == "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "evaluation_service" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M18_EVALUATION_REPORTING_GAP_FILLS"


def test_missing_probe_policy_persistence_is_reported(tmp_path: Path) -> None:
    """Probe policy truth must be visible to evaluation as observed context."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/evaluation/repository.py",
        _repository_fixture().replace("probe_policy_active", "probe_policy_missing"),
    )

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "probe_policy_is_observed_not_authoritative" in gap_names


def test_missing_degradation_reporting_is_reported(tmp_path: Path) -> None:
    """Paper-to-live degradation reporting is part of the M18 contract."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/evaluation/metrics.py",
        _metrics_fixture().replace("def compute_paper_to_live_degradation", "def missing"),
    )

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "paper_to_live_degradation_metric" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M18 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m18_evaluation_reporting(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["evaluation_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(
        root / "configs/evaluation.yaml",
        (
            "schema_version: m18_evaluation_config_v1\n"
            "latency_drift_ms_threshold: 25.0\n"
            "fill_price_drift_bps_threshold: 10.0\n"
            "slippage_drift_bps_threshold: 5.0\n"
        ),
    )
    _write(root / "app/evaluation/config.py", "class EvaluationConfig:\n    pass\n")
    _write(root / "app/evaluation/schemas.py", _schemas_fixture())
    _write(root / "app/evaluation/service.py", _service_fixture())
    _write(root / "app/evaluation/repository.py", _repository_fixture())
    _write(root / "app/evaluation/__main__.py", "EvaluationService\n")
    _write(root / "app/evaluation/artifacts.py", _artifacts_fixture())
    _write(root / "app/evaluation/normalize.py", "def build_decision_opportunities():\n    pass\n")
    _write(root / "app/evaluation/matching.py", "def build_comparison_windows():\n    pass\n")
    _write(root / "app/evaluation/metrics.py", _metrics_fixture())
    _write(
        root / "app/training/live_policy_challenger.py",
        "class LivePolicyChallengerTracker:\n    pass\n",
    )
    _write(root / "app/trading/runner.py", "def _observe_live_policy_challengers():\n    pass\n")
    _write(
        root / "scripts/show_live_policy_challengers.ps1",
        "Live policy challenger scoreboard\n",
    )
    _write(root / "scripts/status_paper_vps.ps1", "challenger artifacts exist\n")
    _write(
        root / "scripts/show_live_policy_challengers_vps.ps1",
        "Paper VPS challenger scoreboard\n",
    )
    _write(root / "tests/test_evaluation_service.py", "M18\n")
    _write(root / "tests/test_evaluation_metrics.py", "M18\n")
    _write(root / "tests/test_evaluation_matching.py", "M18\n")
    _write(root / "tests/test_evaluation_normalize.py", "M18\n")
    _write(root / "tests/trading/test_runner_idempotency.py", "policy_challengers\n")
    _write(root / "tests/test_training_scripts.py", "show_live_policy_challengers.ps1\n")
    _write(root / "README.md", "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _schemas_fixture() -> str:
    return (
        "paper_to_tiny_live\n"
        "fill_quality\n"
        "SLIPPAGE_DRIFT\n"
        "class EvaluationRequest:\n    pass\n"
        "class DecisionOpportunity:\n    pass\n"
        "class DivergenceEvent:\n    pass\n"
        "class EvaluationManifest:\n    pass\n"
        "class EvaluationReport:\n    pass\n"
        "class PaperToLiveDegradationReport:\n    pass\n"
    )


def _service_fixture() -> str:
    return (
        "class EvaluationService:\n"
        "    async def generate_run(self):\n        pass\n"
        "EVALUATION_INDEX_SCHEMA_VERSION\n"
        "EXPERIMENT_INDEX_SCHEMA_VERSION\n"
        "PROMOTION_INDEX_SCHEMA_VERSION\n"
        "load_current_registry_entry\n"
    )


def _repository_fixture() -> str:
    return "class EvaluationRepository:\n    pass\nprobe_policy_active\n"


def _artifacts_fixture() -> str:
    return (
        "def write_evaluation_artifacts():\n    pass\n"
        "evaluation_manifest.json\n"
        "evaluation_report.json\n"
        "decision_opportunities.csv\n"
        "performance_by_asset.csv\n"
        "performance_by_regime.csv\n"
        "divergence_events.csv\n"
        "latency_distribution.csv\n"
        "slippage_distribution.csv\n"
        "paper_to_live_degradation.json\n"
    )


def _metrics_fixture() -> str:
    return (
        "def compute_paper_to_live_degradation():\n    pass\n"
        "def compute_cost_aware_precision_by_mode():\n    pass\n"
        "def compute_layer_comparison():\n    pass\n"
        "def compute_uptime_and_failures():\n    pass\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
