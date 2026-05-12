"""Tests for the M14 explainability audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.explainability.m14_explainability_audit import audit_m14_explainability


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M14 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m14_explainability(repo_root=tmp_path)

    assert result["m14_state"] == "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M15_OPERATOR_CONSOLE_AUDIT"
    assert result["next_required_action"] == "AUDIT_M15_OPERATOR_CONSOLE_AND_VISIBILITY_CONTROLS"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m14_explainability(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m14_explainability_audit.json",
        "m14_explainability_audit.md",
        "explainability_surface_audit.csv",
        "decision_trace_audit.csv",
        "linkage_audit.csv",
        "rationale_artifact_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_decision_trace_builder_produces_partial_state(tmp_path: Path) -> None:
    """Missing trace construction should be a blocking M14 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/decision_trace.py",
        _decision_trace_fixture().replace("def build_initial_decision_trace", ""),
    )

    result = audit_m14_explainability(repo_root=tmp_path)

    assert result["m14_state"] == "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "initial_decision_trace_builder" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M14_EXPLAINABILITY_GAP_FILLS"


def test_missing_rationale_writer_is_reported(tmp_path: Path) -> None:
    """Rationale report writing is required for M14 operator evidence."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/decision_trace.py",
        _decision_trace_fixture().replace("def write_rationale_reports", ""),
    )

    result = audit_m14_explainability(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "rationale_report_writer" in gap_names


def test_missing_order_trace_linkage_is_reported(tmp_path: Path) -> None:
    """Order request trace linkage is required for M14 decision lineage."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/repository.py",
        _repository_fixture().replace("order_request.decision_trace_id", ""),
    )

    result = audit_m14_explainability(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "order_request_trace_linkage" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M14 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m14_explainability(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["explainability_surface_audit_csv"])
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
        root / "configs/explainability.yaml",
        "schema_version: m14_explainability_v1\n"
        "artifact_root\n"
        "reference_filename\n"
        "top_feature_count\n",
    )
    _write(root / "app/explainability/config.py", "class ExplainabilityConfig:\n    pass\n")
    _write(root / "app/explainability/schemas.py", _schemas_fixture())
    _write(root / "app/explainability/service.py", _service_fixture())
    _write(root / "app/trading/schemas.py", "class DecisionTraceRecord:\n    pass\n")
    _write(root / "app/trading/decision_trace.py", _decision_trace_fixture())
    _write(root / "app/trading/repository.py", _repository_fixture())
    _write(root / "app/inference/service.py", _inference_fixture())
    _write(root / "app/trading/runner.py", _runner_fixture())
    _write(root / "app/trading/execution.py", "decision_trace_id\n")
    _write(root / "app/trading/risk_engine.py", "decision_trace_id\n")
    _write(
        root / "tests/test_dashboard_data_sources.py",
        "recent_decision_traces\njson_report_path\n",
    )
    _write(root / "README.md", "M13_RELIABILITY_RECOVERY_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _schemas_fixture() -> str:
    return (
        "class PredictionExplanation:\n    pass\n"
        "class TopFeatureContribution:\n    pass\n"
        "class SignalExplanation:\n    pass\n"
        "class DecisionTracePayload:\n    pass\n"
        "class DecisionTracePrediction:\n    pass\n"
        "class DecisionTraceSignal:\n    pass\n"
        "class DecisionTraceRisk:\n    pass\n"
        "class DecisionTraceBlockedTrade:\n    pass\n"
        "class ThresholdSnapshot:\n    pass\n"
        "class RegimeReason:\n    pass\n"
    )


def _service_fixture() -> str:
    return (
        "REFERENCE_ABLATION_METHOD = 'ONE_AT_A_TIME_REFERENCE_ABLATION'\n"
        "def build_prediction_details():\n    pass\n"
        "def build_signal_explanation():\n    pass\n"
        "def build_regime_reason():\n    pass\n"
    )


def _decision_trace_fixture() -> str:
    return (
        "DECISION_TRACE_SCHEMA_VERSION = 'm14_decision_trace_v1'\n"
        "RATIONALE_REPORT_SCHEMA_VERSION = 'm14_rationale_report_v1'\n"
        "json_report_path\n"
        "markdown_report_path\n"
        "def build_initial_decision_trace():\n    pass\n"
        "def enrich_decision_trace_with_risk():\n    pass\n"
        "def write_rationale_reports():\n    pass\n"
        "def resolve_rationale_report_paths():\n    pass\n"
    )


def _repository_fixture() -> str:
    return (
        "def ensure_decision_trace():\n    pass\n"
        "def update_decision_trace():\n    pass\n"
        "def load_decision_trace():\n    pass\n"
        "order_request.decision_trace_id\n"
        "event.decision_trace_id\n"
        "entry_decision_trace_id\n"
        "decision_trace_id\n"
    )


def _inference_fixture() -> str:
    return (
        "def _build_prediction_explainability():\n    pass\n"
        "build_signal_explanation\n"
    )


def _runner_fixture() -> str:
    return (
        "ensure_decision_trace\n"
        "update_decision_trace\n"
        "write_rationale_reports\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
