"""Tests for the M15 operator-console audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dashboards.m15_operator_console_audit import audit_m15_operator_console


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M15 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m15_operator_console(repo_root=tmp_path)

    assert result["m15_state"] == "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M16_DEPLOYMENT_ENVIRONMENT_AUDIT"
    assert result["next_required_action"] == "AUDIT_M16_DEPLOYMENT_ENVIRONMENT_CONTROLS"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m15_operator_console(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m15_operator_console_audit.json",
        "m15_operator_console_audit.md",
        "operator_console_surface_audit.csv",
        "data_source_audit.csv",
        "view_model_audit.csv",
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


def test_missing_data_sources_contract_produces_partial_state(tmp_path: Path) -> None:
    """Missing dashboard data-source contract should be a blocking M15 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "dashboards/data_sources.py",
        _data_sources_fixture().replace("class DashboardDataSources", ""),
    )

    result = audit_m15_operator_console(repo_root=tmp_path)

    assert result["m15_state"] == "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "dashboard_data_sources_contract" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M15_OPERATOR_CONSOLE_GAP_FILLS"


def test_missing_system_reliability_source_is_reported(tmp_path: Path) -> None:
    """The operator console should expose canonical reliability summary data."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "dashboards/data_sources.py",
        _data_sources_fixture().replace('"/reliability/system"', '"/missing"'),
    )

    result = audit_m15_operator_console(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "system_reliability_endpoint" in gap_names


def test_missing_operator_banner_is_reported(tmp_path: Path) -> None:
    """The persistent operator banner is required for M15 visibility."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "dashboards/view_models.py",
        _view_models_fixture().replace("def build_operator_banner", ""),
    )

    result = audit_m15_operator_console(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "operator_banner_view_model" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M15 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m15_operator_console(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["operator_console_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "dashboards/__init__.py", "")
    _write(root / "dashboards/data_sources.py", _data_sources_fixture())
    _write(root / "dashboards/view_models.py", _view_models_fixture())
    _write(root / "dashboards/widgets.py", _widgets_fixture())
    _write(root / "dashboards/streamlit_app.py", _streamlit_fixture())
    _write(root / "README.md", "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _data_sources_fixture() -> str:
    return (
        "class ApiHealthSnapshot:\n    pass\n"
        "class DatabaseSnapshot:\n    pass\n"
        "class DecisionTraceSnapshot:\n    pass\n"
        "class SystemReliabilitySnapshot:\n    pass\n"
        "class ActiveAlertSnapshot:\n    pass\n"
        "class AlertTimelineSnapshot:\n    pass\n"
        "class DashboardSnapshot:\n    pass\n"
        "class DashboardDataSources:\n"
        "    async def load_snapshot(self):\n"
        "        pass\n"
        '"/health"\n'
        '"/signal"\n'
        '"/freshness"\n'
        '"/reliability/system"\n'
        '"/alerts/active"\n'
        '"/alerts/timeline"\n'
        "POSITIONS_TABLE\n"
        "LEDGER_TABLE\n"
        "ORDER_EVENTS_TABLE\n"
        "LIVE_SAFETY_TABLE\n"
        "DECISION_TRACES_TABLE\n"
        "RELIABILITY_STATE_TABLE\n"
        "RELIABILITY_EVENTS_TABLE\n"
    )


def _view_models_fixture() -> str:
    return (
        "def build_overview_metrics():\n    pass\n"
        "def build_latest_signal_rows():\n    pass\n"
        "def build_symbol_freshness_rows():\n    pass\n"
        "def build_reliability_status_rows():\n    pass\n"
        "def build_service_health_rows():\n    pass\n"
        "def build_feature_lag_rows():\n    pass\n"
        "def build_live_status_rows():\n    pass\n"
        "def build_live_critical_state_strip():\n    pass\n"
        "def build_recent_decision_trace_rows():\n    pass\n"
        "def build_latest_blocked_trade_rows():\n    pass\n"
        "def build_trade_journal_rows():\n    pass\n"
        "def build_operator_incident_rows():\n    pass\n"
        "def build_operator_banner():\n    pass\n"
        "def build_config_summary_rows():\n    pass\n"
        "def build_model_reference_rows():\n    pass\n"
        "def build_performance_by_regime_rows():\n    pass\n"
    )


def _widgets_fixture() -> str:
    return (
        "def render_operator_banner():\n    pass\n"
        "def render_incidents_panel():\n    pass\n"
        "def render_live_critical_state_strip():\n    pass\n"
    )


def _streamlit_fixture() -> str:
    return (
        "Read-only Stream Alpha M15 operator console\n"
        "render_operator_banner\n"
        "render_live_critical_state_strip\n"
        "render_incidents_panel\n"
        "market_view\n"
        "signals_view\n"
        "trades_view\n"
        "health_view\n"
        "models_view\n"
        "incidents_view\n"
        "_render_rationale_report_downloads\n"
        "_render_continual_learning_operator_guidance\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
