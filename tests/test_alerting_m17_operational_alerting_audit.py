"""Tests for the M17 operational alerting audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.alerting.m17_operational_alerting_audit import audit_m17_operational_alerting


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M17 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m17_operational_alerting(repo_root=tmp_path)

    assert result["m17_state"] == "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M18_EVALUATION_REPORTING_AUDIT"
    assert (
        result["next_required_action"]
        == "AUDIT_M18_EVALUATION_REPORTING_AND_DEGRADATION_CONTROLS"
    )
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m17_operational_alerting(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m17_operational_alerting_audit.json",
        "m17_operational_alerting_audit.md",
        "alerting_surface_audit.csv",
        "alert_rule_audit.csv",
        "persistence_api_audit.csv",
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


def test_missing_alert_service_produces_partial_state(tmp_path: Path) -> None:
    """Missing alert service should be reported as an M17 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/alerting/service.py",
        _service_fixture().replace("class OperationalAlertService", ""),
    )

    result = audit_m17_operational_alerting(repo_root=tmp_path)

    assert result["m17_state"] == "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "alert_service" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M17_OPERATIONAL_ALERTING_GAP_FILLS"


def test_missing_active_alert_endpoint_is_reported(tmp_path: Path) -> None:
    """The active-alert API read surface is required for M17."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/inference/main.py",
        _inference_main_fixture().replace("/alerts/active", "/alerts/missing"),
    )

    result = audit_m17_operational_alerting(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "active_alerts_endpoint" in gap_names


def test_missing_daily_summary_writer_is_reported(tmp_path: Path) -> None:
    """Daily operations summaries are part of the M17 operator contract."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/alerting/service.py",
        _service_fixture().replace("def write_daily_summary", ""),
    )

    result = audit_m17_operational_alerting(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "daily_summary_writer" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M17 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m17_operational_alerting(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["alerting_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "configs/alerting.yaml", "schema_version: m17_alerting_v1\n")
    _write(root / "app/alerting/config.py", "class AlertingConfig:\n    pass\n")
    _write(root / "app/alerting/schemas.py", _schemas_fixture())
    _write(root / "app/alerting/service.py", _service_fixture())
    _write(root / "app/alerting/repository.py", _repository_fixture())
    _write(root / "app/inference/main.py", _inference_main_fixture())
    _write(root / "app/inference/service.py", _inference_service_fixture())
    _write(root / "app/trading/runner.py", _runner_fixture())
    _write(root / "dashboards/data_sources.py", _dashboard_data_sources_fixture())
    _write(root / "dashboards/streamlit_app.py", _streamlit_fixture())
    _write(root / "dashboards/widgets.py", "def render_incidents_panel():\n    pass\n")
    _write(root / "tests/test_alerting_service.py", "Focused M17 alerting backend tests.\n")
    _write(root / "tests/test_alerting_repository.py", "M17\n")
    _write(root / "README.md", "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _schemas_fixture() -> str:
    return (
        "class OperationalAlertEvent:\n    pass\n"
        "class OperationalAlertState:\n    pass\n"
        "class StartupSafetyReport:\n    pass\n"
        "class DailyOperationsSummary:\n    pass\n"
    )


def _service_fixture() -> str:
    return (
        "FEED_STALE\n"
        "CONSUMER_LAG\n"
        "class OperationalAlertService:\n"
        "    def _order_failure_spike_observation(self):\n        pass\n"
        "    def _drawdown_breach_observation(self):\n        pass\n"
        "    def _signal_silence_observation(self):\n        pass\n"
        "    def _signal_flood_observation(self):\n        pass\n"
        "    def record_live_mode_activation(self):\n        pass\n"
        "    def _record_startup_safety_alert(self):\n        pass\n"
        "    def write_daily_summary(self):\n        pass\n"
        "def build_alert_fingerprint():\n    pass\n"
    )


def _repository_fixture() -> str:
    return (
        "class OperationalAlertRepository:\n"
        "    def insert_event(self):\n        pass\n"
        "    def load_state(self):\n        pass\n"
        "    def save_state(self):\n        pass\n"
        "    def load_active_states(self):\n        pass\n"
        "    def load_events_for_day(self):\n        pass\n"
        "    def load_timeline_events(self):\n        pass\n"
        "is_active BOOLEAN\n"
    )


def _inference_main_fixture() -> str:
    return (
        "/alerts/active\n"
        "/alerts/timeline\n"
        "/operations/daily-summary\n"
        "/operations/startup-safety\n"
    )


def _inference_service_fixture() -> str:
    return (
        "async def active_alerts():\n    pass\n"
        "async def alert_timeline():\n    pass\n"
        "async def daily_operations_summary():\n    pass\n"
        "async def startup_safety_report():\n    pass\n"
    )


def _runner_fixture() -> str:
    return (
        "def _evaluate_alerting_cycle():\n    pass\n"
        "write_startup_safety_artifact\n"
        "write_daily_summary\n"
    )


def _dashboard_data_sources_fixture() -> str:
    return (
        "class ActiveAlertsSnapshot:\n    pass\n"
        "class AlertTimelineSnapshot:\n    pass\n"
        "class StartupSafetySnapshot:\n    pass\n"
        "class DailyOperationsSummarySnapshot:\n    pass\n"
        "def _load_active_alerts():\n    pass\n"
        "def _load_alert_timeline():\n    pass\n"
        "def _load_startup_safety():\n    pass\n"
        "def _load_daily_operations_summary():\n    pass\n"
    )


def _streamlit_fixture() -> str:
    return (
        "_build_active_alert_rows\n"
        "_build_incident_timeline_rows\n"
        "_build_startup_safety_rows\n"
        "_build_daily_operations_summary_rows\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
