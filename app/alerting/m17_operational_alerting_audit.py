"""Artifact-backed M17 operational alerting and incident audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m17/operational_alerting_audit"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
CRITICAL_SURFACES = (
    "alerting_config_file",
    "alerting_config_contract",
    "alert_event_contract",
    "alert_state_contract",
    "startup_safety_contract",
    "daily_summary_contract",
    "alert_service",
    "alert_fingerprint_builder",
    "alert_repository",
    "active_alerts_endpoint",
    "alert_timeline_endpoint",
    "daily_summary_endpoint",
    "startup_safety_endpoint",
    "runner_alerting_cycle",
    "runner_startup_safety_writer",
    "runner_daily_summary_writer",
    "dashboard_active_alerts",
    "dashboard_alert_timeline",
    "dashboard_startup_safety",
    "dashboard_daily_summary",
    "m16_deployment_documented",
    "m20_pause_documentation",
)


def audit_m17_operational_alerting(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M17 alerting/incident controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _alerting_surface_audit(root)
    rule_rows = _alert_rule_audit(root)
    persistence_rows = _persistence_api_audit(root)
    operator_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(surface_rows, rule_rows, persistence_rows, operator_rows)
    state = _m17_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m17_operational_alerting_audit_v1",
        "repo_root": str(root),
        "m17_state": state,
        "critical_surface_count": len(CRITICAL_SURFACES),
        "present_critical_surface_count": sum(
            1
            for row in surface_rows
            if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
        ),
        "gap_count": len(gap_rows),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = _manifest(
        root=root,
        output_dir=resolved_output_dir,
        output_files=output_files,
        audit_rows={
            "alerting_surfaces": surface_rows,
            "alert_rules": rule_rows,
            "persistence_api": persistence_rows,
            "operator_visibility": operator_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["alerting_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["alert_rule_audit_csv"]), rule_rows)
    write_csv(Path(output_files["persistence_api_audit_csv"]), persistence_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), operator_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "alerting_surfaces": surface_rows,
                "alert_rules": rule_rows,
                "persistence_api": persistence_rows,
                "operator_visibility": operator_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "alerting_surface_audit": surface_rows,
            "alert_rule_audit": rule_rows,
            "persistence_api_audit": persistence_rows,
            "operator_visibility_audit": operator_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _alerting_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "alerting_config_file", "configs/alerting.yaml", "m17_alerting_v1"),
        _surface_check(
            root,
            "alerting_config_contract",
            "app/alerting/config.py",
            "class AlertingConfig",
        ),
        _surface_check(
            root,
            "alert_event_contract",
            "app/alerting/schemas.py",
            "class OperationalAlertEvent",
        ),
        _surface_check(
            root,
            "alert_state_contract",
            "app/alerting/schemas.py",
            "class OperationalAlertState",
        ),
        _surface_check(
            root,
            "startup_safety_contract",
            "app/alerting/schemas.py",
            "class StartupSafetyReport",
        ),
        _surface_check(
            root,
            "daily_summary_contract",
            "app/alerting/schemas.py",
            "class DailyOperationsSummary",
        ),
        _surface_check(
            root,
            "alert_service",
            "app/alerting/service.py",
            "class OperationalAlertService",
        ),
        _surface_check(
            root,
            "alert_fingerprint_builder",
            "app/alerting/service.py",
            "def build_alert_fingerprint",
        ),
        _surface_check(
            root,
            "alert_repository",
            "app/alerting/repository.py",
            "class OperationalAlertRepository",
        ),
        _surface_check(root, "active_alerts_endpoint", "app/inference/main.py", "/alerts/active"),
        _surface_check(
            root,
            "alert_timeline_endpoint",
            "app/inference/main.py",
            "/alerts/timeline",
        ),
        _surface_check(
            root,
            "daily_summary_endpoint",
            "app/inference/main.py",
            "/operations/daily-summary",
        ),
        _surface_check(
            root,
            "startup_safety_endpoint",
            "app/inference/main.py",
            "/operations/startup-safety",
        ),
        _surface_check(
            root,
            "runner_alerting_cycle",
            "app/trading/runner.py",
            "def _evaluate_alerting_cycle",
        ),
        _surface_check(
            root,
            "runner_startup_safety_writer",
            "app/trading/runner.py",
            "write_startup_safety_artifact",
        ),
        _surface_check(
            root,
            "runner_daily_summary_writer",
            "app/trading/runner.py",
            "write_daily_summary",
        ),
        _surface_check(
            root,
            "dashboard_active_alerts",
            "dashboards/data_sources.py",
            "class ActiveAlertsSnapshot",
        ),
        _surface_check(
            root,
            "dashboard_alert_timeline",
            "dashboards/data_sources.py",
            "class AlertTimelineSnapshot",
        ),
        _surface_check(
            root,
            "dashboard_startup_safety",
            "dashboards/data_sources.py",
            "class StartupSafetySnapshot",
        ),
        _surface_check(
            root,
            "dashboard_daily_summary",
            "dashboards/data_sources.py",
            "class DailyOperationsSummarySnapshot",
        ),
        _docs_surface_check(
            root,
            "m16_deployment_documented",
            "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_CONSOLIDATED",
            "M16 deployment/environment status is documented before M17.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _alert_rule_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "feed_stale_rule", "app/alerting/service.py", "FEED_STALE"),
        _audit_row(root, "consumer_lag_rule", "app/alerting/service.py", "CONSUMER_LAG"),
        _audit_row(
            root,
            "order_failure_spike_rule",
            "app/alerting/service.py",
            "_order_failure_spike_observation",
        ),
        _audit_row(
            root,
            "drawdown_breach_rule",
            "app/alerting/service.py",
            "_drawdown_breach_observation",
        ),
        _audit_row(
            root,
            "signal_silence_rule",
            "app/alerting/service.py",
            "_signal_silence_observation",
        ),
        _audit_row(
            root,
            "signal_flood_rule",
            "app/alerting/service.py",
            "_signal_flood_observation",
        ),
        _audit_row(
            root,
            "live_mode_activation_event",
            "app/alerting/service.py",
            "record_live_mode_activation",
        ),
        _audit_row(
            root,
            "startup_safety_alert",
            "app/alerting/service.py",
            "_record_startup_safety_alert",
        ),
        _audit_row(
            root,
            "daily_summary_writer",
            "app/alerting/service.py",
            "def write_daily_summary",
        ),
    ]


def _persistence_api_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "insert_alert_event", "app/alerting/repository.py", "def insert_event"),
        _audit_row(root, "load_alert_state", "app/alerting/repository.py", "def load_state"),
        _audit_row(root, "save_alert_state", "app/alerting/repository.py", "def save_state"),
        _audit_row(
            root,
            "load_active_states",
            "app/alerting/repository.py",
            "def load_active_states",
        ),
        _audit_row(
            root,
            "load_events_for_day",
            "app/alerting/repository.py",
            "def load_events_for_day",
        ),
        _audit_row(
            root,
            "load_timeline_events",
            "app/alerting/repository.py",
            "def load_timeline_events",
        ),
        _audit_row(root, "alert_state_table", "app/alerting/repository.py", "is_active BOOLEAN"),
        _audit_row(
            root,
            "inference_active_alerts",
            "app/inference/service.py",
            "async def active_alerts",
        ),
        _audit_row(
            root,
            "inference_alert_timeline",
            "app/inference/service.py",
            "async def alert_timeline",
        ),
        _audit_row(
            root,
            "inference_daily_summary",
            "app/inference/service.py",
            "async def daily_operations_summary",
        ),
        _audit_row(
            root,
            "inference_startup_safety",
            "app/inference/service.py",
            "async def startup_safety_report",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "active_alerts_loader",
            "dashboards/data_sources.py",
            "_load_active_alerts",
        ),
        _audit_row(
            root,
            "alert_timeline_loader",
            "dashboards/data_sources.py",
            "_load_alert_timeline",
        ),
        _audit_row(
            root,
            "startup_safety_loader",
            "dashboards/data_sources.py",
            "_load_startup_safety",
        ),
        _audit_row(
            root,
            "daily_summary_loader",
            "dashboards/data_sources.py",
            "_load_daily_operations_summary",
        ),
        _audit_row(
            root,
            "active_alert_rows",
            "dashboards/streamlit_app.py",
            "_build_active_alert_rows",
        ),
        _audit_row(
            root,
            "incident_timeline_rows",
            "dashboards/streamlit_app.py",
            "_build_incident_timeline_rows",
        ),
        _audit_row(
            root,
            "startup_safety_rows",
            "dashboards/streamlit_app.py",
            "_build_startup_safety_rows",
        ),
        _audit_row(
            root,
            "daily_operations_rows",
            "dashboards/streamlit_app.py",
            "_build_daily_operations_summary_rows",
        ),
        _audit_row(root, "incidents_panel", "dashboards/widgets.py", "render_incidents_panel"),
        _audit_row(root, "alerting_service_tests", "tests/test_alerting_service.py", "M17"),
        _audit_row(root, "alerting_repository_tests", "tests/test_alerting_repository.py", "M17"),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m17": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M17 alerting/incident surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m17": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _audit_row(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "audit_name": name,
        "path": str(relative_path),
        "status": status,
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M17 audit `{name}` checks `{needle}`.",
    }


def _file_contains(root: Path, relative_path: Path, needle: str) -> bool:
    path = root / relative_path
    return path.is_file() and needle in path.read_text(encoding="utf-8")


def _docs_contain(root: Path, needle: str) -> bool:
    return any(
        (root / path).is_file()
        and needle in (root / path).read_text(encoding="utf-8")
        for path in (Path("README.md"), Path("docs/training.md"), Path("PLANS.md"))
    )


def _gap_analysis(*row_sets: list[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row_set in row_sets:
        for row in row_set:
            if row["status"] == "PRESENT":
                continue
            name = row.get("surface_name", row.get("audit_name", "unknown_gap"))
            rows.append(
                {
                    "gap_name": name,
                    "gap_status": "BLOCKING",
                    "recommended_action": "RESTORE_M17_OPERATIONAL_ALERTING_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M17 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m17_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_CONSOLIDATED"
    if present:
        return "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_PARTIAL"
    return "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M18_EVALUATION_REPORTING_AUDIT"
        next_required_action = "AUDIT_M18_EVALUATION_REPORTING_AND_DEGRADATION_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M17_OPERATIONAL_ALERTING_GAP_FILLS"
        next_required_action = "FILL_M17_OPERATIONAL_ALERTING_INCIDENT_GAPS"
    else:
        recommendation = "RESTORE_M17_OPERATIONAL_ALERTING_PREREQUISITES"
        next_required_action = "RESTORE_M17_OPERATIONAL_ALERTING_PREREQUISITES"
    return {
        "recommendation": recommendation,
        "next_required_action": next_required_action,
        "next_actions": [
            {
                "action": next_required_action,
                "scope": "platform_maturity",
                "runtime_effect": "NO_RUNTIME_EFFECT",
                "m20_status": "M20_PAUSED",
            }
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _manifest(
    *,
    root: Path,
    output_dir: Path,
    output_files: Mapping[str, str],
    audit_rows: Mapping[str, list[Mapping[str, str]]],
) -> dict[str, Any]:
    source_paths = {
        row["path"]
        for rows in audit_rows.values()
        for row in rows
        if row.get("path") and "|" not in row["path"]
    }
    return {
        "schema_version": "m17_operational_alerting_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "alerting_surface_count": len(audit_rows["alerting_surfaces"]),
        "alert_rule_count": len(audit_rows["alert_rules"]),
        "persistence_api_count": len(audit_rows["persistence_api"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m17_operational_alerting_audit.json"),
        "report_md": str(output_dir / "m17_operational_alerting_audit.md"),
        "alerting_surface_audit_csv": str(output_dir / "alerting_surface_audit.csv"),
        "alert_rule_audit_csv": str(output_dir / "alert_rule_audit.csv"),
        "persistence_api_audit_csv": str(output_dir / "persistence_api_audit.csv"),
        "operator_visibility_audit_csv": str(output_dir / "operator_visibility_audit.csv"),
        "gap_analysis_csv": str(output_dir / "gap_analysis.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    audit_rows: Mapping[str, list[Mapping[str, str]]],
    gap_rows: list[Mapping[str, str]],
) -> str:
    sections = [
        _section("Alerting Surfaces", "surface_name", audit_rows["alerting_surfaces"]),
        _section("Alert Rules", "audit_name", audit_rows["alert_rules"]),
        _section("Persistence And API", "audit_name", audit_rows["persistence_api"]),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M17 Operational Alerting And Incident Audit",
        "",
        f"- M17 state: `{report['m17_state']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        "- M20 status: `M20_PAUSED`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        *sections,
        "## Gaps",
    ]
    if gap_rows:
        for row in gap_rows:
            lines.append(f"- `{row['gap_name']}`: `{row['recommended_action']}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _section(title: str, name_key: str, rows: list[Mapping[str, str]]) -> str:
    lines = [f"## {title}"]
    for row in rows:
        lines.append(f"- `{row[name_key]}`: `{row['status']}`")
    lines.append("")
    return "\n".join(lines)
