"""Artifact-backed M15 operator-console visibility audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m15/operator_console_audit"
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
    "dashboard_package",
    "streamlit_operator_console",
    "dashboard_data_sources_contract",
    "dashboard_snapshot_contract",
    "api_health_snapshot_contract",
    "database_snapshot_contract",
    "decision_trace_snapshot_contract",
    "snapshot_loader",
    "operator_banner_view_model",
    "operator_incident_view_model",
    "operator_banner_widget",
    "operator_incident_widget",
    "live_critical_state_strip",
    "service_health_view",
    "feature_lag_view",
    "live_safety_view",
    "decision_trace_view",
    "blocked_trade_view",
    "trade_journal_view",
    "model_reference_view",
    "config_summary_view",
    "m14_explainability_documented",
    "m20_pause_documentation",
)


def audit_m15_operator_console(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M15 operator-console visibility controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _operator_console_surface_audit(root)
    data_source_rows = _data_source_audit(root)
    view_model_rows = _view_model_audit(root)
    visibility_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(
        surface_rows,
        data_source_rows,
        view_model_rows,
        visibility_rows,
    )
    state = _m15_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m15_operator_console_audit_v1",
        "repo_root": str(root),
        "m15_state": state,
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
            "operator_console_surfaces": surface_rows,
            "data_sources": data_source_rows,
            "view_models": view_model_rows,
            "operator_visibility": visibility_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["operator_console_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["data_source_audit_csv"]), data_source_rows)
    write_csv(Path(output_files["view_model_audit_csv"]), view_model_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), visibility_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "operator_console_surfaces": surface_rows,
                "data_sources": data_source_rows,
                "view_models": view_model_rows,
                "operator_visibility": visibility_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "operator_console_surface_audit": surface_rows,
            "data_source_audit": data_source_rows,
            "view_model_audit": view_model_rows,
            "operator_visibility_audit": visibility_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _operator_console_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "dashboard_package", "dashboards/__init__.py", ""),
        _surface_check(
            root,
            "streamlit_operator_console",
            "dashboards/streamlit_app.py",
            "Read-only Stream Alpha M15 operator console",
        ),
        _surface_check(
            root,
            "dashboard_data_sources_contract",
            "dashboards/data_sources.py",
            "class DashboardDataSources",
        ),
        _surface_check(
            root,
            "dashboard_snapshot_contract",
            "dashboards/data_sources.py",
            "class DashboardSnapshot",
        ),
        _surface_check(
            root,
            "api_health_snapshot_contract",
            "dashboards/data_sources.py",
            "class ApiHealthSnapshot",
        ),
        _surface_check(
            root,
            "database_snapshot_contract",
            "dashboards/data_sources.py",
            "class DatabaseSnapshot",
        ),
        _surface_check(
            root,
            "decision_trace_snapshot_contract",
            "dashboards/data_sources.py",
            "class DecisionTraceSnapshot",
        ),
        _surface_check(root, "snapshot_loader", "dashboards/data_sources.py", "load_snapshot"),
        _surface_check(
            root,
            "operator_banner_view_model",
            "dashboards/view_models.py",
            "def build_operator_banner",
        ),
        _surface_check(
            root,
            "operator_incident_view_model",
            "dashboards/view_models.py",
            "def build_operator_incident_rows",
        ),
        _surface_check(
            root,
            "operator_banner_widget",
            "dashboards/widgets.py",
            "def render_operator_banner",
        ),
        _surface_check(
            root,
            "operator_incident_widget",
            "dashboards/widgets.py",
            "def render_incidents_panel",
        ),
        _surface_check(
            root,
            "live_critical_state_strip",
            "dashboards/view_models.py",
            "def build_live_critical_state_strip",
        ),
        _surface_check(
            root,
            "service_health_view",
            "dashboards/view_models.py",
            "def build_service_health_rows",
        ),
        _surface_check(
            root,
            "feature_lag_view",
            "dashboards/view_models.py",
            "def build_feature_lag_rows",
        ),
        _surface_check(
            root,
            "live_safety_view",
            "dashboards/view_models.py",
            "def build_live_status_rows",
        ),
        _surface_check(
            root,
            "decision_trace_view",
            "dashboards/view_models.py",
            "def build_recent_decision_trace_rows",
        ),
        _surface_check(
            root,
            "blocked_trade_view",
            "dashboards/view_models.py",
            "def build_latest_blocked_trade_rows",
        ),
        _surface_check(
            root,
            "trade_journal_view",
            "dashboards/view_models.py",
            "def build_trade_journal_rows",
        ),
        _surface_check(
            root,
            "model_reference_view",
            "dashboards/view_models.py",
            "def build_model_reference_rows",
        ),
        _surface_check(
            root,
            "config_summary_view",
            "dashboards/view_models.py",
            "def build_config_summary_rows",
        ),
        _docs_surface_check(
            root,
            "m14_explainability_documented",
            "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_CONSOLIDATED",
            "M14 explainability/trace status is documented before M15.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _data_source_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "api_health_endpoint", "dashboards/data_sources.py", '"/health"'),
        _audit_row(root, "signal_endpoint", "dashboards/data_sources.py", '"/signal"'),
        _audit_row(root, "freshness_endpoint", "dashboards/data_sources.py", '"/freshness"'),
        _audit_row(
            root,
            "system_reliability_endpoint",
            "dashboards/data_sources.py",
            '"/reliability/system"',
        ),
        _audit_row(
            root,
            "active_alerts_endpoint",
            "dashboards/data_sources.py",
            '"/alerts/active"',
        ),
        _audit_row(
            root,
            "alert_timeline_endpoint",
            "dashboards/data_sources.py",
            '"/alerts/timeline"',
        ),
        _audit_row(root, "positions_table_source", "dashboards/data_sources.py", "POSITIONS_TABLE"),
        _audit_row(root, "ledger_table_source", "dashboards/data_sources.py", "LEDGER_TABLE"),
        _audit_row(root, "order_events_source", "dashboards/data_sources.py", "ORDER_EVENTS_TABLE"),
        _audit_row(root, "live_safety_source", "dashboards/data_sources.py", "LIVE_SAFETY_TABLE"),
        _audit_row(
            root,
            "decision_trace_source",
            "dashboards/data_sources.py",
            "DECISION_TRACES_TABLE",
        ),
        _audit_row(
            root,
            "reliability_state_source",
            "dashboards/data_sources.py",
            "RELIABILITY_STATE_TABLE",
        ),
        _audit_row(
            root,
            "reliability_events_source",
            "dashboards/data_sources.py",
            "RELIABILITY_EVENTS_TABLE",
        ),
    ]


def _view_model_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "overview_metrics",
            "dashboards/view_models.py",
            "def build_overview_metrics",
        ),
        _audit_row(
            root,
            "latest_signal_rows",
            "dashboards/view_models.py",
            "def build_latest_signal_rows",
        ),
        _audit_row(
            root,
            "symbol_freshness_rows",
            "dashboards/view_models.py",
            "def build_symbol_freshness_rows",
        ),
        _audit_row(
            root,
            "reliability_status_rows",
            "dashboards/view_models.py",
            "def build_reliability_status_rows",
        ),
        _audit_row(
            root,
            "service_health_rows",
            "dashboards/view_models.py",
            "def build_service_health_rows",
        ),
        _audit_row(
            root,
            "feature_lag_rows",
            "dashboards/view_models.py",
            "def build_feature_lag_rows",
        ),
        _audit_row(
            root,
            "live_status_rows",
            "dashboards/view_models.py",
            "def build_live_status_rows",
        ),
        _audit_row(
            root,
            "live_critical_strip",
            "dashboards/view_models.py",
            "def build_live_critical_state_strip",
        ),
        _audit_row(
            root,
            "decision_trace_rows",
            "dashboards/view_models.py",
            "def build_recent_decision_trace_rows",
        ),
        _audit_row(
            root,
            "blocked_trade_rows",
            "dashboards/view_models.py",
            "def build_latest_blocked_trade_rows",
        ),
        _audit_row(
            root,
            "trade_journal_rows",
            "dashboards/view_models.py",
            "def build_trade_journal_rows",
        ),
        _audit_row(
            root,
            "operator_incident_rows",
            "dashboards/view_models.py",
            "def build_operator_incident_rows",
        ),
        _audit_row(
            root,
            "operator_banner",
            "dashboards/view_models.py",
            "def build_operator_banner",
        ),
        _audit_row(
            root,
            "config_summary_rows",
            "dashboards/view_models.py",
            "def build_config_summary_rows",
        ),
        _audit_row(
            root,
            "model_reference_rows",
            "dashboards/view_models.py",
            "def build_model_reference_rows",
        ),
        _audit_row(
            root,
            "performance_by_regime",
            "dashboards/view_models.py",
            "def build_performance_by_regime_rows",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "operator_banner_rendered",
            "dashboards/streamlit_app.py",
            "render_operator_banner",
        ),
        _audit_row(
            root,
            "live_critical_strip_rendered",
            "dashboards/streamlit_app.py",
            "render_live_critical_state_strip",
        ),
        _audit_row(
            root,
            "incident_panel_rendered",
            "dashboards/streamlit_app.py",
            "render_incidents_panel",
        ),
        _audit_row(root, "market_view", "dashboards/streamlit_app.py", "market_view"),
        _audit_row(root, "signals_view", "dashboards/streamlit_app.py", "signals_view"),
        _audit_row(root, "trades_view", "dashboards/streamlit_app.py", "trades_view"),
        _audit_row(root, "health_view", "dashboards/streamlit_app.py", "health_view"),
        _audit_row(root, "models_view", "dashboards/streamlit_app.py", "models_view"),
        _audit_row(root, "incidents_view", "dashboards/streamlit_app.py", "incidents_view"),
        _audit_row(
            root,
            "rationale_downloads",
            "dashboards/streamlit_app.py",
            "_render_rationale_report_downloads",
        ),
        _audit_row(
            root,
            "continual_learning_guidance",
            "dashboards/streamlit_app.py",
            "_render_continual_learning_operator_guidance",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m15": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M15 operator-console surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m15": "True",
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
        "detail": f"M15 audit `{name}` checks `{needle}`.",
    }


def _file_contains(root: Path, relative_path: Path, needle: str) -> bool:
    path = root / relative_path
    if not path.is_file():
        return False
    if needle == "":
        return True
    return needle in path.read_text(encoding="utf-8")


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
                    "recommended_action": "RESTORE_M15_OPERATOR_CONSOLE_VISIBILITY_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M15 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m15_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_CONSOLIDATED"
    if present:
        return "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_PARTIAL"
    return "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M16_DEPLOYMENT_ENVIRONMENT_AUDIT"
        next_required_action = "AUDIT_M16_DEPLOYMENT_ENVIRONMENT_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M15_OPERATOR_CONSOLE_GAP_FILLS"
        next_required_action = "FILL_M15_OPERATOR_CONSOLE_VISIBILITY_GAPS"
    else:
        recommendation = "RESTORE_M15_OPERATOR_CONSOLE_PREREQUISITES"
        next_required_action = "RESTORE_M15_OPERATOR_CONSOLE_PREREQUISITES"
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
        "schema_version": "m15_operator_console_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "operator_console_surface_count": len(audit_rows["operator_console_surfaces"]),
        "data_source_count": len(audit_rows["data_sources"]),
        "view_model_count": len(audit_rows["view_models"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m15_operator_console_audit.json"),
        "report_md": str(output_dir / "m15_operator_console_audit.md"),
        "operator_console_surface_audit_csv": str(
            output_dir / "operator_console_surface_audit.csv"
        ),
        "data_source_audit_csv": str(output_dir / "data_source_audit.csv"),
        "view_model_audit_csv": str(output_dir / "view_model_audit.csv"),
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
        _section(
            "Operator Console Surfaces",
            "surface_name",
            audit_rows["operator_console_surfaces"],
        ),
        _section("Data Sources", "audit_name", audit_rows["data_sources"]),
        _section("View Models", "audit_name", audit_rows["view_models"]),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M15 Operator Console Audit",
        "",
        f"- M15 state: `{report['m15_state']}`",
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
