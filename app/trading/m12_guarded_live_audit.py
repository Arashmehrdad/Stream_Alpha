"""Artifact-backed M12 guarded-live control audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m12/guarded_live_audit"
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
    "live_config_contract",
    "paper_probe_config_contract",
    "live_safety_state_contract",
    "startup_checklist_contract",
    "broker_account_contract",
    "broker_submit_result_contract",
    "runtime_live_arming_switch",
    "runtime_confirmation_phrase",
    "startup_validation",
    "startup_assertion",
    "submit_gate_resolution",
    "submit_state_derivation",
    "manual_disable_gate",
    "failure_hard_stop_gate",
    "reconciliation_control",
    "reconciliation_assertion",
    "health_gate_control",
    "health_gate_assertion",
    "startup_checklist_artifact_writer",
    "live_status_artifact_writer",
    "live_safety_persistence",
    "live_safety_loading",
    "runner_live_startup_gates",
    "runner_runtime_live_gates",
    "live_adapter_precheck",
    "live_adapter_submit_gate",
    "m11_execution_interface_documented",
    "m20_pause_documentation",
)


def audit_m12_guarded_live(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M12 guarded-live controls and write deterministic artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _guarded_live_surface_audit(root)
    startup_rows = _startup_gate_audit(root)
    submit_rows = _live_submit_gate_audit(root)
    reconciliation_rows = _reconciliation_health_audit(root)
    artifact_rows = _artifact_status_audit(root)
    gap_rows = _gap_analysis(
        surface_rows,
        startup_rows,
        submit_rows,
        reconciliation_rows,
        artifact_rows,
    )
    state = _m12_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m12_guarded_live_audit_v1",
        "repo_root": str(root),
        "m12_state": state,
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
            "guarded_live_surfaces": surface_rows,
            "startup_gates": startup_rows,
            "submit_gates": submit_rows,
            "reconciliation_health": reconciliation_rows,
            "artifact_status": artifact_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["guarded_live_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["startup_gate_audit_csv"]), startup_rows)
    write_csv(Path(output_files["live_submit_gate_audit_csv"]), submit_rows)
    write_csv(
        Path(output_files["reconciliation_health_audit_csv"]),
        reconciliation_rows,
    )
    write_csv(Path(output_files["artifact_status_audit_csv"]), artifact_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "guarded_live_surfaces": surface_rows,
                "startup_gates": startup_rows,
                "submit_gates": submit_rows,
                "reconciliation_health": reconciliation_rows,
                "artifact_status": artifact_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "guarded_live_surface_audit": surface_rows,
            "startup_gate_audit": startup_rows,
            "live_submit_gate_audit": submit_rows,
            "reconciliation_health_audit": reconciliation_rows,
            "artifact_status_audit": artifact_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _guarded_live_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "live_config_contract", "config.py", "class LiveConfig"),
        _surface_check(root, "paper_probe_config_contract", "config.py", "class PaperProbeConfig"),
        _surface_check(
            root,
            "live_safety_state_contract",
            "schemas.py",
            "class LiveSafetyState",
        ),
        _surface_check(
            root,
            "startup_checklist_contract",
            "schemas.py",
            "class LiveStartupChecklist",
        ),
        _surface_check(root, "broker_account_contract", "schemas.py", "class BrokerAccount"),
        _surface_check(
            root,
            "broker_submit_result_contract",
            "schemas.py",
            "class BrokerSubmitResult",
        ),
        _surface_check(
            root,
            "runtime_live_arming_switch",
            "live.py",
            "STREAMALPHA_ENABLE_LIVE",
        ),
        _surface_check(
            root,
            "runtime_confirmation_phrase",
            "live.py",
            "LIVE_CONFIRMATION_PHRASE",
        ),
        _surface_check(root, "startup_validation", "live.py", "def validate_live_startup"),
        _surface_check(
            root,
            "startup_assertion",
            "live.py",
            "def assert_live_startup_passed",
        ),
        _surface_check(
            root,
            "submit_gate_resolution",
            "live.py",
            "def resolve_live_submit_gate",
        ),
        _surface_check(
            root,
            "submit_state_derivation",
            "live.py",
            "def derive_live_submit_state",
        ),
        _surface_check(
            root,
            "manual_disable_gate",
            "live.py",
            "LIVE_MANUAL_DISABLE_ACTIVE",
        ),
        _surface_check(
            root,
            "failure_hard_stop_gate",
            "live.py",
            "LIVE_FAILURE_HARD_STOP_ACTIVE",
        ),
        _surface_check(root, "reconciliation_control", "live.py", "def reconcile_live_state"),
        _surface_check(
            root,
            "reconciliation_assertion",
            "live.py",
            "def assert_live_reconciliation_clear",
        ),
        _surface_check(root, "health_gate_control", "live.py", "def apply_live_health_gate"),
        _surface_check(
            root,
            "health_gate_assertion",
            "live.py",
            "def assert_live_health_gate_clear",
        ),
        _surface_check(
            root,
            "startup_checklist_artifact_writer",
            "live.py",
            "def write_startup_checklist_artifact",
        ),
        _surface_check(
            root,
            "live_status_artifact_writer",
            "live.py",
            "def write_live_status_artifact",
        ),
        _surface_check(
            root,
            "live_safety_persistence",
            "repository.py",
            "save_live_safety_state",
        ),
        _surface_check(
            root,
            "live_safety_loading",
            "repository.py",
            "load_live_safety_state",
        ),
        _surface_check(
            root,
            "runner_live_startup_gates",
            "runner.py",
            "validate_live_startup",
        ),
        _surface_check(
            root,
            "runner_runtime_live_gates",
            "runner.py",
            "_refresh_live_reconciliation",
        ),
        _surface_check(
            root,
            "live_adapter_precheck",
            "execution.py",
            "def _live_precheck_failure",
        ),
        _surface_check(
            root,
            "live_adapter_submit_gate",
            "execution.py",
            "resolve_live_submit_gate",
        ),
        _docs_surface_check(
            root,
            "m11_execution_interface_documented",
            "M11_EXECUTION_INTERFACE_CONSOLIDATED",
            "Docs preserve M11 as consolidated upstream execution interface.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "Docs preserve M20 as paused and non-authoritative.",
        ),
    ]


def _startup_gate_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "execution_mode_live_check", "live.py", "execution_mode_live"),
        _audit_row(root, "live_config_enabled_check", "live.py", "live_config_enabled"),
        _audit_row(root, "runtime_live_armed_check", "live.py", "runtime_live_armed"),
        _audit_row(root, "runtime_confirmation_check", "live.py", "runtime_confirmation"),
        _audit_row(root, "alpaca_api_key_check", "live.py", "alpaca_api_key_present"),
        _audit_row(root, "alpaca_api_secret_check", "live.py", "alpaca_api_secret_present"),
        _audit_row(root, "alpaca_base_url_check", "live.py", "alpaca_base_url_present"),
        _audit_row(root, "broker_authentication_check", "live.py", "broker_authentication"),
        _audit_row(root, "account_environment_check", "live.py", "account_environment_match"),
        _audit_row(root, "account_id_check", "live.py", "account_id_match"),
        _audit_row(root, "symbol_whitelist_check", "live.py", "symbol_whitelist_non_empty"),
        _audit_row(root, "max_notional_check", "live.py", "max_order_notional_positive"),
        _audit_row(
            root,
            "failure_hard_stop_threshold_check",
            "live.py",
            "failure_hard_stop_threshold_positive",
        ),
        _audit_row(root, "manual_disable_startup_check", "live.py", "manual_disable_inactive"),
    ]


def _live_submit_gate_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "live_disabled_blocks_submit", "live.py", "LIVE_DISABLED"),
        _audit_row(root, "account_mismatch_blocks_submit", "live.py", "LIVE_ACCOUNT_ID_MISMATCH"),
        _audit_row(
            root,
            "environment_mismatch_blocks_submit",
            "live.py",
            "LIVE_ENVIRONMENT_MISMATCH",
        ),
        _audit_row(
            root,
            "startup_failed_blocks_submit",
            "live.py",
            "LIVE_STARTUP_CHECKS_NOT_PASSED",
        ),
        _audit_row(root, "manual_disable_blocks_submit", "live.py", "LIVE_MANUAL_DISABLE_ACTIVE"),
        _audit_row(root, "hard_stop_blocks_submit", "live.py", "LIVE_FAILURE_HARD_STOP_ACTIVE"),
        _audit_row(
            root,
            "reconciliation_blocks_submit",
            "live.py",
            "LIVE_RECONCILIATION_BLOCKED",
        ),
        _audit_row(root, "health_gate_blocks_submit", "live.py", "LIVE_SYSTEM_HEALTH_UNAVAILABLE"),
        _audit_row(root, "stale_signal_blocks_adapter", "execution.py", "LIVE_SIGNAL_STALE"),
        _audit_row(
            root,
            "symbol_whitelist_blocks_adapter",
            "execution.py",
            "LIVE_SYMBOL_NOT_WHITELISTED",
        ),
        _audit_row(
            root,
            "max_notional_blocks_adapter",
            "execution.py",
            "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED",
        ),
        _audit_row(
            root,
            "broker_failure_records_hard_stop",
            "execution.py",
            "record_live_failure",
        ),
        _audit_row(
            root,
            "paper_probe_submit_constraints",
            "execution.py",
            "LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED",
        ),
    ]


def _reconciliation_health_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "broker_unavailable_fails_closed",
            "live.py",
            "LIVE_RECONCILIATION_BROKER_UNAVAILABLE",
        ),
        _audit_row(root, "orphan_order_detection", "live.py", "LIVE_RECONCILIATION_ORPHAN_ORDER"),
        _audit_row(
            root,
            "orphan_position_detection",
            "live.py",
            "LIVE_RECONCILIATION_ORPHAN_POSITION",
        ),
        _audit_row(
            root,
            "order_state_mismatch_detection",
            "live.py",
            "LIVE_RECONCILIATION_ORDER_STATE_MISMATCH",
        ),
        _audit_row(
            root,
            "position_qty_mismatch_detection",
            "live.py",
            "LIVE_RECONCILIATION_POSITION_QTY_MISMATCH",
        ),
        _audit_row(
            root,
            "system_health_unavailable_gate",
            "live.py",
            "LIVE_SYSTEM_HEALTH_UNAVAILABLE",
        ),
        _audit_row(root, "system_health_stale_gate", "live.py", "LIVE_SYSTEM_HEALTH_STALE"),
        _audit_row(root, "signal_stale_health_gate", "live.py", "LIVE_SIGNAL_STALE"),
        _audit_row(
            root,
            "runner_refreshes_reconciliation",
            "runner.py",
            "_refresh_live_reconciliation",
        ),
        _audit_row(root, "runner_refreshes_health_gate", "runner.py", "_refresh_live_health_gate"),
    ]


def _artifact_status_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "startup_checklist_writer", "live.py", "write_startup_checklist_artifact"),
        _audit_row(root, "live_status_writer", "live.py", "write_live_status_artifact"),
        _audit_row(
            root,
            "broker_reconciliation_truth_source",
            "live.py",
            "BROKER_RECONCILIATION_ONLY",
        ),
        _audit_row(root, "cross_venue_context_reported", "live.py", "cross_venue_context"),
        _audit_row(root, "can_submit_live_now_reported", "live.py", "can_submit_live_now"),
        _audit_row(
            root,
            "primary_block_reason_reported",
            "live.py",
            "primary_block_reason_code",
        ),
        _audit_row(root, "manual_disable_path_configured", "config.py", "manual_disable_path"),
        _audit_row(
            root,
            "startup_checklist_path_configured",
            "config.py",
            "startup_checklist_path",
        ),
        _audit_row(root, "live_status_path_configured", "config.py", "live_status_path"),
    ]


def _surface_check(
    root: Path,
    name: str,
    file_name: str,
    needle: str,
) -> dict[str, str]:
    path = Path("app/trading") / file_name
    return {
        "surface_name": name,
        "path": str(path),
        "status": "PRESENT" if _file_contains(root, path, needle) else "MISSING",
        "required_for_m12": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M12 guarded-live surface `{name}` is present.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m12": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _audit_row(root: Path, name: str, file_name: str, needle: str) -> dict[str, str]:
    path = Path("app/trading") / file_name
    return {
        "audit_name": name,
        "path": str(path),
        "status": "PRESENT" if _file_contains(root, path, needle) else "MISSING",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M12 guarded-live audit `{name}` checks `{needle}`.",
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


def _gap_analysis(
    surface_rows: list[Mapping[str, str]],
    startup_rows: list[Mapping[str, str]],
    submit_rows: list[Mapping[str, str]],
    reconciliation_rows: list[Mapping[str, str]],
    artifact_rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in surface_rows:
        if row["required_for_m12"] == "True" and row["status"] != "PRESENT":
            rows.append(_gap_row(row["surface_name"], "RESTORE_M12_GUARDED_LIVE_SURFACE"))
    for row_set, action in (
        (startup_rows, "RESTORE_M12_STARTUP_GATE"),
        (submit_rows, "RESTORE_M12_LIVE_SUBMIT_GATE"),
        (reconciliation_rows, "RESTORE_M12_RECONCILIATION_OR_HEALTH_GATE"),
        (artifact_rows, "RESTORE_M12_OPERATOR_STATUS_ARTIFACT"),
    ):
        for row in row_set:
            if row["status"] != "PRESENT":
                rows.append(_gap_row(row["audit_name"], action))
    return rows


def _gap_row(name: str, action: str) -> dict[str, str]:
    return {
        "gap_name": name,
        "gap_status": "BLOCKING",
        "recommended_action": action,
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": "M12 audit-only finding; no runtime behavior changed.",
    }


def _m12_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M12_GUARDED_LIVE_CONTROLS_CONSOLIDATED"
    if present:
        return "M12_GUARDED_LIVE_CONTROLS_PARTIAL"
    return "M12_GUARDED_LIVE_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M12_GUARDED_LIVE_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M13_RELIABILITY_RECOVERY_AUDIT"
        next_required_action = "AUDIT_M13_RELIABILITY_AND_RECOVERY_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M12_GUARDED_LIVE_GAP_FILLS"
        next_required_action = "FILL_M12_GUARDED_LIVE_GAPS"
    else:
        recommendation = "RESTORE_M12_GUARDED_LIVE_PREREQUISITES"
        next_required_action = "RESTORE_M12_GUARDED_LIVE_PREREQUISITES"
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
        "schema_version": "m12_guarded_live_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "guarded_live_surface_count": len(audit_rows["guarded_live_surfaces"]),
        "startup_gate_count": len(audit_rows["startup_gates"]),
        "submit_gate_count": len(audit_rows["submit_gates"]),
        "reconciliation_health_count": len(audit_rows["reconciliation_health"]),
        "artifact_status_count": len(audit_rows["artifact_status"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m12_guarded_live_audit.json"),
        "report_md": str(output_dir / "m12_guarded_live_audit.md"),
        "guarded_live_surface_audit_csv": str(output_dir / "guarded_live_surface_audit.csv"),
        "startup_gate_audit_csv": str(output_dir / "startup_gate_audit.csv"),
        "live_submit_gate_audit_csv": str(output_dir / "live_submit_gate_audit.csv"),
        "reconciliation_health_audit_csv": str(output_dir / "reconciliation_health_audit.csv"),
        "artifact_status_audit_csv": str(output_dir / "artifact_status_audit.csv"),
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
        _section("Guarded Live Surfaces", "surface_name", audit_rows["guarded_live_surfaces"]),
        _section("Startup Gates", "audit_name", audit_rows["startup_gates"]),
        _section("Live Submit Gates", "audit_name", audit_rows["submit_gates"]),
        _section(
            "Reconciliation And Health",
            "audit_name",
            audit_rows["reconciliation_health"],
        ),
        _section("Operator Artifacts", "audit_name", audit_rows["artifact_status"]),
    ]
    lines = [
        "# M12 Guarded Live Audit",
        "",
        f"- M12 state: `{report['m12_state']}`",
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


def _section(
    title: str,
    name_key: str,
    rows: list[Mapping[str, str]],
) -> str:
    lines = [f"## {title}"]
    for row in rows:
        lines.append(f"- `{row[name_key]}`: `{row['status']}`")
    lines.append("")
    return "\n".join(lines)
