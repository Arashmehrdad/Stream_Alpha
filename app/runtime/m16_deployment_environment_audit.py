"""Artifact-backed M16 deployment and environment audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m16/deployment_environment_audit"
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
    "runtime_profile_contract",
    "runtime_profile_resolver",
    "profile_trading_config_resolver",
    "startup_report_contract",
    "startup_validation_builder",
    "startup_validation_writer",
    "compose_profiles",
    "compose_config_check_service",
    "compose_config_check_command",
    "compose_service_healthchecks",
    "start_stack_helper",
    "stop_stack_helper",
    "env_example_runtime_profile",
    "env_example_startup_report",
    "paper_profile_config",
    "shadow_profile_config",
    "live_profile_config",
    "paper_vps_deploy_helper",
    "paper_vps_runtime_overrides",
    "paper_vps_operator_scripts",
    "m15_operator_console_documented",
    "m20_pause_documentation",
)


def audit_m16_deployment_environment(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M16 deployment/environment controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _deployment_surface_audit(root)
    profile_rows = _runtime_profile_audit(root)
    startup_rows = _startup_validation_audit(root)
    remote_rows = _remote_deployment_audit(root)
    gap_rows = _gap_analysis(surface_rows, profile_rows, startup_rows, remote_rows)
    state = _m16_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m16_deployment_environment_audit_v1",
        "repo_root": str(root),
        "m16_state": state,
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
            "deployment_surfaces": surface_rows,
            "runtime_profiles": profile_rows,
            "startup_validation": startup_rows,
            "remote_deployment": remote_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["deployment_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["runtime_profile_audit_csv"]), profile_rows)
    write_csv(Path(output_files["startup_validation_audit_csv"]), startup_rows)
    write_csv(Path(output_files["remote_deployment_audit_csv"]), remote_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "deployment_surfaces": surface_rows,
                "runtime_profiles": profile_rows,
                "startup_validation": startup_rows,
                "remote_deployment": remote_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "deployment_surface_audit": surface_rows,
            "runtime_profile_audit": profile_rows,
            "startup_validation_audit": startup_rows,
            "remote_deployment_audit": remote_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _deployment_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(
            root,
            "runtime_profile_contract",
            "app/runtime/config.py",
            "RUNTIME_PROFILES",
        ),
        _surface_check(
            root,
            "runtime_profile_resolver",
            "app/runtime/config.py",
            "def resolve_runtime_profile",
        ),
        _surface_check(
            root,
            "profile_trading_config_resolver",
            "app/runtime/config.py",
            "def profile_trading_config_path",
        ),
        _surface_check(
            root,
            "startup_report_contract",
            "app/runtime/validate.py",
            "STARTUP_REPORT_SCHEMA_VERSION",
        ),
        _surface_check(
            root,
            "startup_validation_builder",
            "app/runtime/validate.py",
            "def build_startup_validation_report",
        ),
        _surface_check(
            root,
            "startup_validation_writer",
            "app/runtime/validate.py",
            "def write_startup_validation_report",
        ),
        _surface_check(
            root,
            "compose_profiles",
            "docker-compose.yml",
            'profiles: ["dev", "paper", "shadow", "live"]',
        ),
        _surface_check(
            root,
            "compose_config_check_service",
            "docker-compose.yml",
            "config-check:",
        ),
        _surface_check(
            root,
            "compose_config_check_command",
            "docker-compose.yml",
            "app.runtime.validate",
        ),
        _surface_check(
            root,
            "compose_service_healthchecks",
            "docker-compose.yml",
            "healthcheck:",
        ),
        _surface_check(root, "start_stack_helper", "scripts/start-stack.ps1", "param("),
        _surface_check(
            root,
            "stop_stack_helper",
            "scripts/stop-stack.ps1",
            '@("compose", "down")',
        ),
        _surface_check(
            root,
            "env_example_runtime_profile",
            ".env.example",
            "STREAMALPHA_RUNTIME_PROFILE=",
        ),
        _surface_check(
            root,
            "env_example_startup_report",
            ".env.example",
            "STREAMALPHA_STARTUP_REPORT_PATH=",
        ),
        _surface_check(
            root,
            "paper_profile_config",
            "configs/paper_trading.paper.yaml",
            "mode: paper",
        ),
        _surface_check(
            root,
            "shadow_profile_config",
            "configs/paper_trading.shadow.yaml",
            "mode: shadow",
        ),
        _surface_check(
            root,
            "live_profile_config",
            "configs/paper_trading.live.yaml",
            "mode: live",
        ),
        _surface_check(
            root,
            "paper_vps_deploy_helper",
            "app/deployment/paper_vps.py",
            "def build_deploy_plan",
        ),
        _surface_check(
            root,
            "paper_vps_runtime_overrides",
            "app/deployment/paper_vps.py",
            "REMOTE_RUNTIME_ENV_OVERRIDES",
        ),
        _multi_file_surface_check(
            root,
            "paper_vps_operator_scripts",
            (
                ("scripts/deploy_paper_vps.ps1", "app.deployment.paper_vps"),
                ("scripts/status_paper_vps.ps1", "app.deployment.paper_vps"),
                ("scripts/stop_paper_vps.ps1", "app.deployment.paper_vps"),
            ),
        ),
        _docs_surface_check(
            root,
            "m15_operator_console_documented",
            "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_CONSOLIDATED",
            "M15 operator-console status is documented before M16.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _runtime_profile_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "supported_profile_dev", "app/runtime/config.py", '"dev"'),
        _audit_row(root, "supported_profile_paper", "app/runtime/config.py", '"paper"'),
        _audit_row(root, "supported_profile_shadow", "app/runtime/config.py", '"shadow"'),
        _audit_row(root, "supported_profile_live", "app/runtime/config.py", '"live"'),
        _audit_row(
            root,
            "paper_profile_config_mapping",
            "app/runtime/config.py",
            "paper_trading.paper.yaml",
        ),
        _audit_row(
            root,
            "shadow_profile_config_mapping",
            "app/runtime/config.py",
            "paper_trading.shadow.yaml",
        ),
        _audit_row(
            root,
            "live_profile_config_mapping",
            "app/runtime/config.py",
            "paper_trading.live.yaml",
        ),
        _audit_row(
            root,
            "runtime_metadata_builder",
            "app/runtime/config.py",
            "def build_runtime_metadata",
        ),
        _audit_row(
            root,
            "inference_runtime_metadata",
            "app/inference/service.py",
            "build_runtime_metadata",
        ),
        _audit_row(
            root,
            "runner_runtime_metadata",
            "app/trading/runner.py",
            "build_runtime_metadata",
        ),
    ]


def _startup_validation_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "runtime_profile_check",
            "app/runtime/validate.py",
            '"runtime_profile"',
        ),
        _audit_row(
            root,
            "settings_parse_check",
            "app/runtime/validate.py",
            '"settings_env_parse"',
        ),
        _audit_row(
            root,
            "trading_config_check",
            "app/runtime/validate.py",
            '"trading_config_path"',
        ),
        _audit_row(
            root,
            "execution_mode_match_check",
            "app/runtime/validate.py",
            '"profile_matches_execution_mode"',
        ),
        _audit_row(root, "model_artifact_check", "app/runtime/validate.py", '"model_artifact"'),
        _audit_row(root, "regime_artifact_check", "app/runtime/validate.py", '"regime_artifact"'),
        _audit_row(root, "live_secret_check", "app/runtime/validate.py", "APCA_API_KEY_ID"),
        _audit_row(
            root,
            "live_arming_check",
            "app/runtime/validate.py",
            "STREAMALPHA_ENABLE_LIVE",
        ),
        _audit_row(
            root,
            "live_confirmation_check",
            "app/runtime/validate.py",
            "STREAMALPHA_LIVE_CONFIRM",
        ),
        _audit_row(
            root,
            "startup_report_path",
            "app/runtime/config.py",
            "default_startup_report_path",
        ),
        _audit_row(
            root,
            "runtime_validate_tests",
            "tests/test_runtime_validate.py",
            "test_startup_validation_passes_for_paper_with_artifacts",
        ),
    ]


def _remote_deployment_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "bounded_upload_roots",
            "app/deployment/paper_vps.py",
            "DEFAULT_UPLOAD_ROOTS",
        ),
        _audit_row(
            root,
            "sanitized_remote_env",
            "app/deployment/paper_vps.py",
            "build_remote_env_text",
        ),
        _audit_row(
            root,
            "vps_connection_aliases",
            "app/deployment/paper_vps.py",
            "VPS_HOST_ALIASES",
        ),
        _audit_row(
            root,
            "ssh_error_wrapping",
            "app/deployment/paper_vps.py",
            "_wrap_ssh_connect_error",
        ),
        _audit_row(
            root,
            "deploy_plan",
            "app/deployment/paper_vps.py",
            "def build_deploy_plan",
        ),
        _audit_row(
            root,
            "deploy_script",
            "scripts/deploy_paper_vps.ps1",
            "app.deployment.paper_vps",
        ),
        _audit_row(
            root,
            "status_script",
            "scripts/status_paper_vps.ps1",
            "app.deployment.paper_vps",
        ),
        _audit_row(
            root,
            "stop_script",
            "scripts/stop_paper_vps.ps1",
            "app.deployment.paper_vps",
        ),
        _audit_row(
            root,
            "remote_deployment_tests",
            "tests/test_deployment_paper_vps.py",
            "test_build_deploy_plan_discovers_bounded_upload_set",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m16": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M16 deployment/environment surface `{name}` is {status.lower()}.",
    }


def _multi_file_surface_check(
    root: Path,
    name: str,
    checks: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    missing = [
        path for path, needle in checks if not _file_contains(root, Path(path), needle)
    ]
    status = "PRESENT" if not missing else "MISSING"
    return {
        "surface_name": name,
        "path": "|".join(path for path, _needle in checks),
        "status": status,
        "required_for_m16": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": (
            "All paper VPS operator scripts are present."
            if not missing
            else ",".join(missing)
        ),
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m16": "True",
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
        "detail": f"M16 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M16_DEPLOYMENT_ENVIRONMENT_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M16 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m16_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_CONSOLIDATED"
    if present:
        return "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_PARTIAL"
    return "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M17_OPERATIONAL_ALERTING_AUDIT"
        next_required_action = "AUDIT_M17_OPERATIONAL_ALERTING_AND_INCIDENT_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M16_DEPLOYMENT_ENVIRONMENT_GAP_FILLS"
        next_required_action = "FILL_M16_DEPLOYMENT_ENVIRONMENT_GAPS"
    else:
        recommendation = "RESTORE_M16_DEPLOYMENT_ENVIRONMENT_PREREQUISITES"
        next_required_action = "RESTORE_M16_DEPLOYMENT_ENVIRONMENT_PREREQUISITES"
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
        "schema_version": "m16_deployment_environment_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "deployment_surface_count": len(audit_rows["deployment_surfaces"]),
        "runtime_profile_count": len(audit_rows["runtime_profiles"]),
        "startup_validation_count": len(audit_rows["startup_validation"]),
        "remote_deployment_count": len(audit_rows["remote_deployment"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m16_deployment_environment_audit.json"),
        "report_md": str(output_dir / "m16_deployment_environment_audit.md"),
        "deployment_surface_audit_csv": str(output_dir / "deployment_surface_audit.csv"),
        "runtime_profile_audit_csv": str(output_dir / "runtime_profile_audit.csv"),
        "startup_validation_audit_csv": str(output_dir / "startup_validation_audit.csv"),
        "remote_deployment_audit_csv": str(output_dir / "remote_deployment_audit.csv"),
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
        _section("Deployment Surfaces", "surface_name", audit_rows["deployment_surfaces"]),
        _section("Runtime Profiles", "audit_name", audit_rows["runtime_profiles"]),
        _section("Startup Validation", "audit_name", audit_rows["startup_validation"]),
        _section("Remote Deployment", "audit_name", audit_rows["remote_deployment"]),
    ]
    lines = [
        "# M16 Deployment And Environment Audit",
        "",
        f"- M16 state: `{report['m16_state']}`",
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
