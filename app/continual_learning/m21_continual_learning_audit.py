"""Artifact-backed M21 continual-learning guarded-workflow audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m21/continual_learning_audit"
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
    "continual_learning_config_file",
    "continual_learning_config_contract",
    "experiment_record_contract",
    "profile_record_contract",
    "drift_cap_record_contract",
    "promotion_decision_contract",
    "event_record_contract",
    "runtime_context_contract",
    "promotion_request_contract",
    "rollback_request_contract",
    "workflow_response_contract",
    "continual_learning_service",
    "runtime_context_resolver",
    "runtime_drift_cap_writer",
    "summary_reader",
    "experiments_reader",
    "profiles_reader",
    "drift_caps_reader",
    "promotions_reader",
    "events_reader",
    "guarded_promote_profile",
    "guarded_rollback_profile",
    "continual_learning_repository_persistence",
    "continual_learning_inference_endpoints",
    "runner_runtime_drift_cap_hook",
    "decision_trace_continual_learning_context",
    "dashboard_continual_learning_visibility",
    "m20_dynamic_ensemble_documented",
    "m20_pause_documentation",
)


def audit_m21_continual_learning(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M21 continual-learning workflow controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _continual_learning_surface_audit(root)
    workflow_rows = _guarded_workflow_audit(root)
    drift_rows = _drift_cap_audit(root)
    visibility_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(surface_rows, workflow_rows, drift_rows, visibility_rows)
    state = _m21_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m21_continual_learning_audit_v1",
        "repo_root": str(root),
        "m21_state": state,
        "critical_surface_count": len(CRITICAL_SURFACES),
        "present_critical_surface_count": sum(
            1
            for row in surface_rows
            if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
        ),
        "gap_count": len(gap_rows),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "m20_research_decision": "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
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
            "continual_learning_surfaces": surface_rows,
            "guarded_workflows": workflow_rows,
            "drift_caps": drift_rows,
            "operator_visibility": visibility_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["continual_learning_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["guarded_workflow_audit_csv"]), workflow_rows)
    write_csv(Path(output_files["drift_cap_audit_csv"]), drift_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), visibility_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "continual_learning_surfaces": surface_rows,
                "guarded_workflows": workflow_rows,
                "drift_caps": drift_rows,
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
            "continual_learning_surface_audit": surface_rows,
            "guarded_workflow_audit": workflow_rows,
            "drift_cap_audit": drift_rows,
            "operator_visibility_audit": visibility_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _continual_learning_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(
            root,
            "continual_learning_config_file",
            "configs/continual_learning.yaml",
            "CALIBRATION_OVERLAY",
        ),
        _surface_check(
            root,
            "continual_learning_config_contract",
            "app/continual_learning/config.py",
            "class ContinualLearningConfig",
        ),
        _surface_check(
            root,
            "experiment_record_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningExperimentRecord",
        ),
        _surface_check(
            root,
            "profile_record_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningProfileRecord",
        ),
        _surface_check(
            root,
            "drift_cap_record_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningDriftCapRecord",
        ),
        _surface_check(
            root,
            "promotion_decision_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningPromotionDecisionRecord",
        ),
        _surface_check(
            root,
            "event_record_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningEventRecord",
        ),
        _surface_check(
            root,
            "runtime_context_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningContextPayload",
        ),
        _surface_check(
            root,
            "promotion_request_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningPromoteProfileRequest",
        ),
        _surface_check(
            root,
            "rollback_request_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningRollbackRequest",
        ),
        _surface_check(
            root,
            "workflow_response_contract",
            "app/continual_learning/schemas.py",
            "class ContinualLearningWorkflowResponse",
        ),
        _surface_check(
            root,
            "continual_learning_service",
            "app/continual_learning/service.py",
            "class ContinualLearningService",
        ),
        _surface_check(
            root,
            "runtime_context_resolver",
            "app/continual_learning/service.py",
            "async def resolve_runtime_context",
        ),
        _surface_check(
            root,
            "runtime_drift_cap_writer",
            "app/continual_learning/service.py",
            "async def write_runtime_drift_caps",
        ),
        _surface_check(
            root,
            "summary_reader",
            "app/continual_learning/service.py",
            "async def summary",
        ),
        _surface_check(
            root,
            "experiments_reader",
            "app/continual_learning/service.py",
            "async def experiments",
        ),
        _surface_check(
            root,
            "profiles_reader",
            "app/continual_learning/service.py",
            "async def profiles",
        ),
        _surface_check(
            root,
            "drift_caps_reader",
            "app/continual_learning/service.py",
            "async def drift_caps",
        ),
        _surface_check(
            root,
            "promotions_reader",
            "app/continual_learning/service.py",
            "async def promotions",
        ),
        _surface_check(
            root,
            "events_reader",
            "app/continual_learning/service.py",
            "async def events",
        ),
        _surface_check(
            root,
            "guarded_promote_profile",
            "app/continual_learning/service.py",
            "async def promote_profile",
        ),
        _surface_check(
            root,
            "guarded_rollback_profile",
            "app/continual_learning/service.py",
            "async def rollback_profile",
        ),
        _surface_check(
            root,
            "continual_learning_repository_persistence",
            "app/trading/repository.py",
            "save_continual_learning_drift_cap",
        ),
        _surface_check(
            root,
            "continual_learning_inference_endpoints",
            "app/inference/main.py",
            "/continual-learning/summary",
        ),
        _surface_check(
            root,
            "runner_runtime_drift_cap_hook",
            "app/trading/runner.py",
            "write_runtime_drift_caps",
        ),
        _surface_check(
            root,
            "decision_trace_continual_learning_context",
            "app/trading/schemas.py",
            "ContinualLearningContextPayload",
        ),
        _surface_check(
            root,
            "dashboard_continual_learning_visibility",
            "dashboards/data_sources.py",
            "/continual-learning/drift-caps",
        ),
        _docs_surface_check(
            root,
            "m20_dynamic_ensemble_documented",
            "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_CONSOLIDATED",
            "M20 dynamic ensemble boundary is documented before M21.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _guarded_workflow_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "operator_confirmation_required",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED",
        ),
        _audit_row(
            root,
            "live_eligibility_block",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED",
        ),
        _audit_row(
            root,
            "shadow_challenger_live_block",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED",
        ),
        _audit_row(
            root,
            "drift_cap_breach_block",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_DRIFT_CAP_BREACHED",
        ),
        _audit_row(
            root,
            "health_status_block",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS",
        ),
        _audit_row(
            root,
            "freshness_status_block",
            "app/continual_learning/service.py",
            "CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS",
        ),
        _audit_row(
            root,
            "promote_endpoint",
            "app/inference/main.py",
            "/continual-learning/promotions/promote-profile",
        ),
        _audit_row(
            root,
            "rollback_endpoint",
            "app/inference/main.py",
            "/continual-learning/promotions/rollback-active-profile",
        ),
        _audit_row(
            root,
            "shadow_challenger_schema_validator",
            "app/continual_learning/schemas.py",
            "INCREMENTAL_SHADOW_CHALLENGER",
        ),
    ]


def _drift_cap_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "approved_candidate_types",
            "configs/continual_learning.yaml",
            "candidate_types",
        ),
        _audit_row(
            root,
            "runtime_writer_reuses_m19_truth",
            "app/continual_learning/service.py",
            "load_latest_adaptive_drift_state",
        ),
        _audit_row(
            root,
            "runtime_writer_persists_m21_caps",
            "app/continual_learning/service.py",
            "save_continual_learning_drift_cap",
        ),
        _audit_row(
            root,
            "drift_cap_from_adaptive_state",
            "app/continual_learning/service.py",
            "def _drift_cap_from_adaptive_state",
        ),
        _audit_row(
            root,
            "latest_drift_cap_repository_reader",
            "app/trading/repository.py",
            "load_latest_continual_learning_drift_cap",
        ),
        _audit_row(
            root,
            "all_drift_cap_repository_reader",
            "app/trading/repository.py",
            "load_all_continual_learning_drift_caps",
        ),
        _audit_row(
            root,
            "drift_cap_summary_artifact",
            "app/continual_learning/service.py",
            "_write_drift_caps_summary_artifact",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "continual_learning_service_tests",
            "tests/test_continual_learning_service.py",
            "promote_profile",
        ),
        _audit_row(
            root,
            "continual_learning_api_tests",
            "tests/test_inference_api.py",
            "/continual-learning/summary",
        ),
        _audit_row(
            root,
            "continual_learning_dashboard_tests",
            "tests/test_dashboard_data_sources.py",
            "/continual-learning/drift-caps",
        ),
        _audit_row(
            root,
            "runner_m21_drift_cap_tests",
            "tests/trading/test_runner_idempotency.py",
            "test_runner_persists_runtime_m21_drift_caps",
        ),
        _audit_row(
            root,
            "decision_trace_context_tests",
            "tests/trading/test_decision_trace.py",
            "ContinualLearningContextPayload",
        ),
        _audit_row(
            root,
            "readme_endpoint_documentation",
            "README.md",
            "/continual-learning/promotions/promote-profile",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m21": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M21 continual-learning surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m21": "True",
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
        "detail": f"M21 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M21_CONTINUAL_LEARNING_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M21 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m21_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_CONSOLIDATED"
    if present:
        return "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_PARTIAL"
    return "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M21_CONTINUAL_LEARNING_GUARDED_WORKFLOW_CONSOLIDATED" and not gap_rows:
        recommendation = "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_COMPLETE"
        next_required_action = "PLAN_NEXT_PLATFORM_MATURITY_OR_DATA_UPGRADE_ROUTE"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M21_CONTINUAL_LEARNING_GAP_FILLS"
        next_required_action = "FILL_M21_CONTINUAL_LEARNING_WORKFLOW_GAPS"
    else:
        recommendation = "RESTORE_M21_CONTINUAL_LEARNING_PREREQUISITES"
        next_required_action = "RESTORE_M21_CONTINUAL_LEARNING_PREREQUISITES"
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
        "schema_version": "m21_continual_learning_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "continual_learning_surface_count": len(audit_rows["continual_learning_surfaces"]),
        "guarded_workflow_count": len(audit_rows["guarded_workflows"]),
        "drift_cap_count": len(audit_rows["drift_caps"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m21_continual_learning_audit.json"),
        "report_md": str(output_dir / "m21_continual_learning_audit.md"),
        "continual_learning_surface_audit_csv": str(
            output_dir / "continual_learning_surface_audit.csv"
        ),
        "guarded_workflow_audit_csv": str(output_dir / "guarded_workflow_audit.csv"),
        "drift_cap_audit_csv": str(output_dir / "drift_cap_audit.csv"),
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
            "Continual-Learning Surfaces",
            "surface_name",
            audit_rows["continual_learning_surfaces"],
        ),
        _section("Guarded Workflows", "audit_name", audit_rows["guarded_workflows"]),
        _section("Drift Caps", "audit_name", audit_rows["drift_caps"]),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M21 Continual-Learning Guarded-Workflow Audit",
        "",
        f"- M21 state: `{report['m21_state']}`",
        f"- M20 research decision: `{report['m20_research_decision']}`",
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
