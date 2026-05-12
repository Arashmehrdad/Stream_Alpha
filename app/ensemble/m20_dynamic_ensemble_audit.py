"""Artifact-backed M20 dynamic ensemble and research-boundary audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m20/dynamic_ensemble_audit"
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
    "ensemble_config_file",
    "ensemble_config_contract",
    "ensemble_profile_contract",
    "ensemble_result_contract",
    "ensemble_context_payload",
    "ensemble_service",
    "ensemble_fallback_result",
    "ensemble_resolver",
    "agreement_policy",
    "draft_profile_builder",
    "ensemble_activation_helper",
    "ensemble_rollback_helper",
    "ensemble_research_inventory_truth",
    "ensemble_repository_persistence",
    "inference_runtime_ensemble_state",
    "health_ensemble_identity",
    "signal_ensemble_fields",
    "decision_trace_ensemble_context",
    "dashboard_ensemble_visibility",
    "m19_adaptation_documented",
    "m20_final_negative_summary",
    "m20_pause_documentation",
)


def audit_m20_dynamic_ensemble(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M20 ensemble controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _ensemble_surface_audit(root)
    research_rows = _research_boundary_audit(root)
    promotion_rows = _promotion_boundary_audit(root)
    operator_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(surface_rows, research_rows, promotion_rows, operator_rows)
    state = _m20_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m20_dynamic_ensemble_audit_v1",
        "repo_root": str(root),
        "m20_state": state,
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
            "ensemble_surfaces": surface_rows,
            "research_boundaries": research_rows,
            "promotion_boundaries": promotion_rows,
            "operator_visibility": operator_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["ensemble_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["research_boundary_audit_csv"]), research_rows)
    write_csv(Path(output_files["promotion_boundary_audit_csv"]), promotion_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), operator_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "ensemble_surfaces": surface_rows,
                "research_boundaries": research_rows,
                "promotion_boundaries": promotion_rows,
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
            "ensemble_surface_audit": surface_rows,
            "research_boundary_audit": research_rows,
            "promotion_boundary_audit": promotion_rows,
            "operator_visibility_audit": operator_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _ensemble_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "ensemble_config_file", "configs/ensemble.yaml", "candidate_roles"),
        _surface_check(
            root,
            "ensemble_config_contract",
            "app/ensemble/config.py",
            "class EnsembleConfig",
        ),
        _surface_check(
            root,
            "ensemble_profile_contract",
            "app/ensemble/schemas.py",
            "class EnsembleProfileRecord",
        ),
        _surface_check(
            root,
            "ensemble_result_contract",
            "app/ensemble/schemas.py",
            "class EnsembleResult",
        ),
        _surface_check(
            root,
            "ensemble_context_payload",
            "app/ensemble/schemas.py",
            "class EnsembleContextPayload",
        ),
        _surface_check(
            root,
            "ensemble_service",
            "app/ensemble/service.py",
            "class EnsembleService",
        ),
        _surface_check(
            root,
            "ensemble_fallback_result",
            "app/ensemble/service.py",
            "def build_ensemble_fallback_result",
        ),
        _surface_check(
            root,
            "ensemble_resolver",
            "app/ensemble/service.py",
            "async def resolve_ensemble",
        ),
        _surface_check(
            root,
            "agreement_policy",
            "app/ensemble/service.py",
            "agreement_multiplier",
        ),
        _surface_check(
            root,
            "draft_profile_builder",
            "app/ensemble/promote.py",
            "def build_draft_profile",
        ),
        _surface_check(
            root,
            "ensemble_activation_helper",
            "app/ensemble/promote.py",
            "async def activate_draft_profile",
        ),
        _surface_check(
            root,
            "ensemble_rollback_helper",
            "app/ensemble/rollback.py",
            "async def rollback_active_profile",
        ),
        _surface_check(
            root,
            "ensemble_research_inventory_truth",
            "app/ensemble/research.py",
            "def build_m20_candidate_inventory_truth",
        ),
        _surface_check(
            root,
            "ensemble_repository_persistence",
            "app/trading/repository.py",
            "save_ensemble_profile",
        ),
        _surface_check(
            root,
            "inference_runtime_ensemble_state",
            "app/inference/service.py",
            "_resolve_runtime_ensemble_state",
        ),
        _surface_check(
            root,
            "health_ensemble_identity",
            "app/inference/service.py",
            "dynamic_ensemble",
        ),
        _surface_check(
            root,
            "signal_ensemble_fields",
            "app/inference/service.py",
            "ensemble_roster_status",
        ),
        _surface_check(
            root,
            "decision_trace_ensemble_context",
            "app/trading/schemas.py",
            "EnsembleContextPayload",
        ),
        _surface_check(
            root,
            "dashboard_ensemble_visibility",
            "dashboards/data_sources.py",
            "ensemble_roster_status",
        ),
        _docs_surface_check(
            root,
            "m19_adaptation_documented",
            "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_CONSOLIDATED",
            "M19 bounded adaptation status is documented before M20.",
        ),
        _docs_surface_check(
            root,
            "m20_final_negative_summary",
            "KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY",
            "M20 final negative-result summary is documented.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _research_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "final_research_summary_writer",
            "app/training/m20_final_research_summary.py",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        ),
        _audit_row(
            root,
            "input_redesign_decision",
            "app/training/m20_input_redesign_decision.py",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        ),
        _audit_row(
            root,
            "candidate_comparator",
            "app/training/m20_research_candidate_comparator.py",
            "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE",
        ),
        _audit_row(
            root,
            "specialist_policy_adjudication",
            "app/training/m20_cost_aware_policy_adjudication.py",
            "SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE",
        ),
        _audit_row(
            root,
            "research_feature_enrichment_is_research_only",
            "app/training/m20_research_feature_enrichment.py",
            "NO_RUNTIME_EFFECT",
        ),
        _audit_row(
            root,
            "strategy_candidate_v2_factory_non_promotable",
            "app/training/m20_strategy_candidate_v2_factory.py",
            "NOT_PROMOTABLE",
        ),
    ]


def _promotion_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "ensemble_promotion_decision_contract",
            "app/ensemble/schemas.py",
            "class EnsemblePromotionDecisionRecord",
        ),
        _audit_row(
            root,
            "ensemble_challenger_run_contract",
            "app/ensemble/schemas.py",
            "class EnsembleChallengerRunRecord",
        ),
        _audit_row(
            root,
            "ensemble_promotion_persistence",
            "app/trading/repository.py",
            "save_ensemble_promotion_decision",
        ),
        _audit_row(
            root,
            "ensemble_challenger_persistence",
            "app/trading/repository.py",
            "save_ensemble_challenger_run",
        ),
        _audit_row(
            root,
            "explicit_rollback_target",
            "app/ensemble/schemas.py",
            "rollback_target_profile_id",
        ),
        _audit_row(
            root,
            "runtime_roster_weak_truth",
            "app/inference/service.py",
            "ACTIVE_WEAK",
        ),
        _audit_row(
            root,
            "no_specialist_overclaim_docs",
            "README.md",
            "candidate ecosystem remains narrow",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "ensemble_packet_tests", "tests/test_ensemble_packet2.py", "M20"),
        _audit_row(
            root,
            "decision_trace_ensemble_tests",
            "tests/trading/test_decision_trace.py",
            "dynamic_ensemble",
        ),
        _audit_row(
            root,
            "dashboard_ensemble_tests",
            "tests/test_dashboard_data_sources.py",
            "ensemble_roster_status",
        ),
        _audit_row(
            root,
            "final_research_summary_tests",
            "tests/test_training_m20_final_research_summary.py",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        ),
        _audit_row(
            root,
            "inference_ensemble_tests",
            "tests/test_inference_api.py",
            "dynamic_ensemble",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m20": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M20 dynamic-ensemble surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m20": "True",
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
        "detail": f"M20 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M20_DYNAMIC_ENSEMBLE_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M20 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m20_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_CONSOLIDATED"
    if present:
        return "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_PARTIAL"
    return "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARIES_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M21_CONTINUAL_LEARNING_AUDIT"
        next_required_action = "AUDIT_M21_CONTINUAL_LEARNING_AND_GUARDED_WORKFLOW"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M20_DYNAMIC_ENSEMBLE_GAP_FILLS"
        next_required_action = "FILL_M20_DYNAMIC_ENSEMBLE_RESEARCH_BOUNDARY_GAPS"
    else:
        recommendation = "RESTORE_M20_DYNAMIC_ENSEMBLE_PREREQUISITES"
        next_required_action = "RESTORE_M20_DYNAMIC_ENSEMBLE_PREREQUISITES"
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
        "schema_version": "m20_dynamic_ensemble_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "ensemble_surface_count": len(audit_rows["ensemble_surfaces"]),
        "research_boundary_count": len(audit_rows["research_boundaries"]),
        "promotion_boundary_count": len(audit_rows["promotion_boundaries"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m20_dynamic_ensemble_audit.json"),
        "report_md": str(output_dir / "m20_dynamic_ensemble_audit.md"),
        "ensemble_surface_audit_csv": str(output_dir / "ensemble_surface_audit.csv"),
        "research_boundary_audit_csv": str(output_dir / "research_boundary_audit.csv"),
        "promotion_boundary_audit_csv": str(output_dir / "promotion_boundary_audit.csv"),
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
        _section("Ensemble Surfaces", "surface_name", audit_rows["ensemble_surfaces"]),
        _section("Research Boundaries", "audit_name", audit_rows["research_boundaries"]),
        _section("Promotion Boundaries", "audit_name", audit_rows["promotion_boundaries"]),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M20 Dynamic Ensemble And Research-Boundary Audit",
        "",
        f"- M20 state: `{report['m20_state']}`",
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
