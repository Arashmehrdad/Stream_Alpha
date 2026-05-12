"""Artifact-backed M19 bounded adaptation and drift-control audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m19/bounded_adaptation_audit"
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
    "adaptation_config_file",
    "adaptation_config_contract",
    "drift_record_contract",
    "performance_window_contract",
    "adaptive_profile_contract",
    "promotion_decision_contract",
    "applied_adaptation_contract",
    "adaptation_service",
    "resolve_applied_adaptation",
    "runtime_truth_writer",
    "drift_detector",
    "rolling_performance_builder",
    "bounded_thresholds",
    "bounded_sizing",
    "calibration_application",
    "evidence_based_promotion_helper",
    "adaptation_repository_persistence",
    "adaptation_inference_endpoints",
    "runner_runtime_truth_hook",
    "dashboard_adaptation_visibility",
    "m18_evaluation_documented",
    "m20_pause_documentation",
)


def audit_m19_bounded_adaptation(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M19 bounded adaptation controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _adaptation_surface_audit(root)
    drift_rows = _drift_control_audit(root)
    promotion_rows = _promotion_boundary_audit(root)
    operator_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(surface_rows, drift_rows, promotion_rows, operator_rows)
    state = _m19_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m19_bounded_adaptation_audit_v1",
        "repo_root": str(root),
        "m19_state": state,
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
            "adaptation_surfaces": surface_rows,
            "drift_controls": drift_rows,
            "promotion_boundaries": promotion_rows,
            "operator_visibility": operator_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["adaptation_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["drift_control_audit_csv"]), drift_rows)
    write_csv(Path(output_files["promotion_boundary_audit_csv"]), promotion_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), operator_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "adaptation_surfaces": surface_rows,
                "drift_controls": drift_rows,
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
            "adaptation_surface_audit": surface_rows,
            "drift_control_audit": drift_rows,
            "promotion_boundary_audit": promotion_rows,
            "operator_visibility_audit": operator_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _adaptation_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "adaptation_config_file", "configs/adaptation.yaml", "drift:"),
        _surface_check(
            root,
            "adaptation_config_contract",
            "app/adaptation/config.py",
            "class AdaptationConfig",
        ),
        _surface_check(
            root,
            "drift_record_contract",
            "app/adaptation/schemas.py",
            "class AdaptiveDriftRecord",
        ),
        _surface_check(
            root,
            "performance_window_contract",
            "app/adaptation/schemas.py",
            "class AdaptivePerformanceWindow",
        ),
        _surface_check(
            root,
            "adaptive_profile_contract",
            "app/adaptation/schemas.py",
            "class AdaptiveProfileRecord",
        ),
        _surface_check(
            root,
            "promotion_decision_contract",
            "app/adaptation/schemas.py",
            "class AdaptivePromotionDecisionRecord",
        ),
        _surface_check(
            root,
            "applied_adaptation_contract",
            "app/adaptation/schemas.py",
            "class AppliedAdaptation",
        ),
        _surface_check(
            root,
            "adaptation_service",
            "app/adaptation/service.py",
            "class AdaptationService",
        ),
        _surface_check(
            root,
            "resolve_applied_adaptation",
            "app/adaptation/service.py",
            "async def resolve_applied_adaptation",
        ),
        _surface_check(
            root,
            "runtime_truth_writer",
            "app/adaptation/service.py",
            "async def write_runtime_persisted_truth",
        ),
        _surface_check(root, "drift_detector", "app/adaptation/drift.py", "def classify_drift"),
        _surface_check(
            root,
            "rolling_performance_builder",
            "app/adaptation/performance.py",
            "def build_rolling_performance_windows",
        ),
        _surface_check(
            root,
            "bounded_thresholds",
            "app/adaptation/thresholds.py",
            "def bounded_effective_thresholds",
        ),
        _surface_check(
            root,
            "bounded_sizing",
            "app/adaptation/sizing.py",
            "def bounded_size_multiplier",
        ),
        _surface_check(
            root,
            "calibration_application",
            "app/adaptation/calibration.py",
            "def apply_calibration",
        ),
        _surface_check(
            root,
            "evidence_based_promotion_helper",
            "app/adaptation/promotion.py",
            "def decide_promotion",
        ),
        _surface_check(
            root,
            "adaptation_repository_persistence",
            "app/trading/repository.py",
            "save_adaptive_drift_state",
        ),
        _surface_check(
            root,
            "adaptation_inference_endpoints",
            "app/inference/main.py",
            "/adaptation/summary",
        ),
        _surface_check(
            root,
            "runner_runtime_truth_hook",
            "app/trading/runner.py",
            "write_runtime_persisted_truth",
        ),
        _surface_check(
            root,
            "dashboard_adaptation_visibility",
            "dashboards/data_sources.py",
            "/adaptation/performance",
        ),
        _docs_surface_check(
            root,
            "m18_evaluation_documented",
            "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_CONSOLIDATED",
            "M18 evaluation/degradation status is documented before M19.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _drift_control_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "psi_drift_score",
            "app/adaptation/drift.py",
            "population_stability_index",
        ),
        _audit_row(root, "drift_warning_threshold", "configs/adaptation.yaml", "warning:"),
        _audit_row(root, "drift_breach_threshold", "configs/adaptation.yaml", "breach:"),
        _audit_row(
            root,
            "minimum_reference_samples",
            "configs/adaptation.yaml",
            "minimum_reference_samples",
        ),
        _audit_row(root, "minimum_live_samples", "configs/adaptation.yaml", "minimum_live_samples"),
        _audit_row(
            root,
            "feature_rows_for_adaptation_loader",
            "app/trading/repository.py",
            "load_feature_rows_for_adaptation",
        ),
        _audit_row(
            root,
            "adaptive_drift_state_persistence",
            "app/trading/repository.py",
            "save_adaptive_drift_state",
        ),
        _audit_row(
            root,
            "adaptive_performance_window_persistence",
            "app/trading/repository.py",
            "save_adaptive_performance_window",
        ),
        _audit_row(
            root,
            "drift_summary_artifact",
            "app/adaptation/service.py",
            "_write_drift_summary_artifact",
        ),
        _audit_row(
            root,
            "performance_summary_artifact",
            "app/adaptation/service.py",
            "_write_performance_summary_artifact",
        ),
    ]


def _promotion_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "freeze_on_drift_breach",
            "configs/adaptation.yaml",
            "freeze_on_drift_breach",
        ),
        _audit_row(
            root,
            "freeze_on_degraded_reliability",
            "configs/adaptation.yaml",
            "freeze_on_degraded_reliability",
        ),
        _audit_row(
            root,
            "m10_bounded_sizing",
            "app/adaptation/schemas.py",
            "bounded_by_m10",
        ),
        _audit_row(
            root,
            "rollback_active_profile",
            "app/adaptation/service.py",
            "async def rollback_active_profile",
        ),
        _audit_row(
            root,
            "rollback_repository",
            "app/trading/repository.py",
            "rollback_adaptive_profile",
        ),
        _audit_row(
            root,
            "promotion_thresholds",
            "configs/adaptation.yaml",
            "promotion_thresholds",
        ),
        _audit_row(
            root,
            "promotion_decision_persistence",
            "app/trading/repository.py",
            "save_adaptive_promotion_decision",
        ),
        _audit_row(
            root,
            "promotion_read_endpoint",
            "app/inference/main.py",
            "/adaptation/promotions",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "adaptation_service_tests", "tests/test_adaptation_service.py", "M19"),
        _audit_row(
            root,
            "adaptation_repository_tests",
            "tests/test_adaptation_repository.py",
            "M19",
        ),
        _audit_row(
            root,
            "inference_adaptation_tests",
            "tests/test_inference_api.py",
            "/adaptation/summary",
        ),
        _audit_row(
            root,
            "runner_runtime_truth_test",
            "tests/trading/test_runner_idempotency.py",
            "test_runner_persists_runtime_m19_drift_and_performance_truth",
        ),
        _audit_row(
            root,
            "dashboard_adaptation_tests",
            "tests/test_dashboard_data_sources.py",
            "/adaptation/drift",
        ),
        _audit_row(root, "readme_m19_status", "README.md", "M19 bounded adaptation"),
        _audit_row(root, "docs_m19_status", "docs/training.md", "M19"),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m19": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M19 bounded-adaptation surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m19": "True",
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
        "detail": f"M19 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M19_BOUNDED_ADAPTATION_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M19 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m19_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_CONSOLIDATED"
    if present:
        return "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_PARTIAL"
    return "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M19_BOUNDED_ADAPTATION_DRIFT_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M20_DYNAMIC_ENSEMBLE_AUDIT"
        next_required_action = "AUDIT_M20_DYNAMIC_ENSEMBLE_AND_RESEARCH_BOUNDARIES"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M19_BOUNDED_ADAPTATION_GAP_FILLS"
        next_required_action = "FILL_M19_BOUNDED_ADAPTATION_DRIFT_GAPS"
    else:
        recommendation = "RESTORE_M19_BOUNDED_ADAPTATION_PREREQUISITES"
        next_required_action = "RESTORE_M19_BOUNDED_ADAPTATION_PREREQUISITES"
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
        "schema_version": "m19_bounded_adaptation_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "adaptation_surface_count": len(audit_rows["adaptation_surfaces"]),
        "drift_control_count": len(audit_rows["drift_controls"]),
        "promotion_boundary_count": len(audit_rows["promotion_boundaries"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m19_bounded_adaptation_audit.json"),
        "report_md": str(output_dir / "m19_bounded_adaptation_audit.md"),
        "adaptation_surface_audit_csv": str(output_dir / "adaptation_surface_audit.csv"),
        "drift_control_audit_csv": str(output_dir / "drift_control_audit.csv"),
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
        _section("Adaptation Surfaces", "surface_name", audit_rows["adaptation_surfaces"]),
        _section("Drift Controls", "audit_name", audit_rows["drift_controls"]),
        _section("Promotion Boundaries", "audit_name", audit_rows["promotion_boundaries"]),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M19 Bounded Adaptation And Drift-Control Audit",
        "",
        f"- M19 state: `{report['m19_state']}`",
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
