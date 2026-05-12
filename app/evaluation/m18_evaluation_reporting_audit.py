"""Artifact-backed M18 evaluation reporting and degradation audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m18/evaluation_reporting_audit"
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
    "evaluation_config_file",
    "evaluation_config_contract",
    "evaluation_request_contract",
    "decision_opportunity_contract",
    "divergence_event_contract",
    "evaluation_manifest_contract",
    "evaluation_report_contract",
    "paper_to_live_degradation_contract",
    "evaluation_service",
    "evaluation_service_generate_run",
    "evaluation_repository",
    "evaluation_cli",
    "evaluation_artifact_writer",
    "decision_opportunity_normalizer",
    "comparison_window_builder",
    "paper_to_live_degradation_metric",
    "cost_aware_precision_metric",
    "layer_comparison_metric",
    "uptime_failure_metric",
    "research_only_live_policy_challenger",
    "runner_challenger_observer",
    "challenger_operator_script",
    "m17_alerting_documented",
    "m20_pause_documentation",
)


def audit_m18_evaluation_reporting(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M18 evaluation/degradation controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _evaluation_surface_audit(root)
    reporting_rows = _reporting_contract_audit(root)
    degradation_rows = _degradation_boundary_audit(root)
    operator_rows = _operator_visibility_audit(root)
    gap_rows = _gap_analysis(surface_rows, reporting_rows, degradation_rows, operator_rows)
    state = _m18_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m18_evaluation_reporting_audit_v1",
        "repo_root": str(root),
        "m18_state": state,
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
            "evaluation_surfaces": surface_rows,
            "reporting_contracts": reporting_rows,
            "degradation_boundaries": degradation_rows,
            "operator_visibility": operator_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["evaluation_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["reporting_contract_audit_csv"]), reporting_rows)
    write_csv(Path(output_files["degradation_boundary_audit_csv"]), degradation_rows)
    write_csv(Path(output_files["operator_visibility_audit_csv"]), operator_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "evaluation_surfaces": surface_rows,
                "reporting_contracts": reporting_rows,
                "degradation_boundaries": degradation_rows,
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
            "evaluation_surface_audit": surface_rows,
            "reporting_contract_audit": reporting_rows,
            "degradation_boundary_audit": degradation_rows,
            "operator_visibility_audit": operator_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _evaluation_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(
            root, "evaluation_config_file", "configs/evaluation.yaml", "m18_evaluation_config_v1"
        ),
        _surface_check(
            root,
            "evaluation_config_contract",
            "app/evaluation/config.py",
            "class EvaluationConfig",
        ),
        _surface_check(
            root,
            "evaluation_request_contract",
            "app/evaluation/schemas.py",
            "class EvaluationRequest",
        ),
        _surface_check(
            root,
            "decision_opportunity_contract",
            "app/evaluation/schemas.py",
            "class DecisionOpportunity",
        ),
        _surface_check(
            root,
            "divergence_event_contract",
            "app/evaluation/schemas.py",
            "class DivergenceEvent",
        ),
        _surface_check(
            root,
            "evaluation_manifest_contract",
            "app/evaluation/schemas.py",
            "class EvaluationManifest",
        ),
        _surface_check(
            root,
            "evaluation_report_contract",
            "app/evaluation/schemas.py",
            "class EvaluationReport",
        ),
        _surface_check(
            root,
            "paper_to_live_degradation_contract",
            "app/evaluation/schemas.py",
            "class PaperToLiveDegradationReport",
        ),
        _surface_check(
            root,
            "evaluation_service",
            "app/evaluation/service.py",
            "class EvaluationService",
        ),
        _surface_check(
            root,
            "evaluation_service_generate_run",
            "app/evaluation/service.py",
            "async def generate_run",
        ),
        _surface_check(
            root,
            "evaluation_repository",
            "app/evaluation/repository.py",
            "class EvaluationRepository",
        ),
        _surface_check(root, "evaluation_cli", "app/evaluation/__main__.py", "EvaluationService"),
        _surface_check(
            root,
            "evaluation_artifact_writer",
            "app/evaluation/artifacts.py",
            "def write_evaluation_artifacts",
        ),
        _surface_check(
            root,
            "decision_opportunity_normalizer",
            "app/evaluation/normalize.py",
            "def build_decision_opportunities",
        ),
        _surface_check(
            root,
            "comparison_window_builder",
            "app/evaluation/matching.py",
            "def build_comparison_windows",
        ),
        _surface_check(
            root,
            "paper_to_live_degradation_metric",
            "app/evaluation/metrics.py",
            "def compute_paper_to_live_degradation",
        ),
        _surface_check(
            root,
            "cost_aware_precision_metric",
            "app/evaluation/metrics.py",
            "def compute_cost_aware_precision_by_mode",
        ),
        _surface_check(
            root,
            "layer_comparison_metric",
            "app/evaluation/metrics.py",
            "def compute_layer_comparison",
        ),
        _surface_check(
            root,
            "uptime_failure_metric",
            "app/evaluation/metrics.py",
            "def compute_uptime_and_failures",
        ),
        _surface_check(
            root,
            "research_only_live_policy_challenger",
            "app/training/live_policy_challenger.py",
            "class LivePolicyChallengerTracker",
        ),
        _surface_check(
            root,
            "runner_challenger_observer",
            "app/trading/runner.py",
            "_observe_live_policy_challengers",
        ),
        _surface_check(
            root,
            "challenger_operator_script",
            "scripts/show_live_policy_challengers.ps1",
            "Live policy challenger scoreboard",
        ),
        _docs_surface_check(
            root,
            "m17_alerting_documented",
            "M17_OPERATIONAL_ALERTING_INCIDENT_CONTROLS_CONSOLIDATED",
            "M17 alerting/incident status is documented before M18.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "M20 remains paused and non-authoritative.",
        ),
    ]


def _reporting_contract_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "evaluation_manifest_writer",
            "app/evaluation/artifacts.py",
            "evaluation_manifest.json",
        ),
        _audit_row(
            root,
            "evaluation_report_writer",
            "app/evaluation/artifacts.py",
            "evaluation_report.json",
        ),
        _audit_row(
            root,
            "decision_opportunities_writer",
            "app/evaluation/artifacts.py",
            "decision_opportunities.csv",
        ),
        _audit_row(
            root,
            "performance_by_asset_writer",
            "app/evaluation/artifacts.py",
            "performance_by_asset.csv",
        ),
        _audit_row(
            root,
            "performance_by_regime_writer",
            "app/evaluation/artifacts.py",
            "performance_by_regime.csv",
        ),
        _audit_row(
            root,
            "divergence_events_writer",
            "app/evaluation/artifacts.py",
            "divergence_events.csv",
        ),
        _audit_row(
            root,
            "latency_distribution_writer",
            "app/evaluation/artifacts.py",
            "latency_distribution.csv",
        ),
        _audit_row(
            root,
            "slippage_distribution_writer",
            "app/evaluation/artifacts.py",
            "slippage_distribution.csv",
        ),
        _audit_row(
            root,
            "evaluation_index_writer",
            "app/evaluation/service.py",
            "EVALUATION_INDEX_SCHEMA_VERSION",
        ),
        _audit_row(
            root,
            "experiment_index_writer",
            "app/evaluation/service.py",
            "EXPERIMENT_INDEX_SCHEMA_VERSION",
        ),
        _audit_row(
            root,
            "promotion_index_writer",
            "app/evaluation/service.py",
            "PROMOTION_INDEX_SCHEMA_VERSION",
        ),
    ]


def _degradation_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "paper_shadow_live_family_taxonomy",
            "app/evaluation/schemas.py",
            "paper_to_tiny_live",
        ),
        _audit_row(
            root,
            "divergence_stage_taxonomy",
            "app/evaluation/schemas.py",
            "fill_quality",
        ),
        _audit_row(
            root,
            "divergence_reason_taxonomy",
            "app/evaluation/schemas.py",
            "SLIPPAGE_DRIFT",
        ),
        _audit_row(
            root,
            "degradation_summary_writer",
            "app/evaluation/artifacts.py",
            "paper_to_live_degradation.json",
        ),
        _audit_row(
            root,
            "slippage_drift_threshold",
            "configs/evaluation.yaml",
            "slippage_drift_bps_threshold",
        ),
        _audit_row(
            root,
            "latency_drift_threshold",
            "configs/evaluation.yaml",
            "latency_drift_ms_threshold",
        ),
        _audit_row(
            root,
            "fill_price_drift_threshold",
            "configs/evaluation.yaml",
            "fill_price_drift_bps_threshold",
        ),
        _audit_row(
            root,
            "registry_context_is_read_only",
            "app/evaluation/service.py",
            "load_current_registry_entry",
        ),
        _audit_row(
            root,
            "probe_policy_is_observed_not_authoritative",
            "app/evaluation/repository.py",
            "probe_policy_active",
        ),
    ]


def _operator_visibility_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "evaluation_service_tests", "tests/test_evaluation_service.py", "M18"),
        _audit_row(root, "evaluation_metrics_tests", "tests/test_evaluation_metrics.py", "M18"),
        _audit_row(root, "evaluation_matching_tests", "tests/test_evaluation_matching.py", "M18"),
        _audit_row(root, "evaluation_normalize_tests", "tests/test_evaluation_normalize.py", "M18"),
        _audit_row(
            root,
            "live_policy_challenger_runner_test",
            "tests/trading/test_runner_idempotency.py",
            "policy_challengers",
        ),
        _audit_row(
            root,
            "live_policy_challenger_script_test",
            "tests/test_training_scripts.py",
            "show_live_policy_challengers.ps1",
        ),
        _audit_row(
            root,
            "vps_challenger_status",
            "scripts/status_paper_vps.ps1",
            "challenger artifacts exist",
        ),
        _audit_row(
            root,
            "vps_challenger_inspection",
            "scripts/show_live_policy_challengers_vps.ps1",
            "Paper VPS challenger scoreboard",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": status,
        "required_for_m18": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M18 evaluation/reporting surface `{name}` is {status.lower()}.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m18": "True",
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
        "detail": f"M18 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M18_EVALUATION_REPORTING_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M18 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m18_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_CONSOLIDATED"
    if present:
        return "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_PARTIAL"
    return "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if (
        state == "M18_EVALUATION_REPORTING_DEGRADATION_CONTROLS_CONSOLIDATED"
        and not gap_rows
    ):
        recommendation = "PROCEED_TO_M19_BOUNDED_ADAPTATION_AUDIT"
        next_required_action = "AUDIT_M19_BOUNDED_ADAPTATION_AND_DRIFT_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M18_EVALUATION_REPORTING_GAP_FILLS"
        next_required_action = "FILL_M18_EVALUATION_REPORTING_DEGRADATION_GAPS"
    else:
        recommendation = "RESTORE_M18_EVALUATION_REPORTING_PREREQUISITES"
        next_required_action = "RESTORE_M18_EVALUATION_REPORTING_PREREQUISITES"
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
        "schema_version": "m18_evaluation_reporting_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "evaluation_surface_count": len(audit_rows["evaluation_surfaces"]),
        "reporting_contract_count": len(audit_rows["reporting_contracts"]),
        "degradation_boundary_count": len(audit_rows["degradation_boundaries"]),
        "operator_visibility_count": len(audit_rows["operator_visibility"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m18_evaluation_reporting_audit.json"),
        "report_md": str(output_dir / "m18_evaluation_reporting_audit.md"),
        "evaluation_surface_audit_csv": str(output_dir / "evaluation_surface_audit.csv"),
        "reporting_contract_audit_csv": str(output_dir / "reporting_contract_audit.csv"),
        "degradation_boundary_audit_csv": str(output_dir / "degradation_boundary_audit.csv"),
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
        _section("Evaluation Surfaces", "surface_name", audit_rows["evaluation_surfaces"]),
        _section("Reporting Contracts", "audit_name", audit_rows["reporting_contracts"]),
        _section(
            "Degradation Boundaries",
            "audit_name",
            audit_rows["degradation_boundaries"],
        ),
        _section("Operator Visibility", "audit_name", audit_rows["operator_visibility"]),
    ]
    lines = [
        "# M18 Evaluation Reporting And Degradation Audit",
        "",
        f"- M18 state: `{report['m18_state']}`",
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
