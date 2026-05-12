"""Artifact-backed M14 explainability and decision-trace audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m14/explainability_audit"
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
    "explainability_config_file",
    "explainability_config_contract",
    "prediction_explanation_contract",
    "top_feature_contribution_contract",
    "signal_explanation_contract",
    "decision_trace_payload_contract",
    "decision_trace_record_contract",
    "reference_ablation_method",
    "prediction_details_builder",
    "signal_explanation_builder",
    "regime_reason_builder",
    "initial_decision_trace_builder",
    "risk_trace_enrichment",
    "rationale_report_writer",
    "decision_trace_persistence",
    "decision_trace_update",
    "decision_trace_loading",
    "order_request_trace_linkage",
    "order_event_trace_linkage",
    "position_trace_linkage",
    "ledger_trace_linkage",
    "prediction_explainability_endpoint_surface",
    "signal_explainability_endpoint_surface",
    "dashboard_decision_trace_surface",
    "m13_reliability_documented",
    "m20_pause_documentation",
)


def audit_m14_explainability(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M14 explainability/decision-trace controls and write artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _explainability_surface_audit(root)
    trace_rows = _decision_trace_audit(root)
    linkage_rows = _linkage_audit(root)
    report_rows = _rationale_artifact_audit(root)
    gap_rows = _gap_analysis(surface_rows, trace_rows, linkage_rows, report_rows)
    state = _m14_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m14_explainability_audit_v1",
        "repo_root": str(root),
        "m14_state": state,
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
            "explainability_surfaces": surface_rows,
            "decision_trace": trace_rows,
            "linkage": linkage_rows,
            "rationale_artifacts": report_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["explainability_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["decision_trace_audit_csv"]), trace_rows)
    write_csv(Path(output_files["linkage_audit_csv"]), linkage_rows)
    write_csv(Path(output_files["rationale_artifact_audit_csv"]), report_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "explainability_surfaces": surface_rows,
                "decision_trace": trace_rows,
                "linkage": linkage_rows,
                "rationale_artifacts": report_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "explainability_surface_audit": surface_rows,
            "decision_trace_audit": trace_rows,
            "linkage_audit": linkage_rows,
            "rationale_artifact_audit": report_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _explainability_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "explainability_config_file", "configs/explainability.yaml", "m14"),
        _surface_check(
            root,
            "explainability_config_contract",
            "app/explainability/config.py",
            "class ExplainabilityConfig",
        ),
        _surface_check(
            root,
            "prediction_explanation_contract",
            "app/explainability/schemas.py",
            "class PredictionExplanation",
        ),
        _surface_check(
            root,
            "top_feature_contribution_contract",
            "app/explainability/schemas.py",
            "class TopFeatureContribution",
        ),
        _surface_check(
            root,
            "signal_explanation_contract",
            "app/explainability/schemas.py",
            "class SignalExplanation",
        ),
        _surface_check(
            root,
            "decision_trace_payload_contract",
            "app/explainability/schemas.py",
            "class DecisionTracePayload",
        ),
        _surface_check(
            root,
            "decision_trace_record_contract",
            "app/trading/schemas.py",
            "class DecisionTraceRecord",
        ),
        _surface_check(
            root,
            "reference_ablation_method",
            "app/explainability/service.py",
            "REFERENCE_ABLATION_METHOD",
        ),
        _surface_check(
            root,
            "prediction_details_builder",
            "app/explainability/service.py",
            "build_prediction_details",
        ),
        _surface_check(
            root,
            "signal_explanation_builder",
            "app/explainability/service.py",
            "build_signal_explanation",
        ),
        _surface_check(
            root,
            "regime_reason_builder",
            "app/explainability/service.py",
            "def build_regime_reason",
        ),
        _surface_check(
            root,
            "initial_decision_trace_builder",
            "app/trading/decision_trace.py",
            "def build_initial_decision_trace",
        ),
        _surface_check(
            root,
            "risk_trace_enrichment",
            "app/trading/decision_trace.py",
            "def enrich_decision_trace_with_risk",
        ),
        _surface_check(
            root,
            "rationale_report_writer",
            "app/trading/decision_trace.py",
            "def write_rationale_reports",
        ),
        _surface_check(
            root,
            "decision_trace_persistence",
            "app/trading/repository.py",
            "ensure_decision_trace",
        ),
        _surface_check(
            root,
            "decision_trace_update",
            "app/trading/repository.py",
            "update_decision_trace",
        ),
        _surface_check(
            root,
            "decision_trace_loading",
            "app/trading/repository.py",
            "load_decision_trace",
        ),
        _surface_check(
            root,
            "order_request_trace_linkage",
            "app/trading/repository.py",
            "order_request.decision_trace_id",
        ),
        _surface_check(
            root,
            "order_event_trace_linkage",
            "app/trading/repository.py",
            "event.decision_trace_id",
        ),
        _surface_check(
            root,
            "position_trace_linkage",
            "app/trading/repository.py",
            "entry_decision_trace_id",
        ),
        _surface_check(
            root,
            "ledger_trace_linkage",
            "app/trading/repository.py",
            "decision_trace_id",
        ),
        _surface_check(
            root,
            "prediction_explainability_endpoint_surface",
            "app/inference/service.py",
            "_build_prediction_explainability",
        ),
        _surface_check(
            root,
            "signal_explainability_endpoint_surface",
            "app/inference/service.py",
            "build_signal_explanation",
        ),
        _surface_check(
            root,
            "dashboard_decision_trace_surface",
            "tests/test_dashboard_data_sources.py",
            "recent_decision_traces",
        ),
        _docs_surface_check(
            root,
            "m13_reliability_documented",
            "M13_RELIABILITY_RECOVERY_CONTROLS_CONSOLIDATED",
            "Docs preserve M13 as consolidated upstream reliability truth.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "Docs preserve M20 as paused and non-authoritative.",
        ),
    ]


def _decision_trace_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "trace_schema_version",
            "app/trading/decision_trace.py",
            "m14_decision_trace_v1",
        ),
        _audit_row(
            root,
            "rationale_schema_version",
            "app/trading/decision_trace.py",
            "m14_rationale_report_v1",
        ),
        _audit_row(
            root,
            "prediction_section",
            "app/explainability/schemas.py",
            "DecisionTracePrediction",
        ),
        _audit_row(root, "signal_section", "app/explainability/schemas.py", "DecisionTraceSignal"),
        _audit_row(root, "risk_section", "app/explainability/schemas.py", "DecisionTraceRisk"),
        _audit_row(
            root,
            "blocked_trade_section",
            "app/explainability/schemas.py",
            "DecisionTraceBlockedTrade",
        ),
        _audit_row(
            root,
            "threshold_snapshot",
            "app/explainability/schemas.py",
            "ThresholdSnapshot",
        ),
        _audit_row(root, "regime_reason", "app/explainability/schemas.py", "RegimeReason"),
    ]


def _linkage_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "runner_ensures_initial_trace",
            "app/trading/runner.py",
            "ensure_decision_trace",
        ),
        _audit_row(root, "runner_updates_trace", "app/trading/runner.py", "update_decision_trace"),
        _audit_row(
            root,
            "runner_writes_rationale",
            "app/trading/runner.py",
            "write_rationale_reports",
        ),
        _audit_row(
            root,
            "execution_carries_trace_id",
            "app/trading/execution.py",
            "decision_trace_id",
        ),
        _audit_row(
            root,
            "risk_carries_trace_id",
            "app/trading/risk_engine.py",
            "decision_trace_id",
        ),
        _audit_row(
            root,
            "dashboard_renders_trace_paths",
            "tests/test_dashboard_data_sources.py",
            "json_report_path",
        ),
    ]


def _rationale_artifact_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "reference_artifact_root",
            "configs/explainability.yaml",
            "artifact_root",
        ),
        _audit_row(
            root,
            "reference_filename",
            "configs/explainability.yaml",
            "reference_filename",
        ),
        _audit_row(
            root,
            "top_feature_count",
            "configs/explainability.yaml",
            "top_feature_count",
        ),
        _audit_row(
            root,
            "rationale_json_path",
            "app/trading/decision_trace.py",
            "json_report_path",
        ),
        _audit_row(
            root,
            "rationale_markdown_path",
            "app/trading/decision_trace.py",
            "markdown_report_path",
        ),
        _audit_row(
            root,
            "deterministic_report_paths",
            "app/trading/decision_trace.py",
            "resolve_rationale_report_paths",
        ),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING",
        "required_for_m14": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M14 explainability surface `{name}` is present.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m14": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _audit_row(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    return {
        "audit_name": name,
        "path": str(relative_path),
        "status": "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M14 audit `{name}` checks `{needle}`.",
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
                    "recommended_action": "RESTORE_M14_EXPLAINABILITY_OR_TRACE_SURFACE",
                    "runtime_authority_changed": "False",
                    "m20_reopened": "False",
                    "detail": "M14 audit-only finding; no runtime behavior changed.",
                }
            )
    return rows


def _m14_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_CONSOLIDATED"
    if present:
        return "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_PARTIAL"
    return "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M14_EXPLAINABILITY_DECISION_TRACE_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M15_OPERATOR_CONSOLE_AUDIT"
        next_required_action = "AUDIT_M15_OPERATOR_CONSOLE_AND_VISIBILITY_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M14_EXPLAINABILITY_GAP_FILLS"
        next_required_action = "FILL_M14_EXPLAINABILITY_DECISION_TRACE_GAPS"
    else:
        recommendation = "RESTORE_M14_EXPLAINABILITY_PREREQUISITES"
        next_required_action = "RESTORE_M14_EXPLAINABILITY_PREREQUISITES"
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
        "schema_version": "m14_explainability_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "explainability_surface_count": len(audit_rows["explainability_surfaces"]),
        "decision_trace_count": len(audit_rows["decision_trace"]),
        "linkage_count": len(audit_rows["linkage"]),
        "rationale_artifact_count": len(audit_rows["rationale_artifacts"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m14_explainability_audit.json"),
        "report_md": str(output_dir / "m14_explainability_audit.md"),
        "explainability_surface_audit_csv": str(output_dir / "explainability_surface_audit.csv"),
        "decision_trace_audit_csv": str(output_dir / "decision_trace_audit.csv"),
        "linkage_audit_csv": str(output_dir / "linkage_audit.csv"),
        "rationale_artifact_audit_csv": str(output_dir / "rationale_artifact_audit.csv"),
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
        _section("Explainability Surfaces", "surface_name", audit_rows["explainability_surfaces"]),
        _section("Decision Trace", "audit_name", audit_rows["decision_trace"]),
        _section("Trace Linkage", "audit_name", audit_rows["linkage"]),
        _section("Rationale Artifacts", "audit_name", audit_rows["rationale_artifacts"]),
    ]
    lines = [
        "# M14 Explainability And Decision Trace Audit",
        "",
        f"- M14 state: `{report['m14_state']}`",
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
