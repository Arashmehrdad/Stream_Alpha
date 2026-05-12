"""Artifact-backed M10 risk-interface audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m10/risk_interface_audit"
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
    "risk_config_contract",
    "pure_risk_engine",
    "trade_allowed_blocker",
    "regime_position_caps",
    "service_risk_state_contract",
    "risk_decision_contract",
    "risk_decision_log_contract",
    "risk_decision_persistence",
    "decision_trace_risk_enrichment",
    "runner_risk_authority",
    "execution_after_risk_request",
    "canonical_regime_context_available",
    "m20_pause_documentation",
)


def audit_m10_risk_interface(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M10 risk-interface surfaces and write deterministic artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    risk_surface_rows = _risk_surface_audit(root)
    readiness_rows = _regime_context_readiness(root)
    boundary_rows = _authority_boundary_audit(root)
    gap_rows = _gap_analysis(risk_surface_rows, readiness_rows, boundary_rows)
    state = _m10_state(risk_surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m10_risk_interface_audit_v1",
        "repo_root": str(root),
        "m10_state": state,
        "critical_surface_count": len(CRITICAL_SURFACES),
        "present_critical_surface_count": sum(
            1
            for row in risk_surface_rows
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
            "risk_surfaces": risk_surface_rows,
            "regime_readiness": readiness_rows,
            "authority_boundaries": boundary_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["risk_surface_audit_csv"]), risk_surface_rows)
    write_csv(Path(output_files["regime_context_readiness_csv"]), readiness_rows)
    write_csv(Path(output_files["authority_boundary_audit_csv"]), boundary_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(report, risk_surface_rows, readiness_rows, boundary_rows, gap_rows),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "risk_surface_audit": risk_surface_rows,
            "regime_context_readiness": readiness_rows,
            "authority_boundary_audit": boundary_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _risk_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _file_contains_check(
            root,
            "risk_config_contract",
            Path("app/trading/config.py"),
            "class RiskConfig",
            "Typed M10 risk configuration contract exists.",
        ),
        _file_contains_check(
            root,
            "pure_risk_engine",
            Path("app/trading/risk_engine.py"),
            "def evaluate_risk",
            "Pure M10 risk engine entrypoint exists.",
        ),
        _file_contains_check(
            root,
            "trade_allowed_blocker",
            Path("app/trading/risk_engine.py"),
            "signal.trade_allowed is False",
            "Regime/M4 trade_allowed=False blocks new long entries.",
        ),
        _file_contains_check(
            root,
            "regime_position_caps",
            Path("app/trading/risk_engine.py"),
            "regime_position_fraction_caps",
            "M10 can clamp position fraction by regime label.",
        ),
        _file_contains_check(
            root,
            "service_risk_state_contract",
            Path("app/trading/schemas.py"),
            "class ServiceRiskState",
            "Restart-safe service risk state contract exists.",
        ),
        _file_contains_check(
            root,
            "risk_decision_contract",
            Path("app/trading/schemas.py"),
            "class RiskDecision",
            "RiskDecision contract carries risk outcome and sizing authority.",
        ),
        _file_contains_check(
            root,
            "risk_decision_log_contract",
            Path("app/trading/schemas.py"),
            "class RiskDecisionLogEntry",
            "Risk-decision persistence contract exists.",
        ),
        _file_contains_check(
            root,
            "risk_decision_persistence",
            Path("app/trading/repository.py"),
            "insert_risk_decision",
            "M10 risk decisions have a persistence surface.",
        ),
        _file_contains_check(
            root,
            "decision_trace_risk_enrichment",
            Path("app/trading/decision_trace.py"),
            "enrich_decision_trace_with_risk",
            "Decision traces can record M10 risk rationale.",
        ),
        _file_contains_check(
            root,
            "runner_risk_authority",
            Path("app/trading/runner.py"),
            "risk_decision = evaluate_risk",
            "Trading runner delegates risk authority to M10 before orders.",
        ),
        _file_contains_check(
            root,
            "execution_after_risk_request",
            Path("app/trading/execution.py"),
            "build_order_request",
            "Execution request construction is downstream of risk decisions.",
        ),
        _file_contains_check(
            root,
            "canonical_regime_context_available",
            Path("app/regime/context.py"),
            "REGIME_CONTEXT_SCHEMA_VERSION",
            "Canonical M9 RegimeContext contract is available to downstream gates.",
        ),
        _m20_pause_docs_check(root),
    ]


def _regime_context_readiness(root: Path) -> list[dict[str, str]]:
    return [
        _readiness_row(
            "signal_carries_regime_context",
            _file_contains(root, Path("app/trading/schemas.py"), "regime_label: str | None"),
            "SignalDecision exposes regime label/run metadata for M10.",
        ),
        _readiness_row(
            "risk_decision_carries_regime_context",
            _file_contains(
                root,
                Path("app/trading/schemas.py"),
                "trade_allowed: bool | None",
            ),
            "RiskDecision and logs can preserve trade_allowed/regime context.",
        ),
        _readiness_row(
            "risk_log_carries_regime_context",
            _file_contains(
                root,
                Path("app/trading/risk_engine.py"),
                "regime_run_id=signal.regime_run_id",
            ),
            "Risk-decision log builder propagates regime run id.",
        ),
        _readiness_row(
            "trade_allowed_fail_closed",
            _file_contains(root, Path("app/trading/risk_engine.py"), "return TRADE_NOT_ALLOWED"),
            "M10 fails closed when upstream regime policy disallows a trade.",
        ),
        _readiness_row(
            "regime_position_caps_configured",
            _file_contains(
                root,
                Path("configs/paper_trading.yaml"),
                "regime_position_fraction_caps",
            ),
            "Checked-in paper config contains regime-aware position caps.",
        ),
        _readiness_row(
            "canonical_regime_context_contract_available",
            _file_contains(root, Path("app/regime/context.py"), "missing_regime_context"),
            "M10 can rely on canonical missing/stale regime semantics from M9.",
        ),
    ]


def _authority_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _boundary_row(
            "prediction_to_risk",
            _file_contains(root, Path("app/trading/schemas.py"), "class SignalDecision"),
            "M4 signal payload is the input to M10 risk authority.",
        ),
        _boundary_row(
            "risk_to_order_request",
            _file_contains(root, Path("app/trading/runner.py"), "risk_decision")
            and _file_contains(root, Path("app/trading/runner.py"), "build_order_request"),
            "Order requests are built after a RiskDecision is available.",
        ),
        _boundary_row(
            "risk_to_persistence",
            _file_contains(root, Path("app/trading/repository.py"), "insert_risk_decision"),
            "Risk decisions are persisted separately from execution events.",
        ),
        _boundary_row(
            "risk_to_decision_trace",
            _file_contains(root, Path("app/trading/decision_trace.py"), "build_risk_section"),
            "Decision traces include a canonical risk section.",
        ),
        _boundary_row(
            "no_m20_research_authority",
            _docs_contain(root, "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"),
            "Paused M20 research artifacts have no authority over M10 runtime risk.",
        ),
        _boundary_row(
            "audit_only_no_runtime_change",
            True,
            "This M10 artifact audits interfaces and does not change runtime behavior.",
        ),
    ]


def _file_contains_check(
    root: Path,
    surface_name: str,
    relative_path: Path,
    needle: str,
    detail: str,
) -> dict[str, str]:
    status = "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING"
    return {
        "surface_name": surface_name,
        "path": str(relative_path),
        "status": status,
        "required_for_m10": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _m20_pause_docs_check(root: Path) -> dict[str, str]:
    status = (
        "PRESENT"
        if _docs_contain(root, "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY")
        else "MISSING"
    )
    return {
        "surface_name": "m20_pause_documentation",
        "path": "README.md|docs/training.md|PLANS.md",
        "status": status,
        "required_for_m10": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": "Docs preserve M20 as a paused negative research result.",
    }


def _readiness_row(name: str, ready: bool, detail: str) -> dict[str, str]:
    return {
        "readiness_name": name,
        "status": "READY" if ready else "MISSING",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _boundary_row(name: str, ready: bool, detail: str) -> dict[str, str]:
    return {
        "boundary_name": name,
        "status": "PRESENT" if ready else "MISSING",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
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
    risk_surface_rows: list[Mapping[str, str]],
    readiness_rows: list[Mapping[str, str]],
    boundary_rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in risk_surface_rows:
        if row["required_for_m10"] == "True" and row["status"] != "PRESENT":
            rows.append(
                _gap_row(
                    row["surface_name"],
                    "IMPLEMENT_REUSABLE_M10_RISK_INTERFACE_GAP_FILL",
                    row["detail"],
                )
            )
    for row in readiness_rows:
        if row["status"] != "READY":
            rows.append(
                _gap_row(
                    row["readiness_name"],
                    "RESTORE_REGIME_CONTEXT_READINESS_FOR_M10",
                    row["detail"],
                )
            )
    for row in boundary_rows:
        if row["status"] != "PRESENT":
            rows.append(
                _gap_row(
                    row["boundary_name"],
                    "RESTORE_M10_AUTHORITY_BOUNDARY",
                    row["detail"],
                )
            )
    return rows


def _gap_row(name: str, action: str, detail: str) -> dict[str, str]:
    return {
        "gap_name": name,
        "gap_status": "BLOCKING",
        "recommended_action": action,
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _m10_state(risk_surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in risk_surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M10_RISK_INTERFACE_CONSOLIDATED"
    if present:
        return "M10_RISK_INTERFACE_PARTIAL"
    return "M10_RISK_INTERFACE_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M10_RISK_INTERFACE_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M11_EXECUTION_INTERFACE_AUDIT"
        next_required_action = "AUDIT_M11_EXECUTION_INTERFACE_WITH_RISK_AUTHORITY"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M10_RISK_INTERFACE_GAP_FILLS"
        next_required_action = "FILL_M10_RISK_INTERFACE_GAPS"
    else:
        recommendation = "REVIEW_M10_RISK_INTERFACE_AUDIT_MANUALLY"
        next_required_action = "RESOLVE_M10_AUDIT_AMBIGUITY"
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
        for row in audit_rows["risk_surfaces"]
        if row.get("path") and "|" not in row["path"]
    }
    return {
        "schema_version": "m10_risk_interface_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "risk_surface_count": len(audit_rows["risk_surfaces"]),
        "regime_context_readiness_count": len(audit_rows["regime_readiness"]),
        "authority_boundary_count": len(audit_rows["authority_boundaries"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m10_risk_interface_audit.json"),
        "report_md": str(output_dir / "m10_risk_interface_audit.md"),
        "risk_surface_audit_csv": str(output_dir / "risk_surface_audit.csv"),
        "regime_context_readiness_csv": str(output_dir / "regime_context_readiness.csv"),
        "authority_boundary_audit_csv": str(output_dir / "authority_boundary_audit.csv"),
        "gap_analysis_csv": str(output_dir / "gap_analysis.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    risk_surface_rows: list[Mapping[str, str]],
    readiness_rows: list[Mapping[str, str]],
    boundary_rows: list[Mapping[str, str]],
    gap_rows: list[Mapping[str, str]],
) -> str:
    lines = [
        "# M10 Risk Interface Audit",
        "",
        f"- M10 state: `{report['m10_state']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        "- M20 status: `M20_PAUSED`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Risk Surfaces",
    ]
    for row in risk_surface_rows:
        lines.append(f"- `{row['surface_name']}`: `{row['status']}` ({row['path']})")
    lines.append("")
    lines.append("## RegimeContext Readiness")
    for row in readiness_rows:
        lines.append(f"- `{row['readiness_name']}`: `{row['status']}`")
    lines.append("")
    lines.append("## Authority Boundaries")
    for row in boundary_rows:
        lines.append(f"- `{row['boundary_name']}`: `{row['status']}`")
    lines.append("")
    lines.append("## Gaps")
    if gap_rows:
        for row in gap_rows:
            lines.append(f"- `{row['gap_name']}`: `{row['recommended_action']}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
