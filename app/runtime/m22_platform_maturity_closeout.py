"""Second Foundation platform-maturity closeout artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m22/platform_maturity_closeout"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
COMPLETE_STATE = "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_COMPLETE"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
AUDIT_SPECS = (
    ("m9", "regime_integration_audit", "m9_regime_integration_audit.json"),
    ("m10", "risk_interface_audit", "m10_risk_interface_audit.json"),
    ("m11", "execution_interface_audit", "m11_execution_interface_audit.json"),
    ("m12", "guarded_live_audit", "m12_guarded_live_audit.json"),
    ("m13", "reliability_recovery_audit", "m13_reliability_recovery_audit.json"),
    ("m14", "explainability_audit", "m14_explainability_audit.json"),
    ("m15", "operator_console_audit", "m15_operator_console_audit.json"),
    ("m16", "deployment_environment_audit", "m16_deployment_environment_audit.json"),
    ("m17", "operational_alerting_audit", "m17_operational_alerting_audit.json"),
    ("m18", "evaluation_reporting_audit", "m18_evaluation_reporting_audit.json"),
    ("m19", "bounded_adaptation_audit", "m19_bounded_adaptation_audit.json"),
    ("m20", "dynamic_ensemble_audit", "m20_dynamic_ensemble_audit.json"),
    ("m21", "continual_learning_audit", "m21_continual_learning_audit.json"),
)


def write_m22_platform_maturity_closeout(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write one canonical M9-M21 platform maturity closeout artifact set."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = _audit_evidence_index(root)
    status_rows = _milestone_status_rollup(audit_rows)
    limitations = _remaining_limitations()
    non_claims = _non_claims()
    state = _platform_state(status_rows)
    recommendation = _recommendation(state)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m22_platform_maturity_closeout_v1",
        "repo_root": str(root),
        "platform_maturity_state": state,
        "milestone_count": len(status_rows),
        "consolidated_milestone_count": sum(
            1 for row in status_rows if row["status"] == "CONSOLIDATED"
        ),
        "gap_count": sum(int(row["gap_count"]) for row in status_rows),
        "m20_research_decision": M20_DECISION,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "m22_platform_maturity_closeout_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_artifacts": [row["report_path"] for row in audit_rows],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, status_rows, limitations, non_claims),
        encoding="utf-8",
    )
    write_csv(Path(output_files["audit_evidence_index_csv"]), audit_rows)
    write_csv(Path(output_files["milestone_status_rollup_csv"]), status_rows)
    write_csv(Path(output_files["remaining_limitations_csv"]), limitations)
    write_csv(Path(output_files["non_claims_csv"]), non_claims)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "audit_evidence_index": audit_rows,
            "milestone_status_rollup": status_rows,
            "remaining_limitations": limitations,
            "non_claims": non_claims,
            "recommendation_payload": recommendation,
        }
    )


def _audit_evidence_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for milestone, directory, report_name in AUDIT_SPECS:
        artifact_dir = root / "artifacts" / "platform_maturity" / milestone / directory
        report_path = artifact_dir / report_name
        recommendation_path = artifact_dir / "recommendation.json"
        report_payload = _read_optional_json(report_path)
        recommendation_payload = _read_optional_json(recommendation_path)
        state_value = _state_value(report_payload, milestone)
        rows.append(
            {
                "milestone": milestone.upper(),
                "artifact_dir": str(artifact_dir),
                "report_path": str(report_path),
                "recommendation_path": str(recommendation_path),
                "report_present": report_path.is_file(),
                "recommendation_present": recommendation_path.is_file(),
                "state": state_value,
                "gap_count": int(report_payload.get("gap_count", 999)),
                "recommendation": recommendation_payload.get("recommendation", ""),
                "next_required_action": recommendation_payload.get("next_required_action", ""),
                "m20_research_decision": report_payload.get("m20_research_decision", M20_DECISION),
            }
        )
    return rows


def _milestone_status_rollup(audit_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "milestone": row["milestone"],
            "status": _milestone_status(row),
            "state": row["state"],
            "gap_count": row["gap_count"],
            "artifact_present": row["report_present"] and row["recommendation_present"],
            "runtime_effect": "NO_RUNTIME_EFFECT",
            "promotable": False,
            "profitability_claim": False,
        }
        for row in audit_rows
    ]


def _milestone_status(row: Mapping[str, Any]) -> str:
    if not row["report_present"] or not row["recommendation_present"]:
        return "MISSING_AUDIT_ARTIFACT"
    if row["gap_count"] == 0 and "CONSOLIDATED" in str(row["state"]):
        return "CONSOLIDATED"
    return "NOT_CONSOLIDATED"


def _platform_state(status_rows: list[Mapping[str, Any]]) -> str:
    if all(row["status"] == "CONSOLIDATED" for row in status_rows):
        return COMPLETE_STATE
    if any(row["status"] == "CONSOLIDATED" for row in status_rows):
        return "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_PARTIAL"
    return "SECOND_FOUNDATION_PLATFORM_MATURITY_AUDITS_BLOCKED"


def _state_value(payload: Mapping[str, Any], milestone: str) -> str:
    state_key = f"{milestone}_state"
    if state_key in payload:
        return str(payload[state_key])
    return str(payload.get("platform_maturity_state", "MISSING_AUDIT_REPORT"))


def _remaining_limitations() -> list[dict[str, str]]:
    return [
        _limitation("m20_alpha_evidence", "M20 remains paused after no positive proxy route."),
        _limitation("model_breadth", "Platform controls do not prove stronger model diversity."),
        _limitation("data_breadth", "OHLC-only short-horizon research is likely input-limited."),
        _limitation(
            "economic_acceptance",
            "Platform maturity does not create economic acceptance.",
        ),
        _limitation("live_readiness", "Audit consolidation is not live-trading readiness."),
    ]


def _limitation(name: str, detail: str) -> dict[str, str]:
    return {
        "limitation": name,
        "detail": detail,
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "m20_status": "M20_PAUSED",
    }


def _non_claims() -> list[dict[str, str]]:
    return [
        _non_claim("NOT_BACKTEST", "No new backtest claim is created."),
        _non_claim("NOT_RUNTIME_READY", "No runtime-readiness claim is created."),
        _non_claim("NOT_PROMOTABLE", "No promotion claim is created."),
        _non_claim("NO_PROFIT_CLAIM", "No profitability claim is created."),
        _non_claim("NO_TRADING_AUTHORITY_CHANGE", "No trading authority is changed."),
    ]


def _non_claim(name: str, detail: str) -> dict[str, str]:
    return {"claim": name, "status": "PRESERVED", "detail": detail}


def _recommendation(state: str) -> dict[str, Any]:
    if state == COMPLETE_STATE:
        recommendation = "PLAN_DATA_UPGRADE_BEFORE_REOPENING_ALPHA_RESEARCH"
        next_required_action = "DESIGN_RESEARCH_DATA_UPGRADE_FEASIBILITY_PLAN"
    else:
        recommendation = "RESOLVE_PLATFORM_MATURITY_AUDIT_GAPS"
        next_required_action = "RESTORE_MISSING_OR_PARTIAL_PLATFORM_AUDITS"
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


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m22_platform_maturity_closeout.json"),
        "report_md": str(output_dir / "m22_platform_maturity_closeout.md"),
        "audit_evidence_index_csv": str(output_dir / "audit_evidence_index.csv"),
        "milestone_status_rollup_csv": str(output_dir / "milestone_status_rollup.csv"),
        "remaining_limitations_csv": str(output_dir / "remaining_limitations.csv"),
        "non_claims_csv": str(output_dir / "non_claims.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    status_rows: list[Mapping[str, Any]],
    limitations: list[Mapping[str, str]],
    non_claims: list[Mapping[str, str]],
) -> str:
    lines = [
        "# M22 Platform Maturity Closeout",
        "",
        f"- State: `{report['platform_maturity_state']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Milestones",
    ]
    lines.extend(f"- `{row['milestone']}`: `{row['status']}`" for row in status_rows)
    lines.append("")
    lines.append("## Remaining Limitations")
    lines.extend(f"- `{row['limitation']}`: {row['detail']}" for row in limitations)
    lines.append("")
    lines.append("## Non-Claims")
    lines.extend(f"- `{row['claim']}`: {row['detail']}" for row in non_claims)
    return "\n".join(lines) + "\n"
