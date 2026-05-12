"""Artifact-backed M9 regime integration audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic
from app.regime.context import REGIME_CONTEXT_SCHEMA_VERSION, missing_regime_context
from app.regime.live import SIGNAL_POLICY_SCHEMA_VERSION, THRESHOLDS_SCHEMA_VERSION


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m9/regime_integration_audit"
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
    "m8_threshold_artifact",
    "m9_signal_policy",
    "canonical_regime_context_contract",
    "runtime_regime_resolver",
    "regime_endpoint",
    "signal_regime_fields",
    "freshness_regime_fields",
    "decision_trace_regime_reason",
    "m20_pause_documentation",
)


def audit_m9_regime_integration(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit existing M9 regime integration and write deterministic artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _surface_audit(root)
    gap_rows = _gap_analysis(surface_rows)
    state = _m9_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m9_regime_integration_audit_v1",
        "repo_root": str(root),
        "m9_state": state,
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
    manifest = {
        "schema_version": "m9_regime_integration_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_paths": sorted(
            {
                row["path"]
                for row in surface_rows
                if row["path"]
            }
        ),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    contract = _regime_context_contract()
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["contract_json"]), contract)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(report, surface_rows, gap_rows, recommendation),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "regime_context_contract": contract,
            "surface_audit": surface_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _surface_audit(root: Path) -> list[dict[str, str]]:
    checks = [
        _m8_threshold_artifact_check(
            root,
            "m8_threshold_artifact",
            Path("artifacts/regime/m8"),
            "Saved M8 thresholds are available for deterministic regime resolution.",
        ),
        _json_schema_check(
            root,
            "m9_signal_policy",
            Path("configs/regime_signal_policy.json"),
            SIGNAL_POLICY_SCHEMA_VERSION,
            "Checked-in M9 policy defines per-regime thresholds and long-entry gates.",
        ),
        _file_contains_check(
            root,
            "canonical_regime_context_contract",
            Path("app/regime/context.py"),
            REGIME_CONTEXT_SCHEMA_VERSION,
            "Canonical RegimeContext contract exists.",
        ),
        _file_contains_check(
            root,
            "runtime_regime_resolver",
            Path("app/regime/live.py"),
            "resolve_feature_row_regime",
            "Live runtime resolves exact-row regimes from M8 thresholds.",
        ),
        _file_contains_check(
            root,
            "regime_endpoint",
            Path("app/inference/main.py"),
            '@app.get("/regime"',
            "Inference API exposes a read-only /regime surface.",
        ),
        _file_contains_check(
            root,
            "signal_regime_fields",
            Path("app/inference/schemas.py"),
            "regime_label: str | None",
            "Signal response schema includes regime metadata.",
        ),
        _file_contains_check(
            root,
            "freshness_regime_fields",
            Path("app/inference/schemas.py"),
            "regime_freshness_status: str",
            "Freshness response schema exposes regime freshness.",
        ),
        _file_contains_check(
            root,
            "decision_trace_regime_reason",
            Path("app/trading/decision_trace.py"),
            "regime_reason",
            "Decision traces preserve regime reasoning.",
        ),
        _file_contains_check(
            root,
            "risk_regime_fields",
            Path("app/trading/schemas.py"),
            "regime_label: str | None",
            "Risk and execution schemas carry regime metadata.",
        ),
        _m20_pause_docs_check(root),
    ]
    return checks


def _m8_threshold_artifact_check(
    root: Path,
    surface_name: str,
    relative_path: Path,
    detail: str,
) -> dict[str, str]:
    path = root / relative_path
    status = "MISSING"
    if path.is_dir():
        candidates = sorted(path.glob("*/thresholds.json"), reverse=True)
        if candidates:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            status = (
                "PRESENT"
                if payload.get("schema_version") == THRESHOLDS_SCHEMA_VERSION
                else "SCHEMA_MISMATCH"
            )
    return _surface_row(surface_name, relative_path, status, detail, required=True)


def _json_schema_check(
    root: Path,
    surface_name: str,
    relative_path: Path,
    expected_schema: str,
    detail: str,
) -> dict[str, str]:
    path = root / relative_path
    status = "MISSING"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = (
            "PRESENT"
            if str(payload.get("schema_version")) == expected_schema
            else "SCHEMA_MISMATCH"
        )
    return _surface_row(surface_name, relative_path, status, detail, required=True)


def _file_contains_check(
    root: Path,
    surface_name: str,
    relative_path: Path,
    needle: str,
    detail: str,
) -> dict[str, str]:
    path = root / relative_path
    status = "MISSING"
    if path.is_file():
        status = "PRESENT" if needle in path.read_text(encoding="utf-8") else "MISSING"
    return _surface_row(surface_name, relative_path, status, detail, required=True)


def _m20_pause_docs_check(root: Path) -> dict[str, str]:
    docs = (Path("README.md"), Path("docs/training.md"), Path("PLANS.md"))
    present = any(
        (root / path).is_file()
        and "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
        in (root / path).read_text(encoding="utf-8")
        for path in docs
    )
    status = "PRESENT" if present else "MISSING"
    return _surface_row(
        "m20_pause_documentation",
        Path("README.md|docs/training.md|PLANS.md"),
        status,
        "Docs preserve M20 as a paused negative research result.",
        required=True,
    )


def _surface_row(
    surface_name: str,
    relative_path: Path,
    status: str,
    detail: str,
    *,
    required: bool,
) -> dict[str, str]:
    return {
        "surface_name": surface_name,
        "path": str(relative_path),
        "status": status,
        "required_for_m9": str(required),
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _gap_analysis(surface_rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in surface_rows:
        if row["required_for_m9"] == "True" and row["status"] != "PRESENT":
            rows.append(
                {
                    "gap_name": row["surface_name"],
                    "gap_status": "BLOCKING",
                    "recommended_action": "IMPLEMENT_MISSING_REUSABLE_REGIME_SURFACE",
                    "path": row["path"],
                    "detail": row["detail"],
                }
            )
    return rows


def _m9_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M9_REGIME_INTEGRATION_CONSOLIDATED"
    if present:
        return "M9_REGIME_INTEGRATION_PARTIAL"
    return "M9_REGIME_INTEGRATION_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M9_REGIME_INTEGRATION_CONSOLIDATED":
        recommendation = "PROCEED_TO_M10_RISK_INTERFACE_AUDIT"
        next_required_action = "AUDIT_M10_RISK_INTERFACE_WITH_REGIME_CONTEXT"
    elif gap_rows:
        recommendation = "FILL_MISSING_M9_REGIME_GAPS"
        next_required_action = "IMPLEMENT_REUSABLE_M9_REGIME_GAP_FILLS"
    else:
        recommendation = "REVIEW_M9_REGIME_AUDIT_MANUALLY"
        next_required_action = "RESOLVE_M9_AUDIT_AMBIGUITY"
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


def _regime_context_contract() -> dict[str, Any]:
    missing = missing_regime_context(reason_code="CONTRACT_EXAMPLE_MISSING")
    return {
        "schema_version": REGIME_CONTEXT_SCHEMA_VERSION,
        "description": "Canonical M9 regime context contract for read surfaces and gates.",
        "required_fields": sorted(missing.to_dict()),
        "present_behavior": "FRESH + HEALTHY context may feed configured regime policy.",
        "missing_or_stale_behavior": (
            "Fail closed to HOLD/blocking gates; do not infer M20 authority."
        ),
        "m20_research_authority": False,
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "example_missing_context": missing.to_dict(),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m9_regime_integration_audit.json"),
        "report_md": str(output_dir / "m9_regime_integration_audit.md"),
        "surface_audit_csv": str(output_dir / "surface_audit.csv"),
        "contract_json": str(output_dir / "regime_context_contract.json"),
        "gap_analysis_csv": str(output_dir / "gap_analysis.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    surface_rows: list[Mapping[str, str]],
    gap_rows: list[Mapping[str, str]],
    recommendation: Mapping[str, Any],
) -> str:
    lines = [
        "# M9 Regime Integration Audit",
        "",
        f"- M9 state: `{report['m9_state']}`",
        f"- Recommendation: `{recommendation['recommendation']}`",
        f"- Next required action: `{recommendation['next_required_action']}`",
        "- M20 status: `M20_PAUSED`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Surface Audit",
    ]
    for row in surface_rows:
        lines.append(f"- `{row['surface_name']}`: `{row['status']}` ({row['path']})")
    lines.append("")
    lines.append("## Gaps")
    if gap_rows:
        for row in gap_rows:
            lines.append(f"- `{row['gap_name']}`: `{row['recommended_action']}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
