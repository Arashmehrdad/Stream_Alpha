"""Research-only microstructure coverage, gap, and replay audit writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_replay_audit"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "AUDIT_ONLY",
    "NO_CAPTURE_SERVICE",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_microstructure_replay_audit(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the DU5 microstructure coverage/gap/replay audit artifact."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = _audit_rows(root)
    recommendation = _recommendation(rows["source_artifact_audit_csv"])
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_replay_audit_v1",
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "audit_status": _audit_status(rows["source_artifact_audit_csv"]),
        "source_artifact_count": len(rows["source_artifact_audit_csv"]),
        "present_source_artifact_count": sum(
            1
            for row in rows["source_artifact_audit_csv"]
            if row["status"] == "PRESENT"
        ),
        "stored_replay_rows_available": False,
        "coverage_gap_metric_status": "BLOCKED_NO_STORED_MICROSTRUCTURE_REPLAY_ROWS",
        "replay_determinism_status": "CONTRACT_DEFINED_DATA_NOT_AVAILABLE",
        "blocked_decision_count": len(rows["blocked_decisions_csv"]),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_replay_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    _write_outputs(
        output_files=output_files,
        manifest=manifest,
        report=report,
        recommendation=recommendation,
        rows_by_file={**rows, "next_actions_csv": recommendation["next_actions"]},
    )
    Path(output_files["report_md"]).write_text(_markdown(report), encoding="utf-8")
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "source_artifact_audit": rows["source_artifact_audit_csv"],
            "coverage_gap_audit": rows["coverage_gap_audit_csv"],
            "replay_determinism_audit": rows["replay_determinism_audit_csv"],
            "data_readiness_audit": rows["data_readiness_audit_csv"],
            "blocked_decisions": rows["blocked_decisions_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _audit_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    source_rows = _source_artifact_audit(root)
    return {
        "source_artifact_audit_csv": source_rows,
        "coverage_gap_audit_csv": _coverage_gap_audit(),
        "replay_determinism_audit_csv": _replay_determinism_audit(source_rows),
        "data_readiness_audit_csv": _data_readiness_audit(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _source_artifact_audit(root: Path) -> list[dict[str, str]]:
    return [
        _artifact(
            root,
            "schema_contracts",
            "artifacts/research_data_upgrade/microstructure_schema_contracts/"
            "microstructure_schema_contracts.json",
        ),
        _artifact(
            root,
            "book_payload_normalizer_contract",
            "artifacts/research_data_upgrade/book_payload_normalizer_contract/"
            "book_payload_normalizer_contract.json",
        ),
        _artifact(
            root,
            "microstructure_feature_derivation",
            "artifacts/research_data_upgrade/microstructure_feature_derivation/"
            "microstructure_feature_derivation.json",
        ),
    ]


def _artifact(root: Path, name: str, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    status = "PRESENT" if path.is_file() else "MISSING"
    schema_version = ""
    if path.is_file():
        schema_version = str(json.loads(path.read_text(encoding="utf-8")).get("schema_version", ""))
    return {
        "artifact": name,
        "path": relative_path,
        "status": status,
        "schema_version": schema_version,
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _coverage_gap_audit() -> list[dict[str, str]]:
    return [
        _coverage_row("research_raw_order_book", "BLOCKED_NO_STORED_ROWS"),
        _coverage_row("research_order_book_replay", "BLOCKED_NO_STORED_ROWS"),
        _coverage_row("research_microstructure_features", "BLOCKED_NO_STORED_ROWS"),
    ]


def _coverage_row(contract: str, status: str) -> dict[str, str]:
    return {
        "contract": contract,
        "coverage_status": status,
        "row_count_available": "False",
        "gap_metrics_available": "False",
        "reason": "Contracts and fixture samples exist, but no stored replay dataset exists.",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _replay_determinism_audit(source_rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    contracts_present = all(row["status"] == "PRESENT" for row in source_rows)
    status = "CONTRACT_READY_DATA_BLOCKED" if contracts_present else "BLOCKED_MISSING_ARTIFACTS"
    return [
        {
            "check": "ordering_contract",
            "status": status,
            "detail": "Stable ordering contract exists; cannot validate on absent stored rows.",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
        {
            "check": "gap_marker_contract",
            "status": status,
            "detail": "Gap marker contract exists; no sequence coverage sample exists yet.",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
        {
            "check": "replay_idempotency",
            "status": status,
            "detail": "Idempotency cannot be measured before stored replay rows exist.",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
    ]


def _data_readiness_audit() -> list[dict[str, str]]:
    return [
        {
            "readiness_item": "static_contracts",
            "status": "READY",
            "detail": "Schema, parser, and feature derivation contracts are defined.",
        },
        {
            "readiness_item": "stored_replay_dataset",
            "status": "MISSING",
            "detail": "No research order-book replay rows are captured or stored.",
        },
        {
            "readiness_item": "coverage_gap_metrics",
            "status": "BLOCKED",
            "detail": "Requires stored replay rows and explicit coverage windows.",
        },
    ]


def _blocked_decisions() -> list[dict[str, str]]:
    return [
        {
            "decision": "research_capture_scope",
            "blocker": "NO_ISOLATED_CAPTURE_PLAN_APPROVED",
            "required_action": "PLAN_OPTIONAL_ISOLATED_MICROSTRUCTURE_CAPTURE_SERVICE",
        },
        {
            "decision": "coverage_windows",
            "blocker": "COVERAGE_WINDOWS_NOT_SELECTED",
            "required_action": "SELECT_SYMBOLS_DEPTHS_AND_CAPTURE_WINDOWS",
        },
        {
            "decision": "checksum_validation",
            "blocker": "NO_STORED_SEQUENCE_OR_CHECKSUM_SERIES",
            "required_action": "VALIDATE_CHECKSUMS_AFTER_STORED_REPLAY_EXISTS",
        },
    ]


def _audit_status(source_rows: list[Mapping[str, str]]) -> str:
    if all(row["status"] == "PRESENT" for row in source_rows):
        return "MICROSTRUCTURE_REPLAY_AUDIT_COMPLETE_DATA_BLOCKED"
    return "MICROSTRUCTURE_REPLAY_AUDIT_BLOCKED_MISSING_CONTRACT_ARTIFACTS"


def _recommendation(source_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if all(row["status"] == "PRESENT" for row in source_rows):
        recommendation = "PLAN_OPTIONAL_ISOLATED_MICROSTRUCTURE_CAPTURE_SERVICE"
        next_action = "DESIGN_ISOLATED_RESEARCH_MICROSTRUCTURE_CAPTURE_PLAN"
    else:
        recommendation = "RESTORE_MICROSTRUCTURE_CONTRACT_ARTIFACTS"
        next_action = "RE_RUN_MICROSTRUCTURE_CONTRACT_ARTIFACTS"
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
        "next_actions": [
            {
                "action": next_action,
                "scope": "research_data_upgrade",
                "runtime_effect": "NO_RUNTIME_EFFECT",
                "m20_status": "M20_PAUSED",
            }
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _write_outputs(
    *,
    output_files: Mapping[str, str],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    rows_by_file: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    write_json_atomic(Path(output_files["manifest_json"]), dict(manifest))
    write_json_atomic(Path(output_files["report_json"]), dict(report))
    write_json_atomic(Path(output_files["recommendation_json"]), dict(recommendation))
    for key, rows in rows_by_file.items():
        write_csv(Path(output_files[key]), [dict(row) for row in rows])


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "microstructure_replay_audit.json"),
        "report_md": str(output_dir / "microstructure_replay_audit.md"),
        "source_artifact_audit_csv": str(output_dir / "source_artifact_audit.csv"),
        "coverage_gap_audit_csv": str(output_dir / "coverage_gap_audit.csv"),
        "replay_determinism_audit_csv": str(output_dir / "replay_determinism_audit.csv"),
        "data_readiness_audit_csv": str(output_dir / "data_readiness_audit.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Microstructure Coverage Gap Replay Audit",
        "",
        f"- Audit status: `{report['audit_status']}`",
        f"- Stored replay rows available: `{report['stored_replay_rows_available']}`",
        f"- Coverage/gap metric status: `{report['coverage_gap_metric_status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "This is audit-only. It does not capture, persist, replay live data,",
        "or change runtime ingestion behavior.",
    ]
    return "\n".join(lines) + "\n"
