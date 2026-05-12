"""Artifact-backed M13 reliability and recovery control audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m13/reliability_recovery_audit"
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
    "reliability_config_file",
    "reliability_config_contract",
    "freshness_status_contract",
    "service_heartbeat_contract",
    "reliability_state_contract",
    "recovery_event_contract",
    "feature_lag_contract",
    "system_reliability_contract",
    "feed_freshness_evaluator",
    "feature_freshness_evaluator",
    "heartbeat_freshness_evaluator",
    "regime_freshness_evaluator",
    "feature_consumer_lag_evaluator",
    "system_reliability_aggregator",
    "circuit_breaker_transition",
    "pending_signal_expiry",
    "reliability_store",
    "heartbeat_persistence",
    "reliability_state_persistence",
    "recovery_event_persistence",
    "feature_lag_persistence",
    "system_reliability_persistence",
    "health_artifact_writer",
    "recovery_jsonl_writer",
    "freshness_endpoint",
    "system_reliability_endpoint",
    "trading_runner_recovery_events",
    "live_health_gate_integration",
    "m12_guarded_live_documented",
    "m20_pause_documentation",
)


def audit_m13_reliability_recovery(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M13 reliability/recovery controls and write deterministic artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    surface_rows = _reliability_surface_audit(root)
    freshness_rows = _freshness_recovery_audit(root)
    persistence_rows = _persistence_audit(root)
    runtime_rows = _runtime_integration_audit(root)
    artifact_rows = _artifact_status_audit(root)
    gap_rows = _gap_analysis(
        surface_rows,
        freshness_rows,
        persistence_rows,
        runtime_rows,
        artifact_rows,
    )
    state = _m13_state(surface_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m13_reliability_recovery_audit_v1",
        "repo_root": str(root),
        "m13_state": state,
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
            "reliability_surfaces": surface_rows,
            "freshness_recovery": freshness_rows,
            "persistence": persistence_rows,
            "runtime_integration": runtime_rows,
            "artifact_status": artifact_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["reliability_surface_audit_csv"]), surface_rows)
    write_csv(Path(output_files["freshness_recovery_audit_csv"]), freshness_rows)
    write_csv(Path(output_files["persistence_audit_csv"]), persistence_rows)
    write_csv(Path(output_files["runtime_integration_audit_csv"]), runtime_rows)
    write_csv(Path(output_files["artifact_status_audit_csv"]), artifact_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "reliability_surfaces": surface_rows,
                "freshness_recovery": freshness_rows,
                "persistence": persistence_rows,
                "runtime_integration": runtime_rows,
                "artifact_status": artifact_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "reliability_surface_audit": surface_rows,
            "freshness_recovery_audit": freshness_rows,
            "persistence_audit": persistence_rows,
            "runtime_integration_audit": runtime_rows,
            "artifact_status_audit": artifact_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _reliability_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(root, "reliability_config_file", "configs/reliability.yaml", "m13"),
        _surface_check(
            root,
            "reliability_config_contract",
            "app/reliability/config.py",
            "class ReliabilityConfig",
        ),
        _surface_check(
            root,
            "freshness_status_contract",
            "app/reliability/schemas.py",
            "class FreshnessStatus",
        ),
        _surface_check(
            root,
            "service_heartbeat_contract",
            "app/reliability/schemas.py",
            "class ServiceHeartbeat",
        ),
        _surface_check(
            root,
            "reliability_state_contract",
            "app/reliability/schemas.py",
            "class ReliabilityState",
        ),
        _surface_check(
            root,
            "recovery_event_contract",
            "app/reliability/schemas.py",
            "class RecoveryEvent",
        ),
        _surface_check(
            root,
            "feature_lag_contract",
            "app/reliability/schemas.py",
            "class FeatureLagSnapshot",
        ),
        _surface_check(
            root,
            "system_reliability_contract",
            "app/reliability/schemas.py",
            "class SystemReliabilitySnapshot",
        ),
        _surface_check(
            root,
            "feed_freshness_evaluator",
            "app/reliability/service.py",
            "def evaluate_feed_freshness",
        ),
        _surface_check(
            root,
            "feature_freshness_evaluator",
            "app/reliability/service.py",
            "def evaluate_feature_freshness",
        ),
        _surface_check(
            root,
            "heartbeat_freshness_evaluator",
            "app/reliability/service.py",
            "def evaluate_heartbeat_freshness",
        ),
        _surface_check(
            root,
            "regime_freshness_evaluator",
            "app/reliability/service.py",
            "def evaluate_regime_freshness",
        ),
        _surface_check(
            root,
            "feature_consumer_lag_evaluator",
            "app/reliability/service.py",
            "def evaluate_feature_consumer_lag",
        ),
        _surface_check(
            root,
            "system_reliability_aggregator",
            "app/reliability/service.py",
            "def aggregate_system_reliability",
        ),
        _surface_check(
            root,
            "circuit_breaker_transition",
            "app/reliability/service.py",
            "def transition_circuit_breaker",
        ),
        _surface_check(
            root,
            "pending_signal_expiry",
            "app/reliability/service.py",
            "def evaluate_pending_signal_expiry",
        ),
        _surface_check(
            root,
            "reliability_store",
            "app/reliability/store.py",
            "class ReliabilityStore",
        ),
        _surface_check(
            root,
            "heartbeat_persistence",
            "app/reliability/store.py",
            "save_service_heartbeat",
        ),
        _surface_check(
            root,
            "reliability_state_persistence",
            "app/reliability/store.py",
            "save_reliability_state",
        ),
        _surface_check(
            root,
            "recovery_event_persistence",
            "app/reliability/store.py",
            "insert_reliability_event",
        ),
        _surface_check(
            root,
            "feature_lag_persistence",
            "app/reliability/store.py",
            "save_feature_lag_state",
        ),
        _surface_check(
            root,
            "system_reliability_persistence",
            "app/reliability/store.py",
            "save_system_reliability_state",
        ),
        _surface_check(
            root,
            "health_artifact_writer",
            "app/reliability/artifacts.py",
            "def write_json_artifact",
        ),
        _surface_check(
            root,
            "recovery_jsonl_writer",
            "app/reliability/artifacts.py",
            "def append_jsonl_artifact",
        ),
        _surface_check(root, "freshness_endpoint", "app/inference/main.py", '"/freshness"'),
        _surface_check(
            root,
            "system_reliability_endpoint",
            "app/inference/main.py",
            '"/reliability/system"',
        ),
        _surface_check(
            root,
            "trading_runner_recovery_events",
            "app/trading/runner.py",
            "RecoveryEvent",
        ),
        _surface_check(
            root,
            "live_health_gate_integration",
            "app/trading/live.py",
            "def apply_live_health_gate",
        ),
        _docs_surface_check(
            root,
            "m12_guarded_live_documented",
            "M12_GUARDED_LIVE_CONTROLS_CONSOLIDATED",
            "Docs preserve M12 as the upstream guarded-live control layer.",
        ),
        _docs_surface_check(
            root,
            "m20_pause_documentation",
            "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
            "Docs preserve M20 as paused and non-authoritative.",
        ),
    ]


def _freshness_recovery_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(root, "feed_stale_reason", "app/reliability/service.py", "FEED_STALE"),
        _audit_row(root, "feature_stale_reason", "app/reliability/service.py", "FEATURE_STALE"),
        _audit_row(
            root,
            "heartbeat_missing_reason",
            "app/reliability/service.py",
            "HEARTBEAT_MISSING",
        ),
        _audit_row(
            root,
            "regime_row_incompatible_reason",
            "app/reliability/service.py",
            "REGIME_ROW_INCOMPATIBLE",
        ),
        _audit_row(
            root,
            "feature_lag_breach_reason",
            "app/reliability/service.py",
            "FEATURE_LAG_BREACH",
        ),
        _audit_row(root, "breaker_opened_reason", "app/reliability/service.py", "BREAKER_OPENED"),
        _audit_row(
            root,
            "breaker_half_opened_reason",
            "app/reliability/service.py",
            "BREAKER_HALF_OPENED",
        ),
        _audit_row(
            root,
            "breaker_restored_reason",
            "app/reliability/service.py",
            "BREAKER_RESTORED",
        ),
        _audit_row(
            root,
            "stale_pending_signal_recovery_reason",
            "app/reliability/service.py",
            "RECOVERY_STALE_PENDING_SIGNAL_CLEARED",
        ),
    ]


def _persistence_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "service_heartbeats_table",
            "app/reliability/store.py",
            "service_heartbeats",
        ),
        _audit_row(
            root,
            "reliability_state_table",
            "app/reliability/store.py",
            "reliability_state",
        ),
        _audit_row(
            root,
            "reliability_events_table",
            "app/reliability/store.py",
            "reliability_events",
        ),
        _audit_row(root, "feature_lag_table", "app/reliability/store.py", "reliability_lag_state"),
        _audit_row(
            root,
            "system_state_table",
            "app/reliability/store.py",
            "reliability_system_state",
        ),
        _audit_row(
            root,
            "trading_repository_reliability_round_trip",
            "app/trading/repository.py",
            "load_latest_service_heartbeat",
        ),
    ]


def _runtime_integration_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "inference_system_reliability_snapshot",
            "app/inference/service.py",
            "async def system_reliability_snapshot",
        ),
        _audit_row(
            root,
            "inference_writes_system_reliability_artifact",
            "app/inference/service.py",
            "_write_system_reliability_artifact",
        ),
        _audit_row(
            root,
            "inference_writes_freshness_artifact",
            "app/inference/service.py",
            "_write_freshness_artifact",
        ),
        _audit_row(
            root,
            "signal_client_fetches_system_reliability",
            "app/trading/signal_client.py",
            "fetch_system_reliability",
        ),
        _audit_row(
            root,
            "runner_skips_open_breaker",
            "app/trading/runner.py",
            "SIGNAL_FETCH_SKIPPED_BREAKER_OPEN",
        ),
        _audit_row(
            root,
            "runner_expires_stale_pending_signals",
            "app/trading/runner.py",
            "_expire_stale_pending_signals",
        ),
        _audit_row(
            root,
            "runner_writes_heartbeat",
            "app/trading/runner.py",
            "_write_runner_heartbeat",
        ),
        _audit_row(
            root,
            "runner_records_reliability_event",
            "app/trading/runner.py",
            "_record_reliability_event",
        ),
    ]


def _artifact_status_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            root,
            "health_snapshot_path",
            "configs/reliability.yaml",
            "health_snapshot_path",
        ),
        _audit_row(
            root,
            "freshness_summary_path",
            "configs/reliability.yaml",
            "freshness_summary_path",
        ),
        _audit_row(
            root,
            "recovery_events_path",
            "configs/reliability.yaml",
            "recovery_events_path",
        ),
        _audit_row(root, "system_health_path", "configs/reliability.yaml", "system_health_path"),
        _audit_row(root, "lag_summary_path", "configs/reliability.yaml", "lag_summary_path"),
    ]


def _surface_check(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    relative_path = Path(path_value)
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING",
        "required_for_m13": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": f"M13 reliability surface `{name}` is present.",
    }


def _docs_surface_check(root: Path, name: str, needle: str, detail: str) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": "README.md|docs/training.md|PLANS.md",
        "status": "PRESENT" if _docs_contain(root, needle) else "MISSING",
        "required_for_m13": "True",
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
        "detail": f"M13 reliability audit `{name}` checks `{needle}`.",
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
    surface_rows: list[Mapping[str, str]],
    freshness_rows: list[Mapping[str, str]],
    persistence_rows: list[Mapping[str, str]],
    runtime_rows: list[Mapping[str, str]],
    artifact_rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in surface_rows:
        if row["required_for_m13"] == "True" and row["status"] != "PRESENT":
            rows.append(_gap_row(row["surface_name"], "RESTORE_M13_RELIABILITY_SURFACE"))
    for row_set, action in (
        (freshness_rows, "RESTORE_M13_FRESHNESS_OR_RECOVERY_CONTROL"),
        (persistence_rows, "RESTORE_M13_RELIABILITY_PERSISTENCE"),
        (runtime_rows, "RESTORE_M13_RUNTIME_INTEGRATION_SURFACE"),
        (artifact_rows, "RESTORE_M13_OPERATOR_ARTIFACT_PATH"),
    ):
        for row in row_set:
            if row["status"] != "PRESENT":
                rows.append(_gap_row(row["audit_name"], action))
    return rows


def _gap_row(name: str, action: str) -> dict[str, str]:
    return {
        "gap_name": name,
        "gap_status": "BLOCKING",
        "recommended_action": action,
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": "M13 audit-only finding; no runtime behavior changed.",
    }


def _m13_state(surface_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in surface_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M13_RELIABILITY_RECOVERY_CONTROLS_CONSOLIDATED"
    if present:
        return "M13_RELIABILITY_RECOVERY_CONTROLS_PARTIAL"
    return "M13_RELIABILITY_RECOVERY_CONTROLS_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M13_RELIABILITY_RECOVERY_CONTROLS_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M14_EXPLAINABILITY_AUDIT"
        next_required_action = "AUDIT_M14_EXPLAINABILITY_AND_DECISION_TRACE_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M13_RELIABILITY_RECOVERY_GAP_FILLS"
        next_required_action = "FILL_M13_RELIABILITY_RECOVERY_GAPS"
    else:
        recommendation = "RESTORE_M13_RELIABILITY_RECOVERY_PREREQUISITES"
        next_required_action = "RESTORE_M13_RELIABILITY_RECOVERY_PREREQUISITES"
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
        "schema_version": "m13_reliability_recovery_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "reliability_surface_count": len(audit_rows["reliability_surfaces"]),
        "freshness_recovery_count": len(audit_rows["freshness_recovery"]),
        "persistence_count": len(audit_rows["persistence"]),
        "runtime_integration_count": len(audit_rows["runtime_integration"]),
        "artifact_status_count": len(audit_rows["artifact_status"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m13_reliability_recovery_audit.json"),
        "report_md": str(output_dir / "m13_reliability_recovery_audit.md"),
        "reliability_surface_audit_csv": str(output_dir / "reliability_surface_audit.csv"),
        "freshness_recovery_audit_csv": str(output_dir / "freshness_recovery_audit.csv"),
        "persistence_audit_csv": str(output_dir / "persistence_audit.csv"),
        "runtime_integration_audit_csv": str(output_dir / "runtime_integration_audit.csv"),
        "artifact_status_audit_csv": str(output_dir / "artifact_status_audit.csv"),
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
        _section("Reliability Surfaces", "surface_name", audit_rows["reliability_surfaces"]),
        _section("Freshness And Recovery", "audit_name", audit_rows["freshness_recovery"]),
        _section("Persistence", "audit_name", audit_rows["persistence"]),
        _section("Runtime Integration", "audit_name", audit_rows["runtime_integration"]),
        _section("Operator Artifacts", "audit_name", audit_rows["artifact_status"]),
    ]
    lines = [
        "# M13 Reliability And Recovery Audit",
        "",
        f"- M13 state: `{report['m13_state']}`",
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


def _section(
    title: str,
    name_key: str,
    rows: list[Mapping[str, str]],
) -> str:
    lines = [f"## {title}"]
    for row in rows:
        lines.append(f"- `{row[name_key]}`: `{row['status']}`")
    lines.append("")
    return "\n".join(lines)
