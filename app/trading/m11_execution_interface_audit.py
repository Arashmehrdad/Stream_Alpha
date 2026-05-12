"""Artifact-backed M11 execution-interface audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/platform_maturity/m11/execution_interface_audit"
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
    "order_request_contract",
    "idempotency_key_builder",
    "risk_approved_order_builder",
    "pending_order_builder",
    "execution_adapter_protocol",
    "paper_execution_adapter",
    "shadow_execution_adapter",
    "live_execution_adapter",
    "order_lifecycle_contract",
    "order_request_persistence",
    "order_event_persistence",
    "runner_order_after_risk",
    "runner_created_event_persistence",
    "live_submit_gate",
    "live_symbol_whitelist",
    "live_max_notional_gate",
    "live_stale_signal_gate",
    "live_broker_failure_handling",
    "m10_risk_authority_documented",
    "m20_pause_documentation",
)


def audit_m11_execution_interface(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit M11 execution-interface surfaces and write deterministic artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    execution_rows = _execution_surface_audit(root)
    risk_boundary_rows = _risk_authority_boundary_audit(root)
    lifecycle_rows = _order_lifecycle_audit(root)
    adapter_rows = _mode_adapter_audit(root)
    gap_rows = _gap_analysis(
        execution_rows,
        risk_boundary_rows,
        lifecycle_rows,
        adapter_rows,
    )
    state = _m11_state(execution_rows)
    recommendation = _recommendation(state, gap_rows)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "m11_execution_interface_audit_v1",
        "repo_root": str(root),
        "m11_state": state,
        "critical_surface_count": len(CRITICAL_SURFACES),
        "present_critical_surface_count": sum(
            1
            for row in execution_rows
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
            "execution_surfaces": execution_rows,
            "risk_boundaries": risk_boundary_rows,
            "order_lifecycle": lifecycle_rows,
            "mode_adapters": adapter_rows,
        },
    )
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    write_csv(Path(output_files["execution_surface_audit_csv"]), execution_rows)
    write_csv(Path(output_files["risk_authority_boundary_audit_csv"]), risk_boundary_rows)
    write_csv(Path(output_files["order_lifecycle_audit_csv"]), lifecycle_rows)
    write_csv(Path(output_files["mode_adapter_audit_csv"]), adapter_rows)
    write_csv(Path(output_files["gap_analysis_csv"]), gap_rows)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            {
                "execution_surfaces": execution_rows,
                "risk_boundaries": risk_boundary_rows,
                "order_lifecycle": lifecycle_rows,
                "mode_adapters": adapter_rows,
            },
            gap_rows,
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "execution_surface_audit": execution_rows,
            "risk_authority_boundary_audit": risk_boundary_rows,
            "order_lifecycle_audit": lifecycle_rows,
            "mode_adapter_audit": adapter_rows,
            "gap_analysis": gap_rows,
            "recommendation_payload": recommendation,
        }
    )


def _execution_surface_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface_check(
            root,
            "order_request_contract",
            Path("app/trading/schemas.py"),
            "class OrderRequest",
            "Deterministic M11 order request contract exists.",
        ),
        _surface_check(
            root,
            "idempotency_key_builder",
            Path("app/trading/execution.py"),
            "def build_idempotency_key",
            "M11 can build deterministic order idempotency keys.",
        ),
        _surface_check(
            root,
            "risk_approved_order_builder",
            Path("app/trading/execution.py"),
            "def build_order_request",
            "M11 builds order requests from risk-approved decisions.",
        ),
        _surface_check(
            root,
            "pending_order_builder",
            Path("app/trading/execution.py"),
            "def build_pending_order_request",
            "M11 rebuilds pending order requests for restart recovery.",
        ),
        _surface_check(
            root,
            "execution_adapter_protocol",
            Path("app/trading/execution.py"),
            "class ExecutionAdapter",
            "Mode-specific execution adapters share one contract.",
        ),
        _surface_check(
            root,
            "paper_execution_adapter",
            Path("app/trading/execution.py"),
            "class PaperExecutionAdapter",
            "Paper execution adapter exists.",
        ),
        _surface_check(
            root,
            "shadow_execution_adapter",
            Path("app/trading/execution.py"),
            "class ShadowExecutionAdapter",
            "Shadow execution adapter exists.",
        ),
        _surface_check(
            root,
            "live_execution_adapter",
            Path("app/trading/execution.py"),
            "class LiveExecutionAdapter",
            "Guarded live execution adapter exists.",
        ),
        _surface_check(
            root,
            "order_lifecycle_contract",
            Path("app/trading/schemas.py"),
            "class OrderLifecycleEvent",
            "Explicit order lifecycle event contract exists.",
        ),
        _surface_check(
            root,
            "order_request_persistence",
            Path("app/trading/repository.py"),
            "ensure_order_request",
            "Order requests are persisted through an idempotent repository surface.",
        ),
        _surface_check(
            root,
            "order_event_persistence",
            Path("app/trading/repository.py"),
            "insert_order_event_if_absent",
            "Order lifecycle events are persisted idempotently.",
        ),
        _surface_check(
            root,
            "runner_order_after_risk",
            Path("app/trading/runner.py"),
            "build_order_request",
            "Runner creates order requests downstream of M10 risk decisions.",
        ),
        _surface_check(
            root,
            "runner_created_event_persistence",
            Path("app/trading/runner.py"),
            "build_created_event",
            "Runner persists CREATED lifecycle events.",
        ),
        _surface_check(
            root,
            "live_submit_gate",
            Path("app/trading/execution.py"),
            "resolve_live_submit_gate",
            "Live adapter checks guarded submit state before broker submission.",
        ),
        _surface_check(
            root,
            "live_symbol_whitelist",
            Path("app/trading/execution.py"),
            "LIVE_SYMBOL_NOT_WHITELISTED",
            "Live adapter enforces symbol whitelist.",
        ),
        _surface_check(
            root,
            "live_max_notional_gate",
            Path("app/trading/execution.py"),
            "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED",
            "Live adapter enforces max order notional.",
        ),
        _surface_check(
            root,
            "live_stale_signal_gate",
            Path("app/trading/execution.py"),
            "LIVE_SIGNAL_STALE",
            "Live adapter rejects stale order requests.",
        ),
        _surface_check(
            root,
            "live_broker_failure_handling",
            Path("app/trading/execution.py"),
            "LIVE_BROKER_SUBMIT_FAILED",
            "Live adapter records broker submit failures.",
        ),
        _surface_check(
            root,
            "m10_risk_authority_documented",
            Path("README.md"),
            "M10_RISK_INTERFACE_CONSOLIDATED",
            "Docs preserve M10 as upstream risk authority.",
        ),
        _m20_pause_docs_check(root),
    ]


def _risk_authority_boundary_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            "risk_decision_input_contract",
            _file_contains(root, Path("app/trading/schemas.py"), "class RiskDecision"),
            "Order request construction depends on RiskDecision.",
        ),
        _audit_row(
            "buy_requires_risk_approval_or_modification",
            _file_contains(
                root,
                Path("app/trading/execution.py"),
                'decision.outcome in {"APPROVED", "MODIFIED"}',
            ),
            "BUY order requests require risk APPROVED or MODIFIED outcomes.",
        ),
        _audit_row(
            "runner_evaluates_risk_before_order_request",
            _file_contains(
                root,
                Path("app/trading/runner.py"),
                "risk_decision = evaluate_risk",
            )
            and _file_contains(
                root,
                Path("app/trading/runner.py"),
                "order_request = build_order_request",
            ),
            "Runner evaluates M10 before creating M11 order requests.",
        ),
        _audit_row(
            "risk_decision_persisted_before_execution_truth",
            _file_contains(root, Path("app/trading/runner.py"), "insert_risk_decision")
            and _file_contains(root, Path("app/trading/runner.py"), "insert_order_event_if_absent"),
            "Runner persists risk and order lifecycle truth on the accepted path.",
        ),
        _audit_row(
            "no_m20_research_authority",
            _docs_contain(root, "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"),
            "Paused M20 artifacts have no authority over M11 execution.",
        ),
    ]


def _order_lifecycle_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            "created_state",
            _file_contains(root, Path("app/trading/schemas.py"), '"CREATED"'),
            "Lifecycle contract includes CREATED.",
        ),
        _audit_row(
            "submitted_state",
            _file_contains(root, Path("app/trading/schemas.py"), '"SUBMITTED"'),
            "Lifecycle contract includes SUBMITTED.",
        ),
        _audit_row(
            "accepted_state",
            _file_contains(root, Path("app/trading/schemas.py"), '"ACCEPTED"'),
            "Lifecycle contract includes ACCEPTED.",
        ),
        _audit_row(
            "filled_state",
            _file_contains(root, Path("app/trading/schemas.py"), '"FILLED"'),
            "Lifecycle contract includes FILLED.",
        ),
        _audit_row(
            "rejected_state",
            _file_contains(root, Path("app/trading/schemas.py"), '"REJECTED"'),
            "Lifecycle contract includes REJECTED.",
        ),
        _audit_row(
            "terminal_events_builder",
            _file_contains(
                root,
                Path("app/trading/execution.py"),
                "def _terminal_lifecycle_events",
            ),
            "Paper and shadow adapters build terminal lifecycle events.",
        ),
        _audit_row(
            "live_broker_event_builder",
            _file_contains(root, Path("app/trading/execution.py"), "def _build_live_broker_event"),
            "Live adapter maps broker responses to lifecycle events.",
        ),
    ]


def _mode_adapter_audit(root: Path) -> list[dict[str, str]]:
    return [
        _audit_row(
            "adapter_factory",
            _file_contains(root, Path("app/trading/execution.py"), "def build_execution_adapter"),
            "Execution mode selects a concrete adapter through one factory.",
        ),
        _audit_row(
            "paper_mode_supported",
            _file_contains(root, Path("app/trading/execution.py"), 'if mode == "paper"'),
            "Paper mode is routed to PaperExecutionAdapter.",
        ),
        _audit_row(
            "shadow_mode_supported",
            _file_contains(root, Path("app/trading/execution.py"), 'if mode == "shadow"'),
            "Shadow mode is routed to ShadowExecutionAdapter.",
        ),
        _audit_row(
            "live_mode_supported",
            _file_contains(root, Path("app/trading/execution.py"), 'if mode == "live"'),
            "Live mode is routed to LiveExecutionAdapter.",
        ),
        _audit_row(
            "broker_client_protocol",
            _file_contains(root, Path("app/trading/execution.py"), "class BrokerClient"),
            "Live broker integration depends on a minimal broker protocol.",
        ),
    ]


def _surface_check(
    root: Path,
    name: str,
    relative_path: Path,
    needle: str,
    detail: str,
) -> dict[str, str]:
    return {
        "surface_name": name,
        "path": str(relative_path),
        "status": "PRESENT" if _file_contains(root, relative_path, needle) else "MISSING",
        "required_for_m11": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _audit_row(name: str, present: bool, detail: str) -> dict[str, str]:
    return {
        "audit_name": name,
        "status": "PRESENT" if present else "MISSING",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": detail,
    }


def _m20_pause_docs_check(root: Path) -> dict[str, str]:
    return {
        "surface_name": "m20_pause_documentation",
        "path": "README.md|docs/training.md|PLANS.md",
        "status": (
            "PRESENT"
            if _docs_contain(root, "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY")
            else "MISSING"
        ),
        "required_for_m11": "True",
        "runtime_authority_changed": "False",
        "m20_reopened": "False",
        "detail": "Docs preserve M20 as a paused negative research result.",
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
    execution_rows: list[Mapping[str, str]],
    risk_boundary_rows: list[Mapping[str, str]],
    lifecycle_rows: list[Mapping[str, str]],
    adapter_rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in execution_rows:
        if row["required_for_m11"] == "True" and row["status"] != "PRESENT":
            rows.append(
                _gap_row(
                    row["surface_name"],
                    "IMPLEMENT_REUSABLE_M11_EXECUTION_INTERFACE_GAP_FILL",
                    row["detail"],
                )
            )
    for row_set, action in (
        (risk_boundary_rows, "RESTORE_M10_TO_M11_AUTHORITY_BOUNDARY"),
        (lifecycle_rows, "RESTORE_M11_ORDER_LIFECYCLE_CONTRACT"),
        (adapter_rows, "RESTORE_M11_MODE_ADAPTER_CONTRACT"),
    ):
        for row in row_set:
            if row["status"] != "PRESENT":
                rows.append(_gap_row(row["audit_name"], action, row["detail"]))
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


def _m11_state(execution_rows: list[Mapping[str, str]]) -> str:
    present = {
        row["surface_name"]
        for row in execution_rows
        if row["surface_name"] in CRITICAL_SURFACES and row["status"] == "PRESENT"
    }
    if present == set(CRITICAL_SURFACES):
        return "M11_EXECUTION_INTERFACE_CONSOLIDATED"
    if present:
        return "M11_EXECUTION_INTERFACE_PARTIAL"
    return "M11_EXECUTION_INTERFACE_BLOCKED"


def _recommendation(state: str, gap_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    if state == "M11_EXECUTION_INTERFACE_CONSOLIDATED" and not gap_rows:
        recommendation = "PROCEED_TO_M12_GUARDED_LIVE_AUDIT"
        next_required_action = "AUDIT_M12_GUARDED_LIVE_CONTROLS"
    elif gap_rows:
        recommendation = "IMPLEMENT_REUSABLE_M11_EXECUTION_INTERFACE_GAP_FILLS"
        next_required_action = "FILL_M11_EXECUTION_INTERFACE_GAPS"
    else:
        recommendation = "RESTORE_M11_EXECUTION_INTERFACE_PREREQUISITES"
        next_required_action = "RESTORE_M11_EXECUTION_INTERFACE_PREREQUISITES"
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
        for row in audit_rows["execution_surfaces"]
        if row.get("path") and "|" not in row["path"]
    }
    return {
        "schema_version": "m11_execution_interface_audit_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "source_paths": sorted(source_paths),
        "execution_surface_count": len(audit_rows["execution_surfaces"]),
        "risk_authority_boundary_count": len(audit_rows["risk_boundaries"]),
        "order_lifecycle_count": len(audit_rows["order_lifecycle"]),
        "mode_adapter_count": len(audit_rows["mode_adapters"]),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": dict(output_files),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "m11_execution_interface_audit.json"),
        "report_md": str(output_dir / "m11_execution_interface_audit.md"),
        "execution_surface_audit_csv": str(output_dir / "execution_surface_audit.csv"),
        "risk_authority_boundary_audit_csv": str(
            output_dir / "risk_authority_boundary_audit.csv"
        ),
        "order_lifecycle_audit_csv": str(output_dir / "order_lifecycle_audit.csv"),
        "mode_adapter_audit_csv": str(output_dir / "mode_adapter_audit.csv"),
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
        _section("Execution Surfaces", "surface_name", audit_rows["execution_surfaces"]),
        _section("Risk Authority Boundaries", "audit_name", audit_rows["risk_boundaries"]),
        _section("Order Lifecycle", "audit_name", audit_rows["order_lifecycle"]),
        _section("Mode Adapters", "audit_name", audit_rows["mode_adapters"]),
    ]
    lines = [
        "# M11 Execution Interface Audit",
        "",
        f"- M11 state: `{report['m11_state']}`",
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
