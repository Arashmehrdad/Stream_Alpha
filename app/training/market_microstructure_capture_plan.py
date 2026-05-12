"""Planning-only isolated microstructure capture service artifact writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_capture_plan"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "PLANNING_ONLY",
    "NO_CAPTURE_IMPLEMENTED",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_microstructure_capture_plan(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a planning-only artifact for isolated research microstructure capture."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = _plan_rows()
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_capture_plan_v1",
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "capture_plan_status": "ISOLATED_RESEARCH_CAPTURE_PLAN_DEFINED",
        "capture_implemented": False,
        "runtime_wiring_changed": False,
        "service_contract_count": len(rows["service_contract_csv"]),
        "implementation_batch_count": len(rows["implementation_batches_csv"]),
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
        "schema_version": "microstructure_capture_plan_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_audit": str(
            root / "artifacts/research_data_upgrade/microstructure_replay_audit"
        ),
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
            "service_contract": rows["service_contract_csv"],
            "isolation_boundary": rows["isolation_boundary_csv"],
            "storage_plan": rows["storage_plan_csv"],
            "implementation_batches": rows["implementation_batches_csv"],
            "blocked_decisions": rows["blocked_decisions_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _plan_rows() -> dict[str, list[dict[str, str]]]:
    return {
        "service_contract_csv": _service_contract(),
        "isolation_boundary_csv": _isolation_boundary(),
        "storage_plan_csv": _storage_plan(),
        "operator_runbook_csv": _operator_runbook(),
        "safety_gate_plan_csv": _safety_gate_plan(),
        "implementation_batches_csv": _implementation_batches(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _service_contract() -> list[dict[str, str]]:
    return [
        _service_row(
            "source",
            "Kraken public WebSocket v2 book channel",
            "research_only",
        ),
        _service_row(
            "symbols",
            "explicit allowlist copied from research config, not runtime trading state",
            "research_only",
        ),
        _service_row(
            "depth",
            "explicit depth level selected before implementation",
            "research_only",
        ),
        _service_row(
            "output",
            "research_raw_order_book contract only",
            "research_only",
        ),
        _service_row(
            "shutdown",
            "bounded capture windows with operator stop condition",
            "research_only",
        ),
    ]


def _service_row(component: str, contract: str, authority: str) -> dict[str, str]:
    return {
        "component": component,
        "contract": contract,
        "authority": authority,
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _isolation_boundary() -> list[dict[str, str]]:
    return [
        _boundary("process", "separate optional research process, not producer service"),
        _boundary("tables", "writes only future research microstructure tables"),
        _boundary("topics", "does not publish to existing raw trade or OHLC topics"),
        _boundary("runtime", "not imported by inference, trading, risk, or execution paths"),
        _boundary("m20", "M20 remains paused and non-authoritative"),
    ]


def _boundary(name: str, rule: str) -> dict[str, str]:
    return {
        "boundary": name,
        "rule": rule,
        "status": "PLANNED_ONLY",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _storage_plan() -> list[dict[str, str]]:
    return [
        _storage("research_raw_order_book", "raw book snapshots and updates"),
        _storage("research_order_book_replay", "deterministic replay states"),
        _storage("research_microstructure_features", "derived causal features"),
        _storage("research_capture_health", "research capture health and gaps"),
    ]


def _storage(table_name: str, purpose: str) -> dict[str, str]:
    return {
        "table_name": table_name,
        "purpose": purpose,
        "creation_status": "PLANNED_NOT_CREATED",
        "mutates_existing_tables": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _operator_runbook() -> list[dict[str, str]]:
    return [
        _runbook("preflight", "validate symbols, depth, output tables, and disk budget"),
        _runbook("start", "start bounded research capture window only after approval"),
        _runbook("monitor", "watch event rate, gap markers, reconnects, and disk growth"),
        _runbook("stop", "stop at configured window or operator interrupt"),
        _runbook("post_run", "run coverage, replay determinism, and checksum audits"),
    ]


def _runbook(step: str, action: str) -> dict[str, str]:
    return {
        "step": step,
        "action": action,
        "implementation_status": "PLANNED_ONLY",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _safety_gate_plan() -> list[dict[str, str]]:
    return [
        _gate("explicit_approval", "capture implementation requires a separate approval"),
        _gate("bounded_window", "capture must require start/end or max duration"),
        _gate("resource_limit", "disk and row-rate limits must be configured"),
        _gate("no_runtime_imports", "runtime/trading modules must not import capture code"),
        _gate("no_claims", "captured data creates no trading/profit/promotion claim"),
    ]


def _gate(name: str, rule: str) -> dict[str, str]:
    return {
        "gate": name,
        "rule": rule,
        "required_before_implementation": "True",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _implementation_batches() -> list[dict[str, str]]:
    return [
        _batch("DU7", "create isolated research schema migration plan", "NEXT"),
        _batch("DU8", "implement optional fixture-tested capture service", "REQUIRES_APPROVAL"),
        _batch(
            "DU9",
            "run bounded local capture smoke with operator approval",
            "REQUIRES_APPROVAL",
        ),
        _batch(
            "DU10",
            "run coverage and replay determinism audit on captured data",
            "BLOCKED_UNTIL_DU9",
        ),
    ]


def _batch(identifier: str, goal: str, status: str) -> dict[str, str]:
    return {
        "batch_id": identifier,
        "goal": goal,
        "status": status,
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "requires_separate_approval": "True",
    }


def _blocked_decisions() -> list[dict[str, str]]:
    return [
        {
            "decision": "capture_implementation",
            "blocker": "SEPARATE_APPROVAL_REQUIRED",
            "required_action": "APPROVE_OR_REJECT_ISOLATED_RESEARCH_CAPTURE_IMPLEMENTATION",
        },
        {
            "decision": "capture_symbols",
            "blocker": "SYMBOL_ALLOWLIST_NOT_SELECTED",
            "required_action": "SELECT_RESEARCH_CAPTURE_SYMBOLS",
        },
        {
            "decision": "capture_depth",
            "blocker": "DEPTH_LEVEL_NOT_SELECTED",
            "required_action": "SELECT_RESEARCH_CAPTURE_DEPTH",
        },
        {
            "decision": "capture_window",
            "blocker": "BOUNDED_CAPTURE_WINDOW_NOT_SELECTED",
            "required_action": "SELECT_CAPTURE_DURATION_AND_STOP_CONDITIONS",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "REQUIRE_APPROVAL_BEFORE_RESEARCH_CAPTURE_IMPLEMENTATION",
        "next_required_action": "APPROVE_OR_PAUSE_ISOLATED_MICROSTRUCTURE_CAPTURE",
        "next_actions": [
            {
                "action": "APPROVE_OR_PAUSE_ISOLATED_MICROSTRUCTURE_CAPTURE",
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
        "report_json": str(output_dir / "microstructure_capture_plan.json"),
        "report_md": str(output_dir / "microstructure_capture_plan.md"),
        "service_contract_csv": str(output_dir / "service_contract.csv"),
        "isolation_boundary_csv": str(output_dir / "isolation_boundary.csv"),
        "storage_plan_csv": str(output_dir / "storage_plan.csv"),
        "operator_runbook_csv": str(output_dir / "operator_runbook.csv"),
        "safety_gate_plan_csv": str(output_dir / "safety_gate_plan.csv"),
        "implementation_batches_csv": str(output_dir / "implementation_batches.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Isolated Research Microstructure Capture Plan",
        "",
        f"- Capture plan status: `{report['capture_plan_status']}`",
        f"- Capture implemented: `{report['capture_implemented']}`",
        f"- Runtime wiring changed: `{report['runtime_wiring_changed']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "This is a planning artifact only. It does not implement capture,",
        "create schemas, or change runtime ingestion.",
    ]
    return "\n".join(lines) + "\n"
