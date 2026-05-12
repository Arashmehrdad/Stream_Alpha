"""Research-only data-upgrade feasibility plan artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/feasibility_plan"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "PLANNING_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_research_data_upgrade_plan(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a planning-only research data-upgrade feasibility artifact set."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    m20_summary = _read_optional_json(
        root
        / "artifacts/training/m20/20260506T054337Z/research_labels/vol_scaled"
        / "m20_final_research_summary/m20_final_research_summary.json"
    )
    m22_closeout = _read_optional_json(
        root
        / "artifacts/platform_maturity/m22/platform_maturity_closeout"
        / "m22_platform_maturity_closeout.json"
    )
    data_families = _required_data_families()
    feasibility_rows = _source_feasibility(root, data_families)
    leakage_rows = _leakage_boundary_audit(data_families)
    batches = _implementation_batches()
    blocked_routes = _blocked_routes(feasibility_rows)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "research_data_upgrade_plan_v1",
        "repo_root": str(root),
        "m20_research_decision": m20_summary.get("final_m20_decision", M20_DECISION),
        "platform_maturity_state": m22_closeout.get("platform_maturity_state", ""),
        "required_data_family_count": len(data_families),
        "blocked_route_count": len(blocked_routes),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "research_data_upgrade_plan_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_artifacts": [
            "artifacts/training/m20/20260506T054337Z/research_labels/vol_scaled/"
            "m20_final_research_summary",
            "artifacts/platform_maturity/m22/platform_maturity_closeout",
        ],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_atomic(Path(output_files["manifest_json"]), manifest)
    write_json_atomic(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, data_families, blocked_routes, batches),
        encoding="utf-8",
    )
    write_csv(Path(output_files["required_data_families_csv"]), data_families)
    write_csv(Path(output_files["source_feasibility_csv"]), feasibility_rows)
    write_csv(Path(output_files["leakage_boundary_audit_csv"]), leakage_rows)
    write_csv(Path(output_files["implementation_batches_csv"]), batches)
    write_csv(Path(output_files["blocked_routes_csv"]), blocked_routes)
    write_csv(Path(output_files["next_actions_csv"]), recommendation["next_actions"])
    write_json_atomic(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "required_data_families": data_families,
            "source_feasibility": feasibility_rows,
            "leakage_and_runtime_boundary_audit": leakage_rows,
            "implementation_batches": batches,
            "blocked_routes": blocked_routes,
            "recommendation_payload": recommendation,
        }
    )


def _required_data_families() -> list[dict[str, str]]:
    return [
        _family("order_book_depth", "depth snapshots", "microstructure_research"),
        _family("spread_liquidity", "bid/ask spread and liquidity", "execution_quality"),
        _family("trade_flow_imbalance", "signed trade-flow imbalance", "microstructure_research"),
        _family(
            "same_venue_execution_quality",
            "paper/shadow fill-quality measurements",
            "execution_quality",
        ),
        _family(
            "untouched_evaluation_segments",
            "strict chronological holdout segments",
            "validation_design",
        ),
        _family(
            "lower_turnover_event_labels",
            "event-sampled lower-turnover labels",
            "label_design",
        ),
        _family("storage_replay", "deterministic storage and replay contract", "platform"),
    ]


def _family(name: str, description: str, category: str) -> dict[str, str]:
    return {
        "data_family": name,
        "description": description,
        "category": category,
        "required_before_reopening_alpha_research": "True",
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "implementation_status": "PLANNED_ONLY",
    }


def _source_feasibility(
    root: Path,
    families: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    has_existing_ohlc = (root / "app/ingestion").is_dir() and (root / "app/features").is_dir()
    rows = []
    for family in families:
        name = family["data_family"]
        existing_source = (
            "existing_ohlcv_pipeline"
            if name == "storage_replay" and has_existing_ohlc
            else ""
        )
        rows.append(
            {
                "data_family": name,
                "existing_source_found": str(bool(existing_source)),
                "candidate_source": existing_source or "TO_BE_SELECTED",
                "credential_or_vendor_decision_required": str(not bool(existing_source)),
                "safe_to_implement_now": "False",
                "reason": (
                    "Requires separate ingestion/storage design before implementation."
                    if not existing_source
                    else "Existing pipeline can inform replay design but not microstructure data."
                ),
            }
        )
    return rows


def _leakage_boundary_audit(families: list[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "data_family": family["data_family"],
            "feature_rule": "features_use_observation_time_or_past_only",
            "label_rule": "labels_may_use_future_path_only_as_research_outcomes",
            "runtime_boundary": "NO_RUNTIME_EFFECT",
            "training_boundary": "NO_TRAINING_IN_THIS_PLAN",
            "profit_claim": "NO_PROFIT_CLAIM",
        }
        for family in families
    ]


def _implementation_batches() -> list[dict[str, str]]:
    return [
        _batch("DU1", "choose data source and schema", "DESIGN_ONLY"),
        _batch("DU2", "build research-only raw capture and replay contract", "NOT_STARTED"),
        _batch("DU3", "derive leakage-safe microstructure features", "NOT_STARTED"),
        _batch("DU4", "build lower-turnover research labels", "NOT_STARTED"),
        _batch(
            "DU5",
            "rerun generic research evaluators on upgraded inputs",
            "BLOCKED_UNTIL_DU2_DU4",
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


def _blocked_routes(feasibility_rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "blocked_route": f"implement_{row['data_family']}",
            "blocker": "DATA_SOURCE_DECISION_REQUIRED",
            "required_action": "DESIGN_MARKET_MICROSTRUCTURE_RESEARCH_INGESTION_PLAN",
        }
        for row in feasibility_rows
        if row["credential_or_vendor_decision_required"] == "True"
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "PLAN_DATA_UPGRADE_IMPLEMENTATION_BATCHES",
        "next_required_action": "DESIGN_MARKET_MICROSTRUCTURE_RESEARCH_INGESTION_PLAN",
        "next_actions": [
            {
                "action": "DESIGN_MARKET_MICROSTRUCTURE_RESEARCH_INGESTION_PLAN",
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


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "research_data_upgrade_plan.json"),
        "report_md": str(output_dir / "research_data_upgrade_plan.md"),
        "required_data_families_csv": str(output_dir / "required_data_families.csv"),
        "source_feasibility_csv": str(output_dir / "source_feasibility.csv"),
        "leakage_boundary_audit_csv": str(
            output_dir / "leakage_and_runtime_boundary_audit.csv"
        ),
        "implementation_batches_csv": str(output_dir / "implementation_batches.csv"),
        "blocked_routes_csv": str(output_dir / "blocked_routes.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    families: list[Mapping[str, str]],
    blocked_routes: list[Mapping[str, str]],
    batches: list[Mapping[str, str]],
) -> str:
    lines = [
        "# Research Data-Upgrade Feasibility Plan",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Required Data Families",
    ]
    lines.extend(f"- `{row['data_family']}`: {row['description']}" for row in families)
    lines.append("")
    lines.append("## Blocked Routes")
    lines.extend(f"- `{row['blocked_route']}`: `{row['blocker']}`" for row in blocked_routes)
    lines.append("")
    lines.append("## Future Batches")
    lines.extend(f"- `{row['batch_id']}`: {row['goal']}" for row in batches)
    return "\n".join(lines) + "\n"
