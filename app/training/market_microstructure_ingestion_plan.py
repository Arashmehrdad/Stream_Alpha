"""Planning-only market microstructure research ingestion artifact writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/market_microstructure_ingestion_plan"
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


def write_market_microstructure_ingestion_plan(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a planning-only market microstructure ingestion artifact set."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = _plan_rows(root)
    recommendation = _recommendation(rows["existing_ingestion_audit_csv"])
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "market_microstructure_ingestion_plan_v1",
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "existing_ingestion_surface_count": len(rows["existing_ingestion_audit_csv"]),
        "present_ingestion_surface_count": sum(
            1
            for row in rows["existing_ingestion_audit_csv"]
            if row["status"] == "PRESENT"
        ),
        "proposed_source": "kraken_public_websocket_v2",
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
        "schema_version": "market_microstructure_ingestion_plan_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "source_paths": [row["path"] for row in rows["existing_ingestion_audit_csv"]],
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
    Path(output_files["report_md"]).write_text(
        _markdown(
            report,
            rows["proposed_source_contract_csv"],
            rows["proposed_storage_contract_csv"],
            rows["implementation_batches_csv"],
            rows["blocked_decisions_csv"],
        ),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "existing_ingestion_audit": rows["existing_ingestion_audit_csv"],
            "proposed_source_contract": rows["proposed_source_contract_csv"],
            "proposed_storage_contract": rows["proposed_storage_contract_csv"],
            "replay_contract": rows["replay_contract_csv"],
            "feature_derivation_plan": rows["feature_derivation_plan_csv"],
            "leakage_boundary_audit": rows["leakage_boundary_audit_csv"],
            "implementation_batches": rows["implementation_batches_csv"],
            "blocked_decisions": rows["blocked_decisions_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _plan_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "existing_ingestion_audit_csv": _existing_ingestion_audit(root),
        "proposed_source_contract_csv": _proposed_source_contract(),
        "proposed_storage_contract_csv": _proposed_storage_contract(),
        "replay_contract_csv": _replay_contract(),
        "feature_derivation_plan_csv": _feature_derivation_plan(),
        "leakage_boundary_audit_csv": _leakage_boundary_audit(),
        "implementation_batches_csv": _implementation_batches(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _existing_ingestion_audit(root: Path) -> list[dict[str, str]]:
    return [
        _surface(root, "kraken_subscription_builder", "app/ingestion/kraken.py", "trade"),
        _surface(root, "kraken_ohlc_subscription", "app/ingestion/kraken.py", "ohlc"),
        _surface(root, "websocket_reconnect_loop", "app/ingestion/service.py", "_backoff"),
        _surface(root, "raw_trade_normalizer", "app/ingestion/normalizers.py", "TradeEvent"),
        _surface(root, "raw_ohlc_normalizer", "app/ingestion/normalizers.py", "OhlcEvent"),
        _surface(root, "raw_trade_persistence", "app/ingestion/db.py", "raw_trades_table"),
        _surface(root, "raw_ohlc_persistence", "app/ingestion/db.py", "raw_ohlc_table"),
        _surface(root, "kafka_raw_trade_topic", "app/ingestion/service.py", "raw_trades"),
        _surface(root, "kafka_raw_ohlc_topic", "app/ingestion/service.py", "raw_ohlc"),
        _surface(
            root,
            "feature_replay_path",
            "app/ingestion/import_kraken_ohlcvt.py",
            "feature_replay",
        ),
        _surface(
            root,
            "training_readiness_checks",
            "app/training/data_readiness.py",
            "raw_ohlc",
        ),
    ]


def _surface(root: Path, name: str, path_value: str, needle: str) -> dict[str, str]:
    path = root / path_value
    status = (
        "PRESENT"
        if path.is_file() and needle in path.read_text(encoding="utf-8")
        else "MISSING"
    )
    return {
        "surface": name,
        "path": path_value,
        "status": status,
        "detail": f"Existing ingestion surface `{name}` is {status.lower()}.",
    }


def _proposed_source_contract() -> list[dict[str, str]]:
    return [
        _source(
            "order_book_depth",
            "book",
            "snapshot_and_update",
            "sequence_or_checksum_if_available",
        ),
        _source("top_of_book_spread", "book", "derived_from_best_bid_ask", "book_event_time"),
        _source(
            "depth_liquidity",
            "book",
            "derived_from_configured_depth_levels",
            "book_event_time",
        ),
        _source("trade_flow_imbalance", "trade", "derived_from_raw_trade_stream", "trade_id"),
    ]


def _source(data_family: str, channel: str, mode: str, ordering_key: str) -> dict[str, str]:
    return {
        "data_family": data_family,
        "source_exchange": "kraken",
        "source_api": "public_websocket_v2",
        "channel": channel,
        "collection_mode": mode,
        "ordering_key": ordering_key,
        "implementation_status": "PLANNED_ONLY",
    }


def _proposed_storage_contract() -> list[dict[str, str]]:
    return [
        _storage("research_raw_order_book", "raw depth events and snapshots"),
        _storage("research_order_book_replay", "deterministic reconstructed book states"),
        _storage("research_microstructure_features", "derived spread/liquidity/imbalance rows"),
    ]


def _storage(table_name: str, purpose: str) -> dict[str, str]:
    return {
        "proposed_table": table_name,
        "purpose": purpose,
        "key_fields": "source_exchange,symbol,event_time,received_at,sequence_or_checksum",
        "mutates_existing_contracts": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _replay_contract() -> list[dict[str, str]]:
    return [
        {
            "replay_component": "raw_event_ordering",
            "required_fields": "source_exchange,symbol,event_time,received_at,sequence_or_checksum",
            "determinism_rule": "stable sort by exchange time then received time then sequence",
        },
        {
            "replay_component": "book_state_reconstruction",
            "required_fields": "bids_json,asks_json,update_type",
            "determinism_rule": (
                "snapshot initializes state; updates apply only forward in replay order"
            ),
        },
        {
            "replay_component": "coverage_gaps",
            "required_fields": "symbol,event_time,sequence_or_checksum",
            "determinism_rule": "missing sequence/checksum creates explicit gap marker",
        },
    ]


def _feature_derivation_plan() -> list[dict[str, str]]:
    return [
        _feature("top_of_book_spread", "best_ask - best_bid", "book_state_current_or_past"),
        _feature(
            "relative_spread",
            "(best_ask - best_bid) / mid_price",
            "book_state_current_or_past",
        ),
        _feature(
            "depth_liquidity",
            "sum bid/ask size by depth level",
            "book_state_current_or_past",
        ),
        _feature(
            "order_book_imbalance",
            "bid_depth / (bid_depth + ask_depth)",
            "book_state_current_or_past",
        ),
        _feature(
            "trade_flow_imbalance",
            "signed buy/sell volume window",
            "raw_trades_current_or_past",
        ),
    ]


def _feature(name: str, definition: str, causal_scope: str) -> dict[str, str]:
    return {
        "feature_name": name,
        "definition": definition,
        "causal_scope": causal_scope,
        "uses_future_data": "False",
        "implementation_status": "PLANNED_ONLY",
    }


def _leakage_boundary_audit() -> list[dict[str, str]]:
    return [
        {
            "boundary": "feature_time",
            "rule": "microstructure features may use current or past book/trade events only",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
        {
            "boundary": "label_time",
            "rule": "future path may be used only in research outcome labels",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
        {
            "boundary": "production_runtime",
            "rule": "no production runtime reads new research tables in this plan",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
    ]


def _implementation_batches() -> list[dict[str, str]]:
    return [
        _batch("DU2", "add research-only schema and replay contracts", "NEXT"),
        _batch("DU3", "add sample Kraken book payload normalizers", "BLOCKED_UNTIL_DU2"),
        _batch("DU4", "derive research-only microstructure features", "BLOCKED_UNTIL_DU2_DU3"),
        _batch("DU5", "add coverage, gap, and replay determinism reports", "BLOCKED_UNTIL_DU4"),
        _batch("DU6", "plan optional isolated capture service", "REQUIRES_SEPARATE_APPROVAL"),
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
            "decision": "kraken_book_payload_fixture_source",
            "blocker": "SAMPLE_PAYLOAD_FIXTURE_REQUIRED_BEFORE_NORMALIZER_IMPLEMENTATION",
            "required_action": "CAPTURE_OR_CURATE_STATIC_KRAKEN_BOOK_FIXTURES",
        },
        {
            "decision": "depth_levels",
            "blocker": "CONFIGURED_DEPTH_LEVELS_NOT_SELECTED",
            "required_action": "SELECT_RESEARCH_DEPTH_LEVELS",
        },
        {
            "decision": "capture_runtime",
            "blocker": "NO_CAPTURE_SERVICE_APPROVAL",
            "required_action": "APPROVE_ISOLATED_RESEARCH_CAPTURE_BATCH",
        },
    ]


def _recommendation(existing_rows: list[Mapping[str, str]]) -> dict[str, Any]:
    boundaries_present = all(row["status"] == "PRESENT" for row in existing_rows)
    if boundaries_present:
        recommendation = "IMPLEMENT_RESEARCH_ONLY_MICROSTRUCTURE_SCHEMA_CONTRACTS"
        next_required_action = "BUILD_RESEARCH_ONLY_MICROSTRUCTURE_SCHEMA_AND_REPLAY_CONTRACTS"
    else:
        recommendation = "BLOCKED_SOURCE_CONTRACT_DECISION_REQUIRED"
        next_required_action = "RESTORE_OR_SELECT_INGESTION_BOUNDARIES"
    return {
        "recommendation": recommendation,
        "next_required_action": next_required_action,
        "next_actions": [
            {
                "action": next_required_action,
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
        "report_json": str(output_dir / "market_microstructure_ingestion_plan.json"),
        "report_md": str(output_dir / "market_microstructure_ingestion_plan.md"),
        "existing_ingestion_audit_csv": str(output_dir / "existing_ingestion_audit.csv"),
        "proposed_source_contract_csv": str(output_dir / "proposed_source_contract.csv"),
        "proposed_storage_contract_csv": str(output_dir / "proposed_storage_contract.csv"),
        "replay_contract_csv": str(output_dir / "replay_contract.csv"),
        "feature_derivation_plan_csv": str(output_dir / "feature_derivation_plan.csv"),
        "leakage_boundary_audit_csv": str(output_dir / "leakage_boundary_audit.csv"),
        "implementation_batches_csv": str(output_dir / "implementation_batches.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    source_rows: list[Mapping[str, str]],
    storage_rows: list[Mapping[str, str]],
    batch_rows: list[Mapping[str, str]],
    blocked_rows: list[Mapping[str, str]],
) -> str:
    lines = [
        "# Market Microstructure Research Ingestion Plan",
        "",
        f"- Proposed source: `{report['proposed_source']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Proposed Sources",
    ]
    lines.extend(f"- `{row['data_family']}` via `{row['channel']}`" for row in source_rows)
    lines.append("")
    lines.append("## Proposed Storage")
    lines.extend(f"- `{row['proposed_table']}`: {row['purpose']}" for row in storage_rows)
    lines.append("")
    lines.append("## Future Batches")
    lines.extend(f"- `{row['batch_id']}`: {row['goal']}" for row in batch_rows)
    lines.append("")
    lines.append("## Blocked Decisions")
    lines.extend(f"- `{row['decision']}`: `{row['blocker']}`" for row in blocked_rows)
    return "\n".join(lines) + "\n"
