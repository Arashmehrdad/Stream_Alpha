"""Research-only market microstructure schema and replay contract writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_schema_contracts"
M20_DECISION = "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "CONTRACT_ONLY",
    "NO_CAPTURE_SERVICE",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
CONTRACT_VERSION = "microstructure_schema_contracts_v1"


def write_market_microstructure_schema_contracts(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write research-only schema and replay contracts for future microstructure data."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rows = _contract_rows()
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": CONTRACT_VERSION,
        "repo_root": str(root),
        "m20_research_decision": M20_DECISION,
        "contract_status": "RESEARCH_ONLY_SCHEMA_AND_REPLAY_CONTRACTS_DEFINED",
        "table_contract_count": len(rows["table_contracts_csv"]),
        "replay_contract_count": len(rows["replay_ordering_contract_csv"]),
        "feature_contract_count": len(rows["feature_contracts_csv"]),
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
        "schema_version": f"{CONTRACT_VERSION}_manifest",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "source_plan": str(
            root / "artifacts/research_data_upgrade/market_microstructure_ingestion_plan"
        ),
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
        _markdown(report, rows["table_contracts_csv"], rows["implementation_batches_csv"]),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "table_contracts": rows["table_contracts_csv"],
            "column_contracts": rows["column_contracts_csv"],
            "ddl_contracts": rows["ddl_contracts_csv"],
            "replay_ordering_contract": rows["replay_ordering_contract_csv"],
            "feature_contracts": rows["feature_contracts_csv"],
            "leakage_boundary_audit": rows["leakage_boundary_audit_csv"],
            "blocked_decisions": rows["blocked_decisions_csv"],
            "recommendation_payload": recommendation,
        }
    )


def _contract_rows() -> dict[str, list[dict[str, str]]]:
    table_rows = _table_contracts()
    return {
        "table_contracts_csv": table_rows,
        "column_contracts_csv": _column_contracts(),
        "ddl_contracts_csv": _ddl_contracts(table_rows),
        "replay_ordering_contract_csv": _replay_ordering_contract(),
        "feature_contracts_csv": _feature_contracts(),
        "leakage_boundary_audit_csv": _leakage_boundary_audit(),
        "implementation_batches_csv": _implementation_batches(),
        "blocked_decisions_csv": _blocked_decisions(),
    }


def _table_contracts() -> list[dict[str, str]]:
    return [
        _table(
            "research_raw_order_book",
            "Raw Kraken book snapshots and updates for research replay.",
            "source_exchange,symbol,event_time,received_at,sequence_or_checksum",
        ),
        _table(
            "research_order_book_replay",
            "Deterministic reconstructed top/depth book states.",
            "source_exchange,symbol,event_time,replay_sequence",
        ),
        _table(
            "research_microstructure_features",
            "Causal spread, liquidity, and imbalance features from replay rows.",
            "source_exchange,symbol,event_time,feature_version",
        ),
    ]


def _table(table_name: str, purpose: str, unique_key: str) -> dict[str, str]:
    return {
        "table_name": table_name,
        "purpose": purpose,
        "unique_key": unique_key,
        "contract_status": "DEFINED_NOT_CREATED",
        "mutates_existing_tables": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _column_contracts() -> list[dict[str, str]]:
    common = [
        _column("source_exchange", "TEXT", "exchange identifier, default kraken", "True"),
        _column("symbol", "TEXT", "exchange symbol", "True"),
        _column("event_time", "TIMESTAMPTZ", "exchange event timestamp", "True"),
        _column("received_at", "TIMESTAMPTZ", "local receive timestamp", "True"),
        _column("sequence_or_checksum", "TEXT", "exchange ordering/checksum field", "False"),
        _column("payload", "JSONB", "raw source payload", "True"),
    ]
    book = [
        _column("update_type", "TEXT", "snapshot or update", "True"),
        _column("bids_json", "JSONB", "bid price/size levels", "True"),
        _column("asks_json", "JSONB", "ask price/size levels", "True"),
    ]
    replay = [
        _column("replay_sequence", "BIGINT", "deterministic replay row number", "True"),
        _column("best_bid", "DOUBLE PRECISION", "best bid after replay update", "False"),
        _column("best_ask", "DOUBLE PRECISION", "best ask after replay update", "False"),
        _column("book_gap_flag", "BOOLEAN", "true when ordering/checksum gap exists", "True"),
    ]
    features = [
        _column("feature_version", "TEXT", "microstructure feature version", "True"),
        _column("top_of_book_spread", "DOUBLE PRECISION", "best ask minus best bid", "False"),
        _column("relative_spread", "DOUBLE PRECISION", "spread divided by mid price", "False"),
        _column("depth_liquidity", "DOUBLE PRECISION", "depth size by configured level", "False"),
        _column("trade_flow_imbalance", "DOUBLE PRECISION", "causal signed trade flow", "False"),
    ]
    return [
        *[_with_table("research_raw_order_book", column) for column in [*common, *book]],
        *[_with_table("research_order_book_replay", column) for column in [*common, *replay]],
        *[
            _with_table("research_microstructure_features", column)
            for column in [*common, *features]
        ],
    ]


def _column(name: str, data_type: str, description: str, required: str) -> dict[str, str]:
    return {
        "column_name": name,
        "data_type": data_type,
        "description": description,
        "required": required,
        "uses_future_data": "False",
    }


def _with_table(table_name: str, column: Mapping[str, str]) -> dict[str, str]:
    return {"table_name": table_name, **dict(column)}


def _ddl_contracts(table_rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "table_name": row["table_name"],
            "ddl_contract": _ddl_for_table(row["table_name"]),
            "execution_status": "NOT_EXECUTED_CONTRACT_ONLY",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        }
        for row in table_rows
    ]


def _ddl_for_table(table_name: str) -> str:
    if table_name == "research_raw_order_book":
        return _compact_sql(
            """
            CREATE TABLE IF NOT EXISTS research_raw_order_book (
                id BIGSERIAL PRIMARY KEY,
                source_exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                received_at TIMESTAMPTZ NOT NULL,
                sequence_or_checksum TEXT NULL,
                update_type TEXT NOT NULL,
                bids_json JSONB NOT NULL,
                asks_json JSONB NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (source_exchange, symbol, event_time, received_at, sequence_or_checksum)
            )
            """
        )
    if table_name == "research_order_book_replay":
        return _compact_sql(
            """
            CREATE TABLE IF NOT EXISTS research_order_book_replay (
                id BIGSERIAL PRIMARY KEY,
                source_exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                received_at TIMESTAMPTZ NOT NULL,
                replay_sequence BIGINT NOT NULL,
                sequence_or_checksum TEXT NULL,
                best_bid DOUBLE PRECISION NULL,
                best_ask DOUBLE PRECISION NULL,
                book_gap_flag BOOLEAN NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (source_exchange, symbol, event_time, replay_sequence)
            )
            """
        )
    return _compact_sql(
        """
        CREATE TABLE IF NOT EXISTS research_microstructure_features (
            id BIGSERIAL PRIMARY KEY,
            source_exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_time TIMESTAMPTZ NOT NULL,
            received_at TIMESTAMPTZ NOT NULL,
            feature_version TEXT NOT NULL,
            sequence_or_checksum TEXT NULL,
            top_of_book_spread DOUBLE PRECISION NULL,
            relative_spread DOUBLE PRECISION NULL,
            depth_liquidity DOUBLE PRECISION NULL,
            trade_flow_imbalance DOUBLE PRECISION NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (source_exchange, symbol, event_time, feature_version)
        )
        """
    )


def _compact_sql(sql: str) -> str:
    return " ".join(line.strip() for line in sql.splitlines() if line.strip())


def _replay_ordering_contract() -> list[dict[str, str]]:
    return [
        {
            "contract": "partitioning",
            "rule": "replay independently by source_exchange and symbol",
            "required_fields": "source_exchange,symbol",
            "gap_behavior": "no cross-symbol state",
        },
        {
            "contract": "ordering",
            "rule": "stable sort by event_time, received_at, sequence_or_checksum, payload hash",
            "required_fields": "event_time,received_at,sequence_or_checksum,payload",
            "gap_behavior": "mark book_gap_flag when sequence/checksum is missing or discontinuous",
        },
        {
            "contract": "state_application",
            "rule": "snapshot initializes book; updates apply only forward in replay order",
            "required_fields": "update_type,bids_json,asks_json",
            "gap_behavior": "retain rows with explicit gap flags instead of silently filling",
        },
    ]


def _feature_contracts() -> list[dict[str, str]]:
    return [
        _feature("top_of_book_spread", "best_ask - best_bid", "book replay current/past"),
        _feature("relative_spread", "spread / midpoint", "book replay current/past"),
        _feature(
            "depth_liquidity",
            "sum bid and ask sizes by depth level",
            "book replay current/past",
        ),
        _feature(
            "order_book_imbalance",
            "bid depth share of total depth",
            "book replay current/past",
        ),
        _feature(
            "trade_flow_imbalance",
            "signed trade volume rolling window",
            "raw trades current/past",
        ),
    ]


def _feature(name: str, definition: str, causal_scope: str) -> dict[str, str]:
    return {
        "feature_name": name,
        "definition": definition,
        "causal_scope": causal_scope,
        "uses_future_data": "False",
        "uses_labels": "False",
        "uses_economic_outcomes": "False",
        "implementation_status": "CONTRACT_ONLY",
    }


def _leakage_boundary_audit() -> list[dict[str, str]]:
    return [
        _boundary("feature_derivation", "current and past book/trade events only"),
        _boundary("research_labeling", "future paths allowed only in separate label artifacts"),
        _boundary("runtime", "no production runtime reads these planned research tables"),
        _boundary("existing_contracts", "raw_trades, raw_ohlc, and feature_ohlc remain unchanged"),
    ]


def _boundary(name: str, rule: str) -> dict[str, str]:
    return {
        "boundary": name,
        "rule": rule,
        "runtime_effect": "NO_RUNTIME_EFFECT",
        "leakage_status": "SAFE_CONTRACT_BOUNDARY",
    }


def _implementation_batches() -> list[dict[str, str]]:
    return [
        _batch("DU3", "add sample Kraken book payload normalizers", "NEXT"),
        _batch("DU4", "derive research-only microstructure features", "BLOCKED_UNTIL_DU3"),
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
            "decision": "sample_payload_fixtures",
            "blocker": "STATIC_KRAKEN_BOOK_FIXTURES_REQUIRED_FOR_NORMALIZER_TESTS",
            "required_action": "ADD_STATIC_BOOK_PAYLOAD_FIXTURES",
        },
        {
            "decision": "depth_levels",
            "blocker": "RESEARCH_DEPTH_LEVELS_NOT_FINALIZED",
            "required_action": "SELECT_DEPTH_LEVELS_FOR_FEATURE_DERIVATION",
        },
        {
            "decision": "capture_service",
            "blocker": "CAPTURE_SERVICE_OUT_OF_SCOPE_FOR_DU2",
            "required_action": "KEEP_CAPTURE_BLOCKED_UNTIL_DU6_APPROVAL",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "ADD_SAMPLE_KRAKEN_BOOK_FIXTURE_NORMALIZERS",
        "next_required_action": "BUILD_SAMPLE_BOOK_PAYLOAD_NORMALIZERS_CONTRACT_ONLY",
        "next_actions": [
            {
                "action": "BUILD_SAMPLE_BOOK_PAYLOAD_NORMALIZERS_CONTRACT_ONLY",
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
        "report_json": str(output_dir / "microstructure_schema_contracts.json"),
        "report_md": str(output_dir / "microstructure_schema_contracts.md"),
        "table_contracts_csv": str(output_dir / "table_contracts.csv"),
        "column_contracts_csv": str(output_dir / "column_contracts.csv"),
        "ddl_contracts_csv": str(output_dir / "ddl_contracts.csv"),
        "replay_ordering_contract_csv": str(output_dir / "replay_ordering_contract.csv"),
        "feature_contracts_csv": str(output_dir / "feature_contracts.csv"),
        "leakage_boundary_audit_csv": str(output_dir / "leakage_boundary_audit.csv"),
        "implementation_batches_csv": str(output_dir / "implementation_batches.csv"),
        "blocked_decisions_csv": str(output_dir / "blocked_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    table_rows: list[Mapping[str, str]],
    batch_rows: list[Mapping[str, str]],
) -> str:
    lines = [
        "# Microstructure Schema And Replay Contracts",
        "",
        f"- Contract status: `{report['contract_status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- M20 decision: `{report['m20_research_decision']}`",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "- Profitability status: `NO_PROFIT_CLAIM`",
        "",
        "## Table Contracts",
    ]
    lines.extend(f"- `{row['table_name']}`: {row['purpose']}" for row in table_rows)
    lines.append("")
    lines.append("## Future Batches")
    lines.extend(f"- `{row['batch_id']}`: {row['goal']}" for row in batch_rows)
    return "\n".join(lines) + "\n"
