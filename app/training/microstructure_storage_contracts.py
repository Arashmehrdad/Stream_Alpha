"""Research-only microstructure storage contract artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_storage_contracts"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "DRY_RUN_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_microstructure_storage_contracts(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
    apply: bool = False,  # pylint: disable=redefined-builtin
) -> dict[str, Any]:
    """Write DU7 research storage contracts; never executes DDL by default."""
    if apply:
        raise ValueError("DDL apply is blocked; this batch is dry-run contract only")
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_storage_contracts_v1",
        "repo_root": str(root),
        "storage_contract_status": "RESEARCH_STORAGE_CONTRACTS_DEFINED_DRY_RUN",
        "ddl_apply_executed": False,
        "table_contract_count": len(rows["table_contracts_csv"]),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_storage_contracts_manifest_v1",
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
        {**report, "manifest": manifest, "recommendation_payload": recommendation}
    )


def _rows() -> dict[str, list[dict[str, str]]]:
    tables = [
        ("research_raw_order_book", "raw Kraken book snapshot/update payloads"),
        ("research_order_book_replay", "deterministic replayed order-book states"),
        ("research_microstructure_features", "causal derived microstructure features"),
        ("research_capture_health", "research capture status and gap observations"),
    ]
    return {
        "table_contracts_csv": [_table(name, purpose) for name, purpose in tables],
        "ddl_contracts_csv": [_ddl_row(name) for name, _purpose in tables],
        "boundary_audit_csv": _boundary_rows(),
        "blocked_actions_csv": _blocked_rows(),
    }


def _table(name: str, purpose: str) -> dict[str, str]:
    return {
        "table_name": name,
        "purpose": purpose,
        "creation_status": "DRY_RUN_ONLY_NOT_CREATED",
        "mutates_existing_tables": "False",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _ddl_row(name: str) -> dict[str, str]:
    return {
        "table_name": name,
        "ddl_sql": _ddl_sql(name),
        "execution_status": "NOT_EXECUTED",
        "runtime_effect": "NO_RUNTIME_EFFECT",
    }


def _ddl_sql(name: str) -> str:
    common = (
        "id BIGSERIAL PRIMARY KEY, source_exchange TEXT NOT NULL, symbol TEXT NOT NULL, "
        "event_time TIMESTAMPTZ NOT NULL, received_at TIMESTAMPTZ NOT NULL, "
        "payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
    )
    if name == "research_raw_order_book":
        extra = (
            "sequence_or_checksum TEXT NULL, update_type TEXT NOT NULL, "
            "bids_json JSONB NOT NULL, asks_json JSONB NOT NULL"
        )
        unique = "source_exchange, symbol, event_time, received_at, sequence_or_checksum"
    elif name == "research_order_book_replay":
        extra = (
            "replay_sequence BIGINT NOT NULL, best_bid DOUBLE PRECISION NULL, "
            "best_ask DOUBLE PRECISION NULL, book_gap_flag BOOLEAN NOT NULL"
        )
        unique = "source_exchange, symbol, event_time, replay_sequence"
    elif name == "research_microstructure_features":
        extra = (
            "feature_version TEXT NOT NULL, top_of_book_spread DOUBLE PRECISION NULL, "
            "relative_spread DOUBLE PRECISION NULL, order_book_imbalance DOUBLE PRECISION NULL"
        )
        unique = "source_exchange, symbol, event_time, feature_version"
    else:
        extra = "status TEXT NOT NULL, detail TEXT NOT NULL"
        unique = "source_exchange, symbol, event_time, status"
    return f"CREATE TABLE IF NOT EXISTS {name} ({common}, {extra}, UNIQUE ({unique}));"


def _boundary_rows() -> list[dict[str, str]]:
    return [
        _boundary("existing_tables", "raw_trades/raw_ohlc/feature_ohlc are not modified"),
        _boundary("runtime", "runtime services do not read research microstructure tables"),
        _boundary("execution", "DDL is emitted as text only in this batch"),
    ]


def _boundary(name: str, detail: str) -> dict[str, str]:
    return {"boundary": name, "detail": detail, "runtime_effect": "NO_RUNTIME_EFFECT"}


def _blocked_rows() -> list[dict[str, str]]:
    return [
        {
            "action": "apply_ddl",
            "blocker": "SEPARATE_APPROVAL_REQUIRED",
            "required_action": "APPROVE_SCHEMA_APPLY_IN_FUTURE_BATCH",
        }
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "BUILD_ISOLATED_RESEARCH_CAPTURE_SERVICE_DRY_RUN",
        "next_required_action": "IMPLEMENT_DRY_RUN_MICROSTRUCTURE_CAPTURE_SERVICE",
        "next_actions": [
            {
                "action": "IMPLEMENT_DRY_RUN_MICROSTRUCTURE_CAPTURE_SERVICE",
                "runtime_effect": "NO_RUNTIME_EFFECT",
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
        "report_json": str(output_dir / "microstructure_storage_contracts.json"),
        "report_md": str(output_dir / "microstructure_storage_contracts.md"),
        "table_contracts_csv": str(output_dir / "table_contracts.csv"),
        "ddl_contracts_csv": str(output_dir / "ddl_contracts.csv"),
        "boundary_audit_csv": str(output_dir / "boundary_audit.csv"),
        "blocked_actions_csv": str(output_dir / "blocked_actions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Microstructure Storage Contracts\n\n"
        f"- Status: `{report['storage_contract_status']}`\n"
        f"- DDL apply executed: `{report['ddl_apply_executed']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
        f"- Next required action: `{report['next_required_action']}`\n"
        "- Runtime status: `NO_RUNTIME_EFFECT`\n"
        "- Profitability status: `NO_PROFIT_CLAIM`\n"
    )
