"""Dry-run isolated research microstructure capture service planning."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import asyncpg
import websockets

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic
from app.training.market_microstructure_book_normalizers import planned_kraken_book_subscription
from app.training.market_microstructure_book_normalizers import (
    normalize_kraken_book_payload_fixture,
)


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_capture_service_dry_run"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "DRY_RUN_ONLY",
    "NO_NETWORK_CAPTURE",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
MAX_DRY_RUN_DURATION_SECONDS = 3600
MAX_DRY_RUN_EVENTS = 1_000_000


@dataclass(frozen=True, slots=True)
class CaptureDryRunPlan:
    """Validated dry-run capture plan."""

    symbols: tuple[str, ...]
    depth: int
    duration_seconds: int
    max_events: int
    dry_run: bool
    subscription: dict[str, Any]


def build_capture_dry_run_plan(
    *,
    symbols: tuple[str, ...],
    depth: int,
    duration_seconds: int,
    max_events: int,
) -> CaptureDryRunPlan:
    """Validate and render a dry-run capture plan."""
    if duration_seconds <= 0 or duration_seconds > MAX_DRY_RUN_DURATION_SECONDS:
        raise ValueError("duration_seconds must be between 1 and 3600")
    if max_events <= 0 or max_events > MAX_DRY_RUN_EVENTS:
        raise ValueError("max_events must be between 1 and 1000000")
    subscription = planned_kraken_book_subscription(symbols, depth=depth)
    return CaptureDryRunPlan(
        symbols=symbols,
        depth=depth,
        duration_seconds=duration_seconds,
        max_events=max_events,
        dry_run=True,
        subscription=subscription,
    )


def write_microstructure_capture_service_dry_run(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    repo_root: Path,
    output_dir: Path | None = None,
    symbols: tuple[str, ...] = ("BTC/USD",),
    depth: int = 10,
    duration_seconds: int = 60,
    max_events: int = 1000,
    execute: bool = False,
    dsn: str | None = None,
    ws_url: str = "wss://ws.kraken.com/v2",
) -> dict[str, Any]:
    """Write an isolated capture service dry-run artifact."""
    if execute and not dsn:
        raise ValueError("Real capture execution requires a PostgreSQL DSN")
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_capture_dry_run_plan(
        symbols=symbols,
        depth=depth,
        duration_seconds=duration_seconds,
        max_events=max_events,
    )
    rows = _rows(plan)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    capture_result: dict[str, Any] | None = None
    if execute:
        capture_result = run_bounded_capture_sync(plan=plan, dsn=str(dsn), ws_url=ws_url)
    report = {
        "schema_version": "microstructure_capture_service_dry_run_v1",
        "repo_root": str(root),
        "capture_service_status": (
            "BOUNDED_CAPTURE_EXECUTED" if capture_result
            else "DRY_RUN_CAPTURE_SERVICE_PLAN_DEFINED"
        ),
        "network_capture_executed": bool(capture_result),
        "database_writes_executed": bool(capture_result),
        "captured_event_count": (
            0 if capture_result is None else capture_result["captured_event_count"]
        ),
        "symbols": list(plan.symbols),
        "depth": plan.depth,
        "duration_seconds": plan.duration_seconds,
        "max_events": plan.max_events,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_capture_service_dry_run_manifest_v1",
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
            "dry_run_plan": asdict(plan),
            "capture_result": capture_result,
        }
    )


async def run_bounded_capture(
    *,
    plan: CaptureDryRunPlan,
    dsn: str,
    ws_url: str = "wss://ws.kraken.com/v2",
) -> dict[str, Any]:
    """Run a bounded isolated research capture."""
    connection = await asyncpg.connect(dsn)
    captured = 0
    started_at = datetime.now(UTC)
    try:
        async with websockets.connect(ws_url) as websocket:
            await websocket.send(json.dumps(plan.subscription))
            while captured < plan.max_events:
                if (datetime.now(UTC) - started_at).total_seconds() >= plan.duration_seconds:
                    break
                raw_message = await websocket.recv()
                payload = json.loads(str(raw_message))
                if not _is_book_data_message(payload):
                    continue
                event = normalize_kraken_book_payload_fixture(
                    payload,
                    received_at=datetime.now(UTC),
                )
                await _insert_book_event(connection, event)
                captured += 1
    finally:
        await connection.close()
    return {
        "captured_event_count": captured,
        "duration_seconds": plan.duration_seconds,
        "max_events": plan.max_events,
    }


def run_bounded_capture_sync(
    *,
    plan: CaptureDryRunPlan,
    dsn: str,
    ws_url: str = "wss://ws.kraken.com/v2",
) -> dict[str, Any]:
    """Synchronous wrapper for the bounded async capture."""
    return asyncio.run(run_bounded_capture(plan=plan, dsn=dsn, ws_url=ws_url))


def _is_book_data_message(payload: Mapping[str, Any]) -> bool:
    return payload.get("channel") == "book" and payload.get("type") in {"snapshot", "update"}


async def _insert_book_event(connection: Any, event: Any) -> None:
    bids_json = json.dumps([asdict(level) for level in event.bids])
    asks_json = json.dumps([asdict(level) for level in event.asks])
    payload_json = json.dumps(make_json_safe(event.payload))
    await connection.execute(
        """
        INSERT INTO research_raw_order_book (
            source_exchange, symbol, event_time, received_at, sequence_or_checksum,
            update_type, bids_json, asks_json, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb)
        ON CONFLICT (source_exchange, symbol, event_time, received_at, sequence_or_checksum)
        DO NOTHING
        """,
        event.source_exchange,
        event.symbol,
        event.event_time,
        event.received_at,
        event.sequence_or_checksum,
        event.update_type,
        bids_json,
        asks_json,
        payload_json,
    )


def _rows(plan: CaptureDryRunPlan) -> dict[str, list[dict[str, str]]]:
    return {
        "capture_plan_csv": [
            {
                "symbols": ",".join(plan.symbols),
                "depth": str(plan.depth),
                "duration_seconds": str(plan.duration_seconds),
                "max_events": str(plan.max_events),
                "dry_run": str(plan.dry_run),
                "runtime_effect": "NO_RUNTIME_EFFECT",
            }
        ],
        "subscription_plan_csv": [
            {
                "method": str(plan.subscription["method"]),
                "channel": str(plan.subscription["params"]["channel"]),
                "symbols": ",".join(plan.subscription["params"]["symbol"]),
                "depth": str(plan.subscription["params"]["depth"]),
                "snapshot": str(plan.subscription["params"]["snapshot"]),
            }
        ],
        "safety_gates_csv": _safety_gates(),
        "blocked_actions_csv": _blocked_actions(),
    }


def _safety_gates() -> list[dict[str, str]]:
    return [
        _gate("dry_run_default", "service produces plan artifacts unless execute is approved"),
        _gate("bounded_duration", "duration is capped at 3600 seconds"),
        _gate("bounded_events", "max event count is capped at 1000000"),
        _gate("isolated_tables", "future writes target research tables only"),
    ]


def _gate(name: str, detail: str) -> dict[str, str]:
    return {"gate": name, "detail": detail, "runtime_effect": "NO_RUNTIME_EFFECT"}


def _blocked_actions() -> list[dict[str, str]]:
    return [
        {
            "action": "network_capture",
            "blocker": "REAL_CAPTURE_BLOCKED_IN_DU8",
            "required_action": "USE_DU9_SMOKE_HARNESS_WITH_SEPARATE_APPROVAL",
        }
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "ADD_BOUNDED_CAPTURE_SMOKE_HARNESS_DRY_RUN",
        "next_required_action": "IMPLEMENT_BOUNDED_CAPTURE_SMOKE_HARNESS_DRY_RUN",
        "next_actions": [
            {
                "action": "IMPLEMENT_BOUNDED_CAPTURE_SMOKE_HARNESS_DRY_RUN",
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
        "report_json": str(output_dir / "microstructure_capture_service_dry_run.json"),
        "report_md": str(output_dir / "microstructure_capture_service_dry_run.md"),
        "capture_plan_csv": str(output_dir / "capture_plan.csv"),
        "subscription_plan_csv": str(output_dir / "subscription_plan.csv"),
        "safety_gates_csv": str(output_dir / "safety_gates.csv"),
        "blocked_actions_csv": str(output_dir / "blocked_actions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Microstructure Capture Service Dry Run\n\n"
        f"- Status: `{report['capture_service_status']}`\n"
        f"- Network capture executed: `{report['network_capture_executed']}`\n"
        f"- Database writes executed: `{report['database_writes_executed']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
        f"- Next required action: `{report['next_required_action']}`\n"
    )
