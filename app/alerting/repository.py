"""PostgreSQL persistence for the M17 operational alerting foundation."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone

import asyncpg

from app.alerting.schemas import OperationalAlertEvent, OperationalAlertState


ALERT_EVENTS_TABLE = "operational_alert_events"
ALERT_STATE_TABLE = "operational_alert_state"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _quote_table_name(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {name}")
    return ".".join(_quote_identifier(part) for part in parts)


def _build_index_name(table_name: str, suffix: str) -> str:
    return _quote_identifier(f"{table_name}_{suffix}")


def _table_basename(name: str) -> str:
    return name.split(".")[-1]


class OperationalAlertRepository:
    """Repository for the normalized M17 alert timeline and state tables."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._events_table = _quote_table_name(ALERT_EVENTS_TABLE)
        self._state_table = _quote_table_name(ALERT_STATE_TABLE)
        self._pool: asyncpg.Pool | None = None

    @property
    def dsn(self) -> str:
        """Expose the underlying DSN for explicit reuse when needed."""
        return self._dsn

    async def connect(self) -> None:
        """Open the repository pool and ensure the M17 tables exist."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        await self._ensure_schema()

    async def close(self) -> None:
        """Close the repository pool."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def insert_event(
        self,
        event: OperationalAlertEvent,
    ) -> OperationalAlertEvent:
        """Insert one canonical alert timeline event."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._events_table} (
                service_name,
                execution_mode,
                category,
                severity,
                event_state,
                reason_code,
                source_component,
                symbol,
                fingerprint,
                summary_text,
                detail,
                event_time,
                related_order_request_id,
                related_decision_trace_id,
                payload_json
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::jsonb
            )
            RETURNING *
            """,
            event.service_name,
            event.execution_mode,
            event.category,
            event.severity,
            event.event_state,
            event.reason_code,
            event.source_component,
            event.symbol,
            event.fingerprint,
            event.summary_text,
            event.detail,
            event.event_time,
            event.related_order_request_id,
            event.related_decision_trace_id,
            json.dumps(event.payload_json),
        )
        if row is None:
            raise RuntimeError("Operational alert event insert returned no row")
        return _event_from_row(row)

    async def load_state(self, *, fingerprint: str) -> OperationalAlertState | None:
        """Load one current-state row by fingerprint."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._state_table}
            WHERE fingerprint = $1
            """,
            fingerprint,
        )
        if row is None:
            return None
        return _state_from_row(row)

    async def save_state(self, state: OperationalAlertState) -> None:
        """Upsert one current-state row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._state_table} (
                fingerprint,
                service_name,
                execution_mode,
                category,
                symbol,
                source_component,
                is_active,
                severity,
                reason_code,
                opened_at,
                last_seen_at,
                last_event_id,
                occurrence_count
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            )
            ON CONFLICT (fingerprint)
            DO UPDATE SET
                service_name = EXCLUDED.service_name,
                execution_mode = EXCLUDED.execution_mode,
                category = EXCLUDED.category,
                symbol = EXCLUDED.symbol,
                source_component = EXCLUDED.source_component,
                is_active = EXCLUDED.is_active,
                severity = EXCLUDED.severity,
                reason_code = EXCLUDED.reason_code,
                opened_at = EXCLUDED.opened_at,
                last_seen_at = EXCLUDED.last_seen_at,
                last_event_id = EXCLUDED.last_event_id,
                occurrence_count = EXCLUDED.occurrence_count
            """,
            state.fingerprint,
            state.service_name,
            state.execution_mode,
            state.category,
            state.symbol,
            state.source_component,
            state.is_active,
            state.severity,
            state.reason_code,
            state.opened_at,
            state.last_seen_at,
            state.last_event_id,
            state.occurrence_count,
        )

    async def load_active_states(
        self,
        *,
        service_name: str,
        execution_mode: str,
    ) -> list[OperationalAlertState]:
        """Load all active alert states for one service and execution mode."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._state_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND is_active = TRUE
            ORDER BY severity DESC, opened_at ASC, fingerprint ASC
            """,
            service_name,
            execution_mode,
        )
        return [_state_from_row(row) for row in rows]

    async def load_events_for_day(
        self,
        *,
        service_name: str,
        execution_mode: str,
        summary_date: date,
    ) -> list[OperationalAlertEvent]:
        """Load all alert events for one UTC day."""
        pool = self._require_pool()
        start = datetime.combine(summary_date, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._events_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND event_time >= $3
              AND event_time < $4
            ORDER BY event_time ASC, id ASC
            """,
            service_name,
            execution_mode,
            start,
            end,
        )
        return [_event_from_row(row) for row in rows]

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        events_time_index = _build_index_name(
            _table_basename(ALERT_EVENTS_TABLE),
            "service_mode_event_time_idx",
        )
        events_fingerprint_index = _build_index_name(
            _table_basename(ALERT_EVENTS_TABLE),
            "fingerprint_idx",
        )
        state_activity_index = _build_index_name(
            _table_basename(ALERT_STATE_TABLE),
            "service_mode_active_idx",
        )
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._events_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    event_state TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    source_component TEXT NOT NULL,
                    symbol TEXT NULL,
                    fingerprint TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    detail TEXT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    related_order_request_id BIGINT NULL,
                    related_decision_trace_id BIGINT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._state_table} (
                    fingerprint TEXT PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    category TEXT NOT NULL,
                    symbol TEXT NULL,
                    source_component TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    opened_at TIMESTAMPTZ NOT NULL,
                    last_seen_at TIMESTAMPTZ NOT NULL,
                    last_event_id BIGINT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {events_time_index}
                ON {self._events_table} (service_name, execution_mode, event_time DESC, id DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {events_fingerprint_index}
                ON {self._events_table} (fingerprint, event_time DESC, id DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {state_activity_index}
                ON {self._state_table} (service_name, execution_mode, is_active, category)
                """
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("OperationalAlertRepository has not been connected")
        return self._pool


def _event_from_row(row: asyncpg.Record) -> OperationalAlertEvent:
    return OperationalAlertEvent(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        category=str(row["category"]),
        severity=str(row["severity"]),
        event_state=str(row["event_state"]),
        reason_code=str(row["reason_code"]),
        source_component=str(row["source_component"]),
        symbol=None if row["symbol"] is None else str(row["symbol"]),
        fingerprint=str(row["fingerprint"]),
        summary_text=str(row["summary_text"]),
        detail=None if row["detail"] is None else str(row["detail"]),
        event_time=row["event_time"],
        related_order_request_id=(
            None
            if row["related_order_request_id"] is None
            else int(row["related_order_request_id"])
        ),
        related_decision_trace_id=(
            None
            if row["related_decision_trace_id"] is None
            else int(row["related_decision_trace_id"])
        ),
        payload_json=_jsonb_to_object(row["payload_json"]),
        event_id=int(row["id"]),
        created_at=row["created_at"],
    )


def _state_from_row(row: asyncpg.Record) -> OperationalAlertState:
    return OperationalAlertState(
        fingerprint=str(row["fingerprint"]),
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        category=str(row["category"]),
        symbol=None if row["symbol"] is None else str(row["symbol"]),
        source_component=str(row["source_component"]),
        is_active=bool(row["is_active"]),
        severity=str(row["severity"]),
        reason_code=str(row["reason_code"]),
        opened_at=row["opened_at"],
        last_seen_at=row["last_seen_at"],
        last_event_id=(
            None if row["last_event_id"] is None else int(row["last_event_id"])
        ),
        occurrence_count=int(row["occurrence_count"]),
    )


def _jsonb_to_object(value: object) -> dict:
    if isinstance(value, str):
        loaded = json.loads(value)
    else:
        loaded = value
    if not isinstance(loaded, dict):
        raise ValueError("Alert payload_json must deserialize into a mapping")
    return loaded
