"""Small PostgreSQL reliability store for non-trading services."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import asyncpg

from app.common.serialization import make_json_safe
from app.reliability.schemas import (
    FeatureLagSnapshot,
    RecoveryEvent,
    ReliabilityState,
    ServiceHeartbeat,
    SystemReliabilitySnapshot,
)


HEARTBEATS_TABLE = "service_heartbeats"
RELIABILITY_STATE_TABLE = "reliability_state"
RELIABILITY_EVENTS_TABLE = "reliability_events"
RELIABILITY_LAG_TABLE = "reliability_lag_state"
RELIABILITY_SYSTEM_TABLE = "reliability_system_state"

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


class ReliabilityStore:
    """Minimal reliability persistence for heartbeats, state, and events."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._heartbeats_table = _quote_table_name(HEARTBEATS_TABLE)
        self._reliability_state_table = _quote_table_name(RELIABILITY_STATE_TABLE)
        self._reliability_events_table = _quote_table_name(RELIABILITY_EVENTS_TABLE)
        self._reliability_lag_table = _quote_table_name(RELIABILITY_LAG_TABLE)
        self._reliability_system_table = _quote_table_name(RELIABILITY_SYSTEM_TABLE)
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the connection pool and ensure the additive reliability tables exist."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
        await self._ensure_schema()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def save_service_heartbeat(self, heartbeat: ServiceHeartbeat) -> None:
        """Insert one reliability heartbeat row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._heartbeats_table} (
                service_name,
                component_name,
                heartbeat_at,
                health_overall_status,
                reason_code,
                details
            ) VALUES (
                $1, $2, $3, $4, $5, $6
            )
            """,
            heartbeat.service_name,
            heartbeat.component_name,
            heartbeat.heartbeat_at,
            heartbeat.health_overall_status,
            heartbeat.reason_code,
            heartbeat.detail,
        )

    async def save_reliability_state(self, state: ReliabilityState) -> None:
        """Upsert one reliability state row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._reliability_state_table} (
                service_name,
                component_name,
                health_overall_status,
                freshness_status,
                breaker_state,
                failure_count,
                success_count,
                last_heartbeat_at,
                last_success_at,
                last_failure_at,
                opened_at,
                reason_code,
                details,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW()
            )
            ON CONFLICT (service_name, component_name)
            DO UPDATE SET
                health_overall_status = EXCLUDED.health_overall_status,
                freshness_status = EXCLUDED.freshness_status,
                breaker_state = EXCLUDED.breaker_state,
                failure_count = EXCLUDED.failure_count,
                success_count = EXCLUDED.success_count,
                last_heartbeat_at = EXCLUDED.last_heartbeat_at,
                last_success_at = EXCLUDED.last_success_at,
                last_failure_at = EXCLUDED.last_failure_at,
                opened_at = EXCLUDED.opened_at,
                reason_code = EXCLUDED.reason_code,
                details = EXCLUDED.details,
                updated_at = NOW()
            """,
            state.service_name,
            state.component_name,
            state.health_overall_status,
            state.freshness_status,
            state.breaker_state,
            state.failure_count,
            state.success_count,
            state.last_heartbeat_at,
            state.last_success_at,
            state.last_failure_at,
            state.opened_at,
            state.reason_code,
            state.detail,
        )

    async def insert_reliability_event(self, event: RecoveryEvent) -> None:
        """Insert one reliability event audit row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._reliability_events_table} (
                service_name,
                component_name,
                event_type,
                event_time,
                reason_code,
                health_overall_status,
                freshness_status,
                breaker_state,
                details
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            )
            """,
            event.service_name,
            event.component_name,
            event.event_type,
            event.event_time,
            event.reason_code,
            event.health_overall_status,
            event.freshness_status,
            event.breaker_state,
            event.detail,
        )

    async def save_feature_lag_state(self, lag_state: FeatureLagSnapshot) -> None:
        """Upsert one per-symbol feature lag snapshot."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._reliability_lag_table} (
                service_name,
                component_name,
                symbol,
                latest_raw_event_received_at,
                latest_feature_interval_begin,
                latest_feature_as_of_time,
                time_lag_seconds,
                processing_lag_seconds,
                time_lag_reason_code,
                processing_lag_reason_code,
                lag_breach,
                health_overall_status,
                reason_code,
                details,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW()
            )
            ON CONFLICT (service_name, component_name, symbol)
            DO UPDATE SET
                latest_raw_event_received_at = EXCLUDED.latest_raw_event_received_at,
                latest_feature_interval_begin = EXCLUDED.latest_feature_interval_begin,
                latest_feature_as_of_time = EXCLUDED.latest_feature_as_of_time,
                time_lag_seconds = EXCLUDED.time_lag_seconds,
                processing_lag_seconds = EXCLUDED.processing_lag_seconds,
                time_lag_reason_code = EXCLUDED.time_lag_reason_code,
                processing_lag_reason_code = EXCLUDED.processing_lag_reason_code,
                lag_breach = EXCLUDED.lag_breach,
                health_overall_status = EXCLUDED.health_overall_status,
                reason_code = EXCLUDED.reason_code,
                details = EXCLUDED.details,
                updated_at = NOW()
            """,
            lag_state.service_name,
            lag_state.component_name,
            lag_state.symbol,
            lag_state.latest_raw_event_received_at,
            lag_state.latest_feature_interval_begin,
            lag_state.latest_feature_as_of_time,
            lag_state.time_lag_seconds,
            lag_state.processing_lag_seconds,
            lag_state.time_lag_reason_code,
            lag_state.processing_lag_reason_code,
            lag_state.lag_breach,
            lag_state.health_overall_status,
            lag_state.reason_code,
            lag_state.detail,
        )

    async def load_latest_service_heartbeat(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ServiceHeartbeat | None:
        """Load the newest heartbeat row for one component."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._heartbeats_table}
            WHERE service_name = $1 AND component_name = $2
            ORDER BY heartbeat_at DESC, id DESC
            LIMIT 1
            """,
            service_name,
            component_name,
        )
        if row is None:
            return None
        return _heartbeat_from_row(row)

    async def load_latest_service_heartbeats(
        self,
        service_components: Sequence[tuple[str, str]],
    ) -> list[ServiceHeartbeat]:
        """Load the latest heartbeat row for each requested service component."""
        heartbeats: list[ServiceHeartbeat] = []
        for service_name, component_name in service_components:
            heartbeat = await self.load_latest_service_heartbeat(
                service_name=service_name,
                component_name=component_name,
            )
            if heartbeat is not None:
                heartbeats.append(heartbeat)
        return heartbeats

    async def load_reliability_state(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ReliabilityState | None:
        """Load one persisted reliability state row."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._reliability_state_table}
            WHERE service_name = $1 AND component_name = $2
            """,
            service_name,
            component_name,
        )
        if row is None:
            return None
        return _reliability_state_from_row(row)

    async def load_latest_recovery_event(self) -> RecoveryEvent | None:
        """Load the newest recovery or reliability event across services."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._reliability_events_table}
            ORDER BY event_time DESC, id DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        return _recovery_event_from_row(row)

    async def load_feature_lag_states(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> list[FeatureLagSnapshot]:
        """Load the latest persisted feature lag rows for one component."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._reliability_lag_table}
            WHERE service_name = $1 AND component_name = $2
            ORDER BY symbol ASC
            """,
            service_name,
            component_name,
        )
        return [_lag_state_from_row(row) for row in rows]

    async def save_system_reliability_state(
        self,
        snapshot: SystemReliabilitySnapshot,
    ) -> None:
        """Upsert the latest canonical cross-service reliability summary."""
        pool = self._require_pool()
        lag_symbols = [
            lag_snapshot.symbol
            for lag_snapshot in snapshot.lag_by_symbol
            if lag_snapshot.lag_breach
        ]
        detail = json.dumps(
            make_json_safe(
                {
                    "reason_codes": list(snapshot.reason_codes),
                    "service_statuses": {
                        service_snapshot.component_name: service_snapshot.health_overall_status
                        for service_snapshot in snapshot.services
                    },
                    "lag_symbols": lag_symbols,
                }
            ),
            sort_keys=True,
        )
        latest_recovery_event = snapshot.latest_recovery_event
        await pool.execute(
            f"""
            INSERT INTO {self._reliability_system_table} (
                service_name,
                checked_at,
                health_overall_status,
                reason_codes,
                lag_breach_active,
                latest_recovery_event_type,
                latest_recovery_event_time,
                latest_recovery_reason_code,
                details,
                updated_at
            ) VALUES (
                $1, $2, $3, $4::text[], $5, $6, $7, $8, $9, NOW()
            )
            ON CONFLICT (service_name)
            DO UPDATE SET
                checked_at = EXCLUDED.checked_at,
                health_overall_status = EXCLUDED.health_overall_status,
                reason_codes = EXCLUDED.reason_codes,
                lag_breach_active = EXCLUDED.lag_breach_active,
                latest_recovery_event_type = EXCLUDED.latest_recovery_event_type,
                latest_recovery_event_time = EXCLUDED.latest_recovery_event_time,
                latest_recovery_reason_code = EXCLUDED.latest_recovery_reason_code,
                details = EXCLUDED.details,
                updated_at = NOW()
            """,
            snapshot.service_name,
            snapshot.checked_at,
            snapshot.health_overall_status,
            list(snapshot.reason_codes),
            snapshot.lag_breach_active,
            None if latest_recovery_event is None else latest_recovery_event.event_type,
            None if latest_recovery_event is None else latest_recovery_event.event_time,
            None if latest_recovery_event is None else latest_recovery_event.reason_code,
            detail,
        )

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._heartbeats_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    heartbeat_at TIMESTAMPTZ NOT NULL,
                    health_overall_status TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    details TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_state_table} (
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    health_overall_status TEXT NOT NULL,
                    freshness_status TEXT NULL,
                    breaker_state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_heartbeat_at TIMESTAMPTZ NULL,
                    last_success_at TIMESTAMPTZ NULL,
                    last_failure_at TIMESTAMPTZ NULL,
                    opened_at TIMESTAMPTZ NULL,
                    reason_code TEXT NULL,
                    details TEXT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, component_name)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_events_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    reason_code TEXT NOT NULL,
                    health_overall_status TEXT NULL,
                    freshness_status TEXT NULL,
                    breaker_state TEXT NULL,
                    details TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_lag_table} (
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    latest_raw_event_received_at TIMESTAMPTZ NULL,
                    latest_feature_interval_begin TIMESTAMPTZ NULL,
                    latest_feature_as_of_time TIMESTAMPTZ NULL,
                    time_lag_seconds DOUBLE PRECISION NULL,
                    processing_lag_seconds DOUBLE PRECISION NULL,
                    time_lag_reason_code TEXT NOT NULL,
                    processing_lag_reason_code TEXT NOT NULL,
                    lag_breach BOOLEAN NOT NULL DEFAULT FALSE,
                    health_overall_status TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    details TEXT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, component_name, symbol)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_system_table} (
                    service_name TEXT NOT NULL,
                    checked_at TIMESTAMPTZ NOT NULL,
                    health_overall_status TEXT NOT NULL,
                    reason_codes TEXT[] NOT NULL DEFAULT '{{}}',
                    lag_breach_active BOOLEAN NOT NULL DEFAULT FALSE,
                    latest_recovery_event_type TEXT NULL,
                    latest_recovery_event_time TIMESTAMPTZ NULL,
                    latest_recovery_reason_code TEXT NULL,
                    details TEXT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name)
                )
                """
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("ReliabilityStore has not been connected")
        return self._pool


def _heartbeat_from_row(row: asyncpg.Record) -> ServiceHeartbeat:
    return ServiceHeartbeat(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        heartbeat_at=row["heartbeat_at"],
        health_overall_status=str(row["health_overall_status"]),
        reason_code=str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
        heartbeat_id=int(row["id"]),
        created_at=row["created_at"],
    )


def _reliability_state_from_row(row: asyncpg.Record) -> ReliabilityState:
    return ReliabilityState(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        health_overall_status=str(row["health_overall_status"]),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=str(row["breaker_state"]),
        failure_count=int(row["failure_count"]),
        success_count=int(row["success_count"]),
        last_heartbeat_at=row["last_heartbeat_at"],
        last_success_at=row["last_success_at"],
        last_failure_at=row["last_failure_at"],
        opened_at=row["opened_at"],
        reason_code=None if row["reason_code"] is None else str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
        updated_at=row["updated_at"],
    )


def _recovery_event_from_row(row: asyncpg.Record) -> RecoveryEvent:
    return RecoveryEvent(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        event_type=str(row["event_type"]),
        event_time=row["event_time"],
        reason_code=str(row["reason_code"]),
        health_overall_status=(
            None
            if row["health_overall_status"] is None
            else str(row["health_overall_status"])
        ),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=(
            None if row["breaker_state"] is None else str(row["breaker_state"])
        ),
        detail=None if row["details"] is None else str(row["details"]),
        event_id=int(row["id"]),
        created_at=row["created_at"],
    )


def _lag_state_from_row(row: asyncpg.Record) -> FeatureLagSnapshot:
    return FeatureLagSnapshot(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        symbol=str(row["symbol"]),
        evaluated_at=row["updated_at"],
        latest_raw_event_received_at=row["latest_raw_event_received_at"],
        latest_feature_interval_begin=row["latest_feature_interval_begin"],
        latest_feature_as_of_time=row["latest_feature_as_of_time"],
        time_lag_seconds=(
            None if row["time_lag_seconds"] is None else float(row["time_lag_seconds"])
        ),
        processing_lag_seconds=(
            None
            if row["processing_lag_seconds"] is None
            else float(row["processing_lag_seconds"])
        ),
        time_lag_reason_code=str(row["time_lag_reason_code"]),
        processing_lag_reason_code=str(row["processing_lag_reason_code"]),
        lag_breach=bool(row["lag_breach"]),
        health_overall_status=str(row["health_overall_status"]),
        reason_code=str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
    )
