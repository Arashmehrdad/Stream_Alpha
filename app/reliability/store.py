"""Small PostgreSQL reliability store for non-trading services."""

from __future__ import annotations

import re

import asyncpg

from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat


HEARTBEATS_TABLE = "service_heartbeats"
RELIABILITY_STATE_TABLE = "reliability_state"
RELIABILITY_EVENTS_TABLE = "reliability_events"

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

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("ReliabilityStore has not been connected")
        return self._pool
