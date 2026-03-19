"""PostgreSQL persistence layer for normalized raw events."""

from __future__ import annotations

import json
import re

import asyncpg

from app.common.config import TableSettings
from app.common.models import HealthEvent, OhlcEvent, TradeEvent
from app.common.serialization import model_to_dict


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


def _extract_schema(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) == 2:
        return parts[0]
    return None


class PostgresWriter:
    """Create required tables and persist normalized producer records."""

    def __init__(self, dsn: str, tables: TableSettings) -> None:
        self._dsn = dsn
        self._tables = tables
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the connection pool and ensure required tables exist."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        await self._ensure_schema()

    async def close(self) -> None:
        """Close the connection pool if it is open."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        schema_names = {
            schema
            for schema in (
                _extract_schema(self._tables.raw_trades),
                _extract_schema(self._tables.raw_ohlc),
                _extract_schema(self._tables.producer_heartbeat),
            )
            if schema
        }
        async with pool.acquire() as connection:
            for schema in schema_names:
                await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(schema)}")

            raw_trades_table = _quote_table_name(self._tables.raw_trades)
            raw_ohlc_table = _quote_table_name(self._tables.raw_ohlc)
            heartbeat_table = _quote_table_name(self._tables.producer_heartbeat)

            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {raw_trades_table} (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    source_exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_id BIGINT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (source_exchange, symbol, trade_id)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {raw_ohlc_table} (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    source_exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval_minutes INTEGER NOT NULL,
                    interval_begin TIMESTAMPTZ NOT NULL,
                    interval_end TIMESTAMPTZ NOT NULL,
                    open_price DOUBLE PRECISION NOT NULL,
                    high_price DOUBLE PRECISION NOT NULL,
                    low_price DOUBLE PRECISION NOT NULL,
                    close_price DOUBLE PRECISION NOT NULL,
                    vwap DOUBLE PRECISION NOT NULL,
                    trade_count INTEGER NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (source_exchange, symbol, interval_minutes, interval_begin)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {heartbeat_table} (
                    service_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source_exchange TEXT NULL,
                    details JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    last_event_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    async def write_trade(self, event: TradeEvent) -> None:
        """Upsert a normalized trade row."""
        pool = self._require_pool()
        table_name = _quote_table_name(self._tables.raw_trades)
        payload_json = json.dumps(model_to_dict(event))
        await pool.execute(
            f"""
            INSERT INTO {table_name} (
                event_id,
                source_exchange,
                symbol,
                trade_id,
                side,
                order_type,
                price,
                quantity,
                event_time,
                received_at,
                payload
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb
            )
            ON CONFLICT (source_exchange, symbol, trade_id)
            DO UPDATE SET
                event_id = EXCLUDED.event_id,
                side = EXCLUDED.side,
                order_type = EXCLUDED.order_type,
                price = EXCLUDED.price,
                quantity = EXCLUDED.quantity,
                event_time = EXCLUDED.event_time,
                received_at = EXCLUDED.received_at,
                payload = EXCLUDED.payload
            """,
            event.event_id,
            event.source_exchange,
            event.symbol,
            event.trade_id,
            event.side,
            event.order_type,
            event.price,
            event.quantity,
            event.event_time,
            event.received_at,
            payload_json,
        )

    async def write_ohlc(self, event: OhlcEvent) -> None:
        """Upsert a normalized OHLC row."""
        pool = self._require_pool()
        table_name = _quote_table_name(self._tables.raw_ohlc)
        payload_json = json.dumps(model_to_dict(event))
        await pool.execute(
            f"""
            INSERT INTO {table_name} (
                event_id,
                source_exchange,
                symbol,
                interval_minutes,
                interval_begin,
                interval_end,
                open_price,
                high_price,
                low_price,
                close_price,
                vwap,
                trade_count,
                volume,
                received_at,
                payload
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::jsonb
            )
            ON CONFLICT (source_exchange, symbol, interval_minutes, interval_begin)
            DO UPDATE SET
                event_id = EXCLUDED.event_id,
                interval_end = EXCLUDED.interval_end,
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                vwap = EXCLUDED.vwap,
                trade_count = EXCLUDED.trade_count,
                volume = EXCLUDED.volume,
                received_at = EXCLUDED.received_at,
                payload = EXCLUDED.payload
            """,
            event.event_id,
            event.source_exchange,
            event.symbol,
            event.interval_minutes,
            event.interval_begin,
            event.interval_end,
            event.open_price,
            event.high_price,
            event.low_price,
            event.close_price,
            event.vwap,
            event.trade_count,
            event.volume,
            event.received_at,
            payload_json,
        )

    async def write_heartbeat(self, event: HealthEvent) -> None:
        """Upsert the latest producer heartbeat row."""
        pool = self._require_pool()
        table_name = _quote_table_name(self._tables.producer_heartbeat)
        details_json = json.dumps(model_to_dict(event)["details"])
        await pool.execute(
            f"""
            INSERT INTO {table_name} (
                service_name,
                status,
                component,
                message,
                source_exchange,
                details,
                last_event_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7, NOW()
            )
            ON CONFLICT (service_name)
            DO UPDATE SET
                status = EXCLUDED.status,
                component = EXCLUDED.component,
                message = EXCLUDED.message,
                source_exchange = EXCLUDED.source_exchange,
                details = EXCLUDED.details,
                last_event_at = EXCLUDED.last_event_at,
                updated_at = NOW()
            """,
            event.service_name,
            event.status,
            event.component,
            event.message,
            event.source_exchange,
            details_json,
            event.observed_at,
        )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresWriter has not been connected")
        return self._pool
