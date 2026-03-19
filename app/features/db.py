"""PostgreSQL persistence and bootstrap support for M2 feature rows."""

# pylint: disable=duplicate-code

from __future__ import annotations

import re

import asyncpg

from app.common.config import TableSettings
from app.common.models import OhlcEvent
from app.features.models import FeatureOhlcRow, deserialize_ohlc_event


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
    return parts[0] if len(parts) == 2 else None


def _build_index_name(table_name: str, suffix: str) -> str:
    index_name = f"{'_'.join(table_name.split('.'))}_{suffix}"
    return _quote_identifier(index_name)


class FeatureStore:
    """Create, bootstrap, and upsert finalized OHLC feature rows."""

    def __init__(self, dsn: str, tables: TableSettings) -> None:
        self._dsn = dsn
        self._tables = tables
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the connection pool and ensure feature storage exists."""
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

    async def load_bootstrap_candles(self, limit_per_symbol: int) -> list[OhlcEvent]:
        """Load the latest raw OHLC candles per symbol for state rebuilds."""
        pool = self._require_pool()
        raw_ohlc_table = _quote_table_name(self._tables.raw_ohlc)
        try:
            rows = await pool.fetch(
                f"""
                WITH ranked AS (
                    SELECT
                        payload::text AS payload_text,
                        source_exchange,
                        symbol,
                        interval_minutes,
                        interval_begin,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_exchange, symbol, interval_minutes
                            ORDER BY interval_begin DESC
                        ) AS row_number
                    FROM {raw_ohlc_table}
                )
                SELECT payload_text
                FROM ranked
                WHERE row_number <= $1
                ORDER BY source_exchange ASC, symbol ASC, interval_minutes ASC, interval_begin ASC
                """,
                limit_per_symbol,
            )
        except (asyncpg.InvalidSchemaNameError, asyncpg.UndefinedTableError):
            return []

        return [deserialize_ohlc_event(row["payload_text"]) for row in rows]

    async def upsert_feature_row(self, row: FeatureOhlcRow) -> None:
        """Upsert one finalized feature row into PostgreSQL."""
        pool = self._require_pool()
        table_name = _quote_table_name(self._tables.feature_ohlc)
        await pool.execute(
            f"""
            INSERT INTO {table_name} (
                source_exchange,
                symbol,
                interval_minutes,
                interval_begin,
                interval_end,
                as_of_time,
                computed_at,
                raw_event_id,
                open_price,
                high_price,
                low_price,
                close_price,
                vwap,
                trade_count,
                volume,
                log_return_1,
                log_return_3,
                momentum_3,
                return_mean_12,
                return_std_12,
                realized_vol_12,
                rsi_14,
                macd_line_12_26,
                volume_mean_12,
                volume_std_12,
                volume_zscore_12,
                close_zscore_12,
                lag_log_return_1,
                lag_log_return_2,
                lag_log_return_3,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, NOW()
            )
            ON CONFLICT (source_exchange, symbol, interval_minutes, interval_begin)
            DO UPDATE SET
                interval_end = EXCLUDED.interval_end,
                as_of_time = EXCLUDED.as_of_time,
                computed_at = EXCLUDED.computed_at,
                raw_event_id = EXCLUDED.raw_event_id,
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                vwap = EXCLUDED.vwap,
                trade_count = EXCLUDED.trade_count,
                volume = EXCLUDED.volume,
                log_return_1 = EXCLUDED.log_return_1,
                log_return_3 = EXCLUDED.log_return_3,
                momentum_3 = EXCLUDED.momentum_3,
                return_mean_12 = EXCLUDED.return_mean_12,
                return_std_12 = EXCLUDED.return_std_12,
                realized_vol_12 = EXCLUDED.realized_vol_12,
                rsi_14 = EXCLUDED.rsi_14,
                macd_line_12_26 = EXCLUDED.macd_line_12_26,
                volume_mean_12 = EXCLUDED.volume_mean_12,
                volume_std_12 = EXCLUDED.volume_std_12,
                volume_zscore_12 = EXCLUDED.volume_zscore_12,
                close_zscore_12 = EXCLUDED.close_zscore_12,
                lag_log_return_1 = EXCLUDED.lag_log_return_1,
                lag_log_return_2 = EXCLUDED.lag_log_return_2,
                lag_log_return_3 = EXCLUDED.lag_log_return_3,
                updated_at = NOW()
            """,
            row.source_exchange,
            row.symbol,
            row.interval_minutes,
            row.interval_begin,
            row.interval_end,
            row.as_of_time,
            row.computed_at,
            row.raw_event_id,
            row.open_price,
            row.high_price,
            row.low_price,
            row.close_price,
            row.vwap,
            row.trade_count,
            row.volume,
            row.log_return_1,
            row.log_return_3,
            row.momentum_3,
            row.return_mean_12,
            row.return_std_12,
            row.realized_vol_12,
            row.rsi_14,
            row.macd_line_12_26,
            row.volume_mean_12,
            row.volume_std_12,
            row.volume_zscore_12,
            row.close_zscore_12,
            row.lag_log_return_1,
            row.lag_log_return_2,
            row.lag_log_return_3,
        )

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        feature_schema = _extract_schema(self._tables.feature_ohlc)
        if feature_schema is not None:
            async with pool.acquire() as connection:
                await connection.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {_quote_identifier(feature_schema)}"
                )

        feature_table = _quote_table_name(self._tables.feature_ohlc)
        as_of_index = _build_index_name(
            self._tables.feature_ohlc,
            "symbol_as_of_time_desc_idx",
        )
        interval_index = _build_index_name(
            self._tables.feature_ohlc,
            "symbol_interval_begin_desc_idx",
        )

        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {feature_table} (
                    id BIGSERIAL PRIMARY KEY,
                    source_exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval_minutes INTEGER NOT NULL,
                    interval_begin TIMESTAMPTZ NOT NULL,
                    interval_end TIMESTAMPTZ NOT NULL,
                    as_of_time TIMESTAMPTZ NOT NULL,
                    computed_at TIMESTAMPTZ NOT NULL,
                    raw_event_id TEXT NOT NULL,
                    open_price DOUBLE PRECISION NOT NULL,
                    high_price DOUBLE PRECISION NOT NULL,
                    low_price DOUBLE PRECISION NOT NULL,
                    close_price DOUBLE PRECISION NOT NULL,
                    vwap DOUBLE PRECISION NOT NULL,
                    trade_count INTEGER NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    log_return_1 DOUBLE PRECISION NOT NULL,
                    log_return_3 DOUBLE PRECISION NOT NULL,
                    momentum_3 DOUBLE PRECISION NOT NULL,
                    return_mean_12 DOUBLE PRECISION NOT NULL,
                    return_std_12 DOUBLE PRECISION NOT NULL,
                    realized_vol_12 DOUBLE PRECISION NOT NULL,
                    rsi_14 DOUBLE PRECISION NOT NULL,
                    macd_line_12_26 DOUBLE PRECISION NOT NULL,
                    volume_mean_12 DOUBLE PRECISION NOT NULL,
                    volume_std_12 DOUBLE PRECISION NOT NULL,
                    volume_zscore_12 DOUBLE PRECISION NOT NULL,
                    close_zscore_12 DOUBLE PRECISION NOT NULL,
                    lag_log_return_1 DOUBLE PRECISION NOT NULL,
                    lag_log_return_2 DOUBLE PRECISION NOT NULL,
                    lag_log_return_3 DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (source_exchange, symbol, interval_minutes, interval_begin)
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {as_of_index}
                ON {feature_table} (symbol, as_of_time DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {interval_index}
                ON {feature_table} (symbol, interval_begin DESC)
                """
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("FeatureStore has not been connected")
        return self._pool
