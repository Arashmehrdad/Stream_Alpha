"""Canonical feature-row loading for the M8 offline regime workflow."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from app.common.config import PostgresSettings, Settings
from app.regime.config import RegimeConfig


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

REQUIRED_SOURCE_COLUMNS = (
    "source_exchange",
    "symbol",
    "interval_minutes",
    "interval_begin",
    "as_of_time",
    "realized_vol_12",
    "momentum_3",
    "macd_line_12_26",
)


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _quote_table_name(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {name}")
    return ".".join(_quote_identifier(part) for part in parts)


@dataclass(frozen=True, slots=True)
class RegimeSourceRow:
    """One canonical feature row used for threshold fitting and regime labeling."""

    symbol: str
    interval_begin: datetime
    as_of_time: datetime
    realized_vol_12: float
    momentum_3: float
    macd_line_12_26: float


@dataclass(frozen=True, slots=True)
class RegimeDataset:
    """Loaded canonical rows plus lightweight source metadata."""

    rows: tuple[RegimeSourceRow, ...]
    source_schema: tuple[str, ...]
    row_counts_by_symbol: dict[str, int]


def load_regime_dataset(config: RegimeConfig) -> RegimeDataset:
    """Load the configured canonical feature rows from PostgreSQL."""
    settings = Settings.from_env()
    return asyncio.run(_load_regime_dataset_with_fallback(settings.postgres, config))


async def _load_regime_dataset_with_fallback(
    postgres: PostgresSettings,
    config: RegimeConfig,
) -> RegimeDataset:
    last_error: Exception | None = None
    for dsn in _candidate_dsns(postgres):
        try:
            return await _load_regime_dataset(dsn, config)
        except (OSError, asyncpg.PostgresConnectionError) as error:
            last_error = error
            continue
    if last_error is None:
        raise ValueError("No PostgreSQL DSN candidates were available for regime loading")
    raise ValueError(f"Could not connect to PostgreSQL for regime loading: {last_error}") from last_error


async def _load_regime_dataset(dsn: str, config: RegimeConfig) -> RegimeDataset:
    connection = await asyncpg.connect(dsn)
    try:
        source_schema = await _fetch_source_schema(connection, config.source_table)
        _validate_source_columns(source_schema, REQUIRED_SOURCE_COLUMNS)
        source_rows = await _fetch_source_rows(connection, config)
    finally:
        await connection.close()

    rows = tuple(_coerce_source_rows(source_rows))
    row_counts_by_symbol = _row_counts_by_symbol(rows, config.symbols)
    _validate_symbol_row_counts(
        row_counts_by_symbol,
        min_rows_per_symbol=config.min_rows_per_symbol,
        symbols=config.symbols,
    )
    return RegimeDataset(
        rows=rows,
        source_schema=tuple(source_schema),
        row_counts_by_symbol=row_counts_by_symbol,
    )


async def _fetch_source_schema(connection: asyncpg.Connection, table_name: str) -> list[str]:
    table_parts = table_name.split(".")
    schema_name = table_parts[0] if len(table_parts) == 2 else "public"
    relation_name = table_parts[-1]
    rows = await connection.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
        """,
        schema_name,
        relation_name,
    )
    if not rows:
        raise ValueError(f"Source table {table_name} was not found in PostgreSQL")
    return [str(row["column_name"]) for row in rows]


async def _fetch_source_rows(
    connection: asyncpg.Connection,
    config: RegimeConfig,
) -> list[dict[str, Any]]:
    table_name = _quote_table_name(config.source_table)
    rows = await connection.fetch(
        f"""
        SELECT
            symbol,
            interval_begin,
            as_of_time,
            realized_vol_12,
            momentum_3,
            macd_line_12_26
        FROM {table_name}
        WHERE source_exchange = $1
          AND interval_minutes = $2
          AND symbol = ANY($3::text[])
        ORDER BY symbol ASC, interval_begin ASC, as_of_time ASC
        """,
        config.source_exchange,
        config.interval_minutes,
        list(config.symbols),
    )
    return [dict(row) for row in rows]


def _candidate_dsns(postgres: PostgresSettings) -> tuple[str, ...]:
    primary_dsn = postgres.dsn
    if postgres.host in {"127.0.0.1", "localhost"}:
        return (primary_dsn,)
    localhost_dsn = PostgresSettings(
        host="127.0.0.1",
        port=postgres.port,
        database=postgres.database,
        user=postgres.user,
        password=postgres.password,
    ).dsn
    return (primary_dsn, localhost_dsn)


def _validate_source_columns(
    source_schema: list[str] | tuple[str, ...],
    required_columns: tuple[str, ...],
) -> None:
    missing_columns = sorted(set(required_columns) - set(source_schema))
    if missing_columns:
        raise ValueError(
            "Regime source schema does not include the required canonical columns. "
            f"Missing columns: {missing_columns}"
        )


def _coerce_source_rows(source_rows: list[dict[str, Any]]) -> list[RegimeSourceRow]:
    coerced_rows: list[RegimeSourceRow] = []
    for row in source_rows:
        missing_values = [
            column
            for column in (
                "symbol",
                "interval_begin",
                "as_of_time",
                "realized_vol_12",
                "momentum_3",
                "macd_line_12_26",
            )
            if row.get(column) is None
        ]
        if missing_values:
            raise ValueError(
                "Regime source rows contain nulls in required fields. "
                f"Missing values: {missing_values}"
            )
        coerced_rows.append(
            RegimeSourceRow(
                symbol=str(row["symbol"]),
                interval_begin=row["interval_begin"],
                as_of_time=row["as_of_time"],
                realized_vol_12=float(row["realized_vol_12"]),
                momentum_3=float(row["momentum_3"]),
                macd_line_12_26=float(row["macd_line_12_26"]),
            )
        )
    return coerced_rows


def _row_counts_by_symbol(
    rows: tuple[RegimeSourceRow, ...],
    symbols: tuple[str, ...],
) -> dict[str, int]:
    counts = {symbol: 0 for symbol in symbols}
    for row in rows:
        counts[row.symbol] = counts.get(row.symbol, 0) + 1
    return counts


def _validate_symbol_row_counts(
    row_counts_by_symbol: dict[str, int],
    *,
    min_rows_per_symbol: int,
    symbols: tuple[str, ...],
) -> None:
    insufficient = [
        f"{symbol} ({row_counts_by_symbol.get(symbol, 0)} found)"
        for symbol in symbols
        if row_counts_by_symbol.get(symbol, 0) < min_rows_per_symbol
    ]
    if insufficient:
        joined = ", ".join(insufficient)
        raise ValueError(
            "Regime source does not contain enough canonical rows per symbol. "
            f"Required at least {min_rows_per_symbol} rows for each symbol; {joined}"
        )
