"""Read-only PostgreSQL access helpers for M4 inference."""

# pylint: disable=duplicate-code

from __future__ import annotations

import re
from typing import Any

import asyncpg


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


class DatabaseUnavailableError(RuntimeError):
    """Raised when the inference service cannot reach PostgreSQL."""


class InferenceDatabase:
    """Read-only feature row access for the inference API."""

    def __init__(self, dsn: str, feature_table: str) -> None:
        self._dsn = dsn
        self._feature_table = _quote_table_name(feature_table)
        self._pool: asyncpg.Pool | Any | None = None

    async def connect(self) -> None:
        """Create the database pool if it is not already open."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        """Close the database pool if one is open."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def is_healthy(self) -> bool:
        """Return whether PostgreSQL is currently reachable."""
        try:
            pool = await self._require_pool()
            value = await pool.fetchval("SELECT 1")
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        return value == 1

    async def fetch_latest_feature_row(
        self,
        *,
        symbol: str,
        interval_minutes: int,
    ) -> dict[str, Any] | None:
        """Fetch the latest canonical feature row for one symbol."""
        try:
            pool = await self._require_pool()
            row = await pool.fetchrow(
                f"""
                SELECT *
                FROM {self._feature_table}
                WHERE source_exchange = 'kraken'
                  AND symbol = $1
                  AND interval_minutes = $2
                ORDER BY as_of_time DESC, interval_begin DESC
                LIMIT 1
                """,
                symbol,
                interval_minutes,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise DatabaseUnavailableError(f"Could not query PostgreSQL: {error}") from error
        return None if row is None else dict(row)

    async def _require_pool(self) -> asyncpg.Pool | Any:
        if self._pool is None:
            await self.connect()
        if self._pool is None:
            raise DatabaseUnavailableError("PostgreSQL pool is not available")
        return self._pool
