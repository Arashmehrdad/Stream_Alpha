"""Read-only PostgreSQL access helpers for M4 inference."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

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
        last_error: Exception | None = None
        for dsn_candidate in _dsn_candidates(self._dsn):
            try:
                self._pool = await asyncpg.create_pool(
                    dsn_candidate,
                    min_size=1,
                    max_size=5,
                )
            except (OSError, asyncpg.PostgresConnectionError) as error:
                last_error = error
                continue
            return
        if last_error is not None:
            raise DatabaseUnavailableError(
                f"Could not connect to PostgreSQL: {last_error}"
            ) from last_error
        raise DatabaseUnavailableError("Could not connect to PostgreSQL")

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
        interval_begin: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the latest canonical feature row for one symbol."""
        try:
            pool = await self._require_pool()
            if interval_begin is None:
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
            else:
                row = await pool.fetchrow(
                    f"""
                    SELECT *
                    FROM {self._feature_table}
                    WHERE source_exchange = 'kraken'
                      AND symbol = $1
                      AND interval_minutes = $2
                      AND interval_begin = $3
                    ORDER BY as_of_time DESC, interval_begin DESC
                    LIMIT 1
                    """,
                    symbol,
                    interval_minutes,
                    interval_begin,
                )
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise DatabaseUnavailableError(f"Could not query PostgreSQL: {error}") from error
        return None if row is None else dict(row)

    async def fetch_feature_reference_vector(
        self,
        *,
        feature_names: Sequence[str],
        interval_minutes: int,
    ) -> dict[str, float]:
        """Fetch deterministic median reference values for explainable numeric features."""
        requested_features = tuple(feature_names)
        if not requested_features:
            return {}

        select_columns = ", ".join(
            (
                "percentile_cont(0.5) WITHIN GROUP (ORDER BY "
                f"{_quote_identifier(feature_name)}) AS {_quote_identifier(feature_name)}"
            )
            for feature_name in requested_features
        )
        try:
            pool = await self._require_pool()
            row = await pool.fetchrow(
                f"""
                SELECT {select_columns}
                FROM {self._feature_table}
                WHERE source_exchange = 'kraken'
                  AND interval_minutes = $1
                """,
                interval_minutes,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise DatabaseUnavailableError(f"Could not query PostgreSQL: {error}") from error

        if row is None:
            return {}
        return {
            feature_name: float(row[feature_name])
            for feature_name in requested_features
            if row[feature_name] is not None
        }

    async def _require_pool(self) -> asyncpg.Pool | Any:
        if self._pool is None:
            await self.connect()
        if self._pool is None:
            raise DatabaseUnavailableError("PostgreSQL pool is not available")
        return self._pool


def _dsn_candidates(dsn: str) -> tuple[str, ...]:
    parsed = urlsplit(dsn)
    host = parsed.hostname
    if host in {None, "127.0.0.1", "localhost"}:
        return (dsn,)

    localhost_netloc = parsed.netloc.replace(host, "127.0.0.1", 1)
    localhost_dsn = urlunsplit(
        (parsed.scheme, localhost_netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return (dsn, localhost_dsn)
