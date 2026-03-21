"""Tests for the read-only inference DB access layer."""

# pylint: disable=too-few-public-methods

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.inference.db import InferenceDatabase


class FakePool:
    """Simple asyncpg pool stub for DB query inspection."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row
        self.query: str | None = None
        self.args: tuple[Any, ...] | None = None

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Return the configured row and capture the executed query."""
        self.query = query
        self.args = args
        return self.row

    async def fetchval(self, query: str) -> int:
        """Return a healthy scalar value for health checks."""
        self.query = query
        return 1

    async def close(self) -> None:
        """Close the fake pool."""
        return None


def test_fetch_latest_feature_row_uses_expected_ordering_and_filters() -> None:
    """The latest-row query should filter by symbol/interval and order newest first."""
    database = InferenceDatabase("postgresql://ignored", "feature_ohlc")
    expected_row = {
        "id": 1,
        "symbol": "BTC/USD",
        "interval_minutes": 5,
        "interval_begin": datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc),
        "as_of_time": datetime(2026, 3, 19, 22, 5, tzinfo=timezone.utc),
    }
    fake_pool = FakePool(expected_row)
    database._pool = fake_pool  # pylint: disable=protected-access

    row = asyncio.run(
        database.fetch_latest_feature_row(symbol="BTC/USD", interval_minutes=5),
    )

    assert row == expected_row
    assert fake_pool.args == ("BTC/USD", 5)
    assert "source_exchange = 'kraken'" in (fake_pool.query or "")
    assert "ORDER BY as_of_time DESC, interval_begin DESC" in (fake_pool.query or "")
    assert "LIMIT 1" in (fake_pool.query or "")


def test_fetch_latest_feature_row_accepts_exact_interval_begin() -> None:
    """The DB layer should allow exact-candle lookups for restart-safe consumers."""
    database = InferenceDatabase("postgresql://ignored", "feature_ohlc")
    interval_begin = datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc)
    fake_pool = FakePool({"id": 1, "interval_begin": interval_begin})
    database._pool = fake_pool  # pylint: disable=protected-access

    row = asyncio.run(
        database.fetch_latest_feature_row(
            symbol="BTC/USD",
            interval_minutes=5,
            interval_begin=interval_begin,
        ),
    )

    assert row == {"id": 1, "interval_begin": interval_begin}
    assert fake_pool.args == ("BTC/USD", 5, interval_begin)
    assert "AND interval_begin = $3" in (fake_pool.query or "")


def test_fetch_feature_reference_vector_queries_medians_for_requested_columns() -> None:
    """The reference-vector query should compute deterministic medians per feature."""
    database = InferenceDatabase("postgresql://ignored", "feature_ohlc")
    fake_pool = FakePool(
        {
            "close_price": 70250.0,
            "momentum_3": 0.015,
        }
    )
    database._pool = fake_pool  # pylint: disable=protected-access

    reference_values = asyncio.run(
        database.fetch_feature_reference_vector(
            feature_names=("close_price", "momentum_3"),
            interval_minutes=5,
        )
    )

    assert reference_values == {
        "close_price": 70250.0,
        "momentum_3": 0.015,
    }
    assert fake_pool.args == (5,)
    assert "percentile_cont(0.5)" in (fake_pool.query or "")
    assert '"close_price"' in (fake_pool.query or "")
    assert '"momentum_3"' in (fake_pool.query or "")
