"""Backfill recent Kraken REST OHLC history into the existing canonical flow."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import asyncpg

from app.common.config import PostgresSettings, Settings
from app.common.logging import configure_logging
from app.common.models import OhlcEvent, generate_event_id
from app.common.time import utc_now
from app.features.db import FeatureStore
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.state import FeatureStateManager
from app.ingestion.db import PostgresWriter


_HTTP_TIMEOUT_SECONDS = 30


def _build_argument_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill recent Kraken REST OHLC history into raw_ohlc and feature_ohlc.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(settings.kraken.symbols),
        help="Symbols to backfill using Kraken REST OHLC.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=settings.kraken.ohlc_interval_minutes,
        help="OHLC interval in minutes.",
    )
    parser.add_argument(
        "--lookback-candles",
        type=int,
        default=128,
        help="Number of closed candles per symbol to keep from the REST response.",
    )
    return parser


def _fetch_kraken_rest_rows(
    rest_ohlc_url: str,
    *,
    symbol: str,
    interval_minutes: int,
) -> list[list[Any]]:
    query_string = urlencode({"pair": symbol, "interval": interval_minutes})
    request_url = f"{rest_ohlc_url}?{query_string}"
    with urlopen(request_url, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    errors = payload.get("error", [])
    if errors:
        raise ValueError(f"Kraken REST returned errors for {symbol}: {errors}")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Kraken REST returned an invalid OHLC payload for {symbol}")

    series_key = next((key for key in result if key != "last"), None)
    if series_key is None:
        raise ValueError(f"Kraken REST returned no OHLC series for {symbol}")

    rows = result.get(series_key)
    if not isinstance(rows, list):
        raise ValueError(f"Kraken REST OHLC series for {symbol} is not a list")
    return rows


def _closed_rest_rows(
    rows: Sequence[Sequence[Any]],
    *,
    lookback_candles: int,
) -> list[list[Any]]:
    if lookback_candles <= 0:
        raise ValueError("lookback_candles must be positive")
    if len(rows) <= 1:
        return []

    closed_rows = [list(row) for row in rows[:-1]]
    deduplicated: dict[int, list[Any]] = {}
    for row in closed_rows:
        deduplicated[int(row[0])] = row
    ordered_rows = [deduplicated[key] for key in sorted(deduplicated)]
    return ordered_rows[-lookback_candles:]


def _event_from_rest_row(
    *,
    app_name: str,
    symbol: str,
    interval_minutes: int,
    row: Sequence[Any],
    received_at: datetime,
) -> OhlcEvent:
    interval_begin = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
    return OhlcEvent(
        event_id=generate_event_id(),
        app_name=app_name,
        source_exchange="kraken",
        channel="ohlc",
        message_type="backfill",
        symbol=symbol,
        interval_minutes=interval_minutes,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=interval_minutes),
        open_price=float(row[1]),
        high_price=float(row[2]),
        low_price=float(row[3]),
        close_price=float(row[4]),
        vwap=float(row[5]),
        trade_count=int(row[7]),
        volume=float(row[6]),
        received_at=received_at,
    )


async def _resolve_postgres_dsn(postgres: PostgresSettings) -> str:
    candidates = [postgres.dsn]
    if postgres.host not in {"127.0.0.1", "localhost"}:
        candidates.append(
            PostgresSettings(
                host="127.0.0.1",
                port=postgres.port,
                database=postgres.database,
                user=postgres.user,
                password=postgres.password,
            ).dsn
        )

    last_error: Exception | None = None
    for dsn in candidates:
        try:
            connection = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresConnectionError) as error:
            last_error = error
            continue
        await connection.close()
        return dsn

    if last_error is None:
        raise ValueError("No PostgreSQL DSN candidates were available for backfill")
    raise ValueError(f"Could not connect to PostgreSQL for backfill: {last_error}") from last_error


async def _regenerate_feature_rows(
    *,
    store: FeatureStore,
    symbols: Sequence[str],
    interval_minutes: int,
    history_limit: int,
) -> int:
    raw_events = await store.load_raw_candles(
        symbols=tuple(symbols),
        interval_minutes=interval_minutes,
    )
    computed_at = utc_now()
    state = FeatureStateManager(
        grace_seconds=0,
        history_limit=max(history_limit, MIN_FINALIZED_CANDLES),
    )
    feature_rows = state.bootstrap(
        raw_events,
        now=computed_at,
        computed_at=computed_at,
    )
    for row in feature_rows:
        await store.upsert_feature_row(row)
    return len(feature_rows)


async def _run_backfill(arguments: argparse.Namespace, settings: Settings) -> None:
    logger = logging.getLogger(f"{settings.app_name}.backfill")
    postgres_dsn = await _resolve_postgres_dsn(settings.postgres)
    writer = PostgresWriter(postgres_dsn, settings.tables)
    store = FeatureStore(postgres_dsn, settings.tables)
    await writer.connect()
    await store.connect()

    total_raw_rows = 0
    per_symbol_counts: dict[str, int] = {}
    try:
        for symbol in arguments.symbols:
            logger.info(
                "Fetching Kraken REST OHLC history",
                extra={
                    "symbol": symbol,
                    "interval_minutes": arguments.interval,
                    "lookback_candles": arguments.lookback_candles,
                },
            )
            rest_rows = _fetch_kraken_rest_rows(
                settings.kraken.rest_ohlc_url,
                symbol=symbol,
                interval_minutes=arguments.interval,
            )
            closed_rows = _closed_rest_rows(
                rest_rows,
                lookback_candles=arguments.lookback_candles,
            )
            received_at = utc_now()
            events = [
                _event_from_rest_row(
                    app_name=settings.app_name,
                    symbol=symbol,
                    interval_minutes=arguments.interval,
                    row=row,
                    received_at=received_at,
                )
                for row in closed_rows
            ]
            for event in events:
                await writer.write_ohlc(event)
            per_symbol_counts[symbol] = len(events)
            total_raw_rows += len(events)

        feature_rows = await _regenerate_feature_rows(
            store=store,
            symbols=arguments.symbols,
            interval_minutes=arguments.interval,
            history_limit=settings.features.bootstrap_candles,
        )
        logger.info(
            "Kraken REST OHLC backfill complete",
            extra={
                "raw_rows_upserted": total_raw_rows,
                "feature_rows_upserted": feature_rows,
                "symbols": per_symbol_counts,
                "integration_path": "raw_ohlc_then_feature_regeneration",
            },
        )
    finally:
        await writer.close()
        await store.close()


def main(argv: Iterable[str] | None = None) -> None:
    """Run the one-shot Kraken REST OHLC backfill command."""
    settings = Settings.from_env()
    parser = _build_argument_parser(settings)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(settings.log_level)
    asyncio.run(_run_backfill(arguments, settings))


if __name__ == "__main__":
    main()
