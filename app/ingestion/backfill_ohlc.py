"""Backfill historical Kraken OHLC history into the real raw -> feature path."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import asyncpg

from app.common.config import PostgresSettings, Settings
from app.common.logging import configure_logging
from app.common.models import OhlcEvent
from app.common.serialization import make_json_safe
from app.common.time import parse_rfc3339, to_rfc3339, utc_now
from app.features.db import FeatureStore
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.models import FeatureOhlcRow
from app.features.state import FeatureStateManager
from app.ingestion.db import PostgresWriter
from app.training.data_readiness import (
    build_data_readiness_report_from_path,
    default_data_readiness_artifact_root,
    write_data_readiness_artifacts,
)


_HTTP_TIMEOUT_SECONDS = 30
_DEFAULT_REQUEST_RETRIES = 3
_DEFAULT_TRAINING_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "training.m7.json"
)


@dataclass(frozen=True, slots=True)
class BackfillWindow:
    """One requested historical backfill window."""

    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "start": to_rfc3339(self.start),
            "end": to_rfc3339(self.end),
        }


@dataclass(frozen=True, slots=True)
class RawSyncStats:
    """Per-symbol raw OHLC synchronization counts for one run."""

    symbol: str
    fetched_rows: int
    created_rows: int
    updated_rows: int
    unchanged_rows: int
    exchange_window_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class FeatureReplayStats:
    """Feature replay counts for one requested window."""

    generated_rows: int
    created_rows: int
    updated_rows: int
    unchanged_rows: int
    skipped_outside_window_rows: int

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


def _build_argument_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill Kraken REST OHLC history into raw_ohlc and feature_ohlc.",
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
        help="OHLC interval in minutes. Must match the live pipeline interval.",
    )
    parser.add_argument(
        "--lookback-candles",
        type=int,
        default=128,
        help="Number of recent closed candles to request when --start/--end are omitted.",
    )
    parser.add_argument(
        "--start",
        help="Inclusive RFC3339 UTC start for historical backfill, for example 2026-04-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end",
        help="Exclusive RFC3339 UTC end for historical backfill, for example 2026-04-02T00:00:00Z.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=_DEFAULT_REQUEST_RETRIES,
        help="How many times to retry one Kraken REST request before failing.",
    )
    parser.add_argument(
        "--skip-raw-backfill",
        action="store_true",
        help="Skip Kraken REST ingestion and only replay features/report from existing raw_ohlc rows.",
    )
    parser.add_argument(
        "--skip-feature-replay",
        action="store_true",
        help="Skip feature replay and only backfill raw_ohlc plus the readiness report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip raw and feature writes and only persist the historical readiness report.",
    )
    parser.add_argument(
        "--training-config",
        default=str(_DEFAULT_TRAINING_CONFIG),
        help="Training config used for the readiness gate and sufficiency report.",
    )
    parser.add_argument(
        "--report-artifact-root",
        default=str(default_data_readiness_artifact_root()),
        help="Artifact root where the sufficiency report should be written.",
    )
    return parser


def _fetch_kraken_rest_rows(
    rest_ohlc_url: str,
    *,
    symbol: str,
    interval_minutes: int,
    since: int | None,
) -> tuple[list[list[Any]], int | None]:
    query: dict[str, Any] = {"pair": symbol, "interval": interval_minutes}
    if since is not None:
        query["since"] = since
    request_url = f"{rest_ohlc_url}?{urlencode(query)}"
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
    last_value = result.get("last")
    return rows, int(last_value) if last_value is not None else None


def _closed_rest_rows(
    rows: list[list[Any]],
    *,
    lookback_candles: int | None,
) -> list[list[Any]]:
    if lookback_candles is not None and lookback_candles <= 0:
        raise ValueError("lookback_candles must be positive when supplied")
    if len(rows) <= 1:
        return []

    closed_rows = [list(row) for row in rows[:-1]]
    deduplicated: dict[int, list[Any]] = {}
    for row in closed_rows:
        deduplicated[int(row[0])] = row
    ordered_rows = [deduplicated[key] for key in sorted(deduplicated)]
    if lookback_candles is None:
        return ordered_rows
    return ordered_rows[-lookback_candles:]


def _stable_backfill_event_id(
    *,
    symbol: str,
    interval_minutes: int,
    row: list[Any],
) -> str:
    payload = "|".join(
        [
            "kraken",
            symbol,
            str(interval_minutes),
            str(int(row[0])),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_from_rest_row(
    *,
    app_name: str,
    symbol: str,
    interval_minutes: int,
    row: list[Any],
    received_at: datetime,
) -> OhlcEvent:
    interval_begin = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
    return OhlcEvent(
        event_id=_stable_backfill_event_id(
            symbol=symbol,
            interval_minutes=interval_minutes,
            row=row,
        ),
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


def _resolve_backfill_window(
    *,
    arguments: argparse.Namespace,
    interval_minutes: int,
) -> BackfillWindow:
    if arguments.start:
        start = _floor_to_interval(parse_rfc3339(str(arguments.start)), interval_minutes)
        if arguments.end:
            end = _floor_to_interval(parse_rfc3339(str(arguments.end)), interval_minutes)
        else:
            end = _floor_to_interval(utc_now(), interval_minutes)
    elif arguments.end:
        end = _floor_to_interval(parse_rfc3339(str(arguments.end)), interval_minutes)
        start = end - timedelta(minutes=interval_minutes * arguments.lookback_candles)
    else:
        end = _floor_to_interval(utc_now(), interval_minutes)
        start = end - timedelta(minutes=interval_minutes * arguments.lookback_candles)

    if end <= start:
        raise ValueError("Historical backfill end must be later than start")
    return BackfillWindow(start=start, end=end)


def _floor_to_interval(timestamp: datetime, interval_minutes: int) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    utc_timestamp = timestamp.astimezone(timezone.utc)
    interval_seconds = interval_minutes * 60
    aligned_epoch = int(utc_timestamp.timestamp()) // interval_seconds * interval_seconds
    return datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)


def _row_in_window(row: list[Any], window: BackfillWindow) -> bool:
    interval_begin = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
    return window.start <= interval_begin < window.end


def _fetch_window_rows_with_retries(
    rest_ohlc_url: str,
    *,
    symbol: str,
    interval_minutes: int,
    window: BackfillWindow,
    request_retries: int,
) -> tuple[list[list[Any]], bool]:
    last_error: Exception | None = None
    for _attempt in range(request_retries):
        try:
            rest_rows, _last_token = _fetch_kraken_rest_rows(
                rest_ohlc_url,
                symbol=symbol,
                interval_minutes=interval_minutes,
                since=int(window.start.timestamp()),
            )
            closed_rows = _closed_rest_rows(rest_rows, lookback_candles=None)
            filtered_rows = [row for row in closed_rows if _row_in_window(row, window)]
            exchange_window_truncated = False
            if closed_rows:
                earliest_closed = datetime.fromtimestamp(int(closed_rows[0][0]), tz=timezone.utc)
                exchange_window_truncated = earliest_closed > window.start
            return filtered_rows, exchange_window_truncated
        except Exception as error:  # pylint: disable=broad-exception-caught
            last_error = error
    if last_error is None:
        raise ValueError(f"Kraken REST fetch failed for {symbol}")
    raise ValueError(f"Kraken REST fetch failed for {symbol}: {last_error}") from last_error


def _ohlc_key(event: OhlcEvent) -> tuple[str, int, datetime]:
    return (event.symbol, event.interval_minutes, event.interval_begin)


def _ohlc_fingerprint(event: OhlcEvent) -> tuple[Any, ...]:
    return (
        event.interval_end,
        event.open_price,
        event.high_price,
        event.low_price,
        event.close_price,
        event.vwap,
        event.trade_count,
        event.volume,
        event.event_id,
        event.received_at,
    )


def _feature_key(row: FeatureOhlcRow) -> tuple[str, int, datetime]:
    return (row.symbol, row.interval_minutes, row.interval_begin)


def _feature_fingerprint(row: FeatureOhlcRow) -> tuple[Any, ...]:
    return (
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


def _normalize_backfill_feature_row(row: FeatureOhlcRow) -> FeatureOhlcRow:
    return replace(row, computed_at=row.as_of_time)


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


async def _sync_symbol_raw_window(
    *,
    writer: PostgresWriter,
    store: FeatureStore,
    rest_ohlc_url: str,
    app_name: str,
    symbol: str,
    interval_minutes: int,
    window: BackfillWindow,
    request_retries: int,
) -> RawSyncStats:
    fetched_rows, exchange_window_truncated = _fetch_window_rows_with_retries(
        rest_ohlc_url,
        symbol=symbol,
        interval_minutes=interval_minutes,
        window=window,
        request_retries=request_retries,
    )
    existing_events = await store.load_raw_candles(
        symbols=(symbol,),
        interval_minutes=interval_minutes,
        start=window.start,
        end=window.end,
    )
    existing_by_key = {_ohlc_key(event): event for event in existing_events}
    created_rows = 0
    updated_rows = 0
    unchanged_rows = 0

    for row in fetched_rows:
        interval_begin = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        event = _event_from_rest_row(
            app_name=app_name,
            symbol=symbol,
            interval_minutes=interval_minutes,
            row=row,
            received_at=interval_begin + timedelta(minutes=interval_minutes),
        )
        existing_event = existing_by_key.get(_ohlc_key(event))
        if existing_event is None:
            created_rows += 1
            await writer.write_ohlc(event)
            continue
        if _ohlc_fingerprint(existing_event) == _ohlc_fingerprint(event):
            unchanged_rows += 1
            continue
        updated_rows += 1
        await writer.write_ohlc(event)

    return RawSyncStats(
        symbol=symbol,
        fetched_rows=len(fetched_rows),
        created_rows=created_rows,
        updated_rows=updated_rows,
        unchanged_rows=unchanged_rows,
        exchange_window_truncated=exchange_window_truncated,
    )


async def _regenerate_feature_rows(
    *,
    store: FeatureStore,
    symbols: tuple[str, ...],
    interval_minutes: int,
    history_limit: int,
    window: BackfillWindow,
) -> FeatureReplayStats:
    raw_events = await store.load_raw_candles(
        symbols=symbols,
        interval_minutes=interval_minutes,
    )
    state = FeatureStateManager(
        grace_seconds=0,
        history_limit=max(history_limit, MIN_FINALIZED_CANDLES),
    )
    rebuilt_rows = state.bootstrap(
        raw_events,
        now=utc_now(),
        computed_at=utc_now(),
    )
    target_rows = [
        _normalize_backfill_feature_row(row)
        for row in rebuilt_rows
        if window.start <= row.interval_begin < window.end
    ]
    existing_rows = await store.load_feature_rows(
        symbols=symbols,
        interval_minutes=interval_minutes,
        start=window.start,
        end=window.end,
    )
    existing_by_key = {_feature_key(row): row for row in existing_rows}
    created_rows = 0
    updated_rows = 0
    unchanged_rows = 0
    for row in target_rows:
        existing_row = existing_by_key.get(_feature_key(row))
        if existing_row is None:
            created_rows += 1
            await store.upsert_feature_row(row)
            continue
        if _feature_fingerprint(existing_row) == _feature_fingerprint(row):
            unchanged_rows += 1
            continue
        updated_rows += 1
        await store.upsert_feature_row(row)
    return FeatureReplayStats(
        generated_rows=len(target_rows),
        created_rows=created_rows,
        updated_rows=updated_rows,
        unchanged_rows=unchanged_rows,
        skipped_outside_window_rows=len(rebuilt_rows) - len(target_rows),
    )


async def _run_backfill(arguments: argparse.Namespace, settings: Settings) -> None:
    logger = logging.getLogger(f"{settings.app_name}.backfill")
    if arguments.interval != settings.kraken.ohlc_interval_minutes:
        raise ValueError(
            "Historical backfill must use the same interval as the live pipeline: "
            f"{settings.kraken.ohlc_interval_minutes} minutes."
        )
    window = _resolve_backfill_window(arguments=arguments, interval_minutes=arguments.interval)
    report_only = bool(arguments.report_only)
    skip_raw_backfill = bool(arguments.skip_raw_backfill or report_only)
    skip_feature_replay = bool(arguments.skip_feature_replay or report_only)
    training_config_path = Path(arguments.training_config).resolve()

    postgres_dsn = await _resolve_postgres_dsn(settings.postgres)
    writer = PostgresWriter(postgres_dsn, settings.tables)
    store = FeatureStore(postgres_dsn, settings.tables)
    await writer.connect()
    await store.connect()

    raw_stats: list[RawSyncStats] = []
    feature_stats: FeatureReplayStats | None = None
    try:
        if not skip_raw_backfill:
            for symbol in arguments.symbols:
                logger.info(
                    "Fetching Kraken REST OHLC history",
                    extra={
                        "symbol": symbol,
                        "interval_minutes": arguments.interval,
                        "start": to_rfc3339(window.start),
                        "end": to_rfc3339(window.end),
                    },
                )
                raw_stats.append(
                    await _sync_symbol_raw_window(
                        writer=writer,
                        store=store,
                        rest_ohlc_url=settings.kraken.rest_ohlc_url,
                        app_name=settings.app_name,
                        symbol=symbol,
                        interval_minutes=arguments.interval,
                        window=window,
                        request_retries=arguments.request_retries,
                    )
                )
        if not skip_feature_replay:
            feature_stats = await _regenerate_feature_rows(
                store=store,
                symbols=tuple(arguments.symbols),
                interval_minutes=arguments.interval,
                history_limit=settings.features.bootstrap_candles,
                window=window,
            )
    finally:
        await writer.close()
        await store.close()

    readiness_report = await asyncio.to_thread(
        build_data_readiness_report_from_path,
        training_config_path,
    )
    artifact_dir = write_data_readiness_artifacts(
        readiness_report,
        artifact_root=Path(arguments.report_artifact_root),
    )
    operation_summary = {
        "generated_at": to_rfc3339(utc_now()),
        "requested_window": window.to_dict(),
        "symbols": list(arguments.symbols),
        "skip_raw_backfill": skip_raw_backfill,
        "skip_feature_replay": skip_feature_replay,
        "raw_sync": [stats.to_dict() for stats in raw_stats],
        "feature_replay": None if feature_stats is None else feature_stats.to_dict(),
        "readiness_report_artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "backfill_operation.json").write_text(
        json.dumps(make_json_safe(operation_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "Historical Kraken OHLC backfill complete",
        extra={
            "requested_window": window.to_dict(),
            "raw_sync": [stats.to_dict() for stats in raw_stats],
            "feature_replay": (
                None if feature_stats is None else feature_stats.to_dict()
            ),
            "readiness_report_artifact_dir": str(artifact_dir),
            "readiness_detail": readiness_report.readiness_detail,
        },
    )
    print(f"requested_window={json.dumps(window.to_dict(), sort_keys=True)}")
    for stats in raw_stats:
        print(
            "raw_sync "
            f"symbol={stats.symbol} "
            f"fetched={stats.fetched_rows} "
            f"created={stats.created_rows} "
            f"updated={stats.updated_rows} "
            f"unchanged={stats.unchanged_rows} "
            f"exchange_window_truncated={stats.exchange_window_truncated}"
        )
    if feature_stats is not None:
        print(
            "feature_replay "
            f"generated={feature_stats.generated_rows} "
            f"created={feature_stats.created_rows} "
            f"updated={feature_stats.updated_rows} "
            f"unchanged={feature_stats.unchanged_rows} "
            f"skipped_outside_window={feature_stats.skipped_outside_window_rows}"
        )
    print(f"readiness_report_artifact_dir={artifact_dir}")
    print(f"ready_for_training={readiness_report.ready_for_training}")
    print(f"readiness_detail={readiness_report.readiness_detail}")


def main() -> None:
    """Run the one-shot Kraken REST OHLC backfill command."""
    settings = Settings.from_env()
    parser = _build_argument_parser(settings)
    arguments = parser.parse_args()
    configure_logging(settings.log_level)
    asyncio.run(_run_backfill(arguments, settings))


if __name__ == "__main__":
    main()
