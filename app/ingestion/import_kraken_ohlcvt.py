"""Import local Kraken OHLCVT CSV files into the real raw -> feature path."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.common.config import Settings
from app.common.logging import configure_logging
from app.common.models import OhlcEvent
from app.common.serialization import make_json_safe
from app.common.time import parse_rfc3339, to_rfc3339, utc_now
from app.features.db import FeatureStore
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.models import FeatureOhlcRow
from app.features.state import FeatureStateManager
from app.ingestion.backfill_ohlc import _resolve_postgres_dsn
from app.ingestion.db import PostgresWriter
from app.training.data_readiness import (
    build_data_readiness_report_from_path,
    default_data_readiness_artifact_root,
    write_data_readiness_artifacts,
)


_DEFAULT_TRAINING_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "training.m7.json"
)
_DEFAULT_BATCH_SIZE = 5_000
_KRAKEN_SYMBOL_FILE_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC/USD": ("XBTUSD", "XXBTZUSD", "BTCUSD"),
    "ETH/USD": ("ETHUSD", "XETHZUSD"),
    "SOL/USD": ("SOLUSD",),
}
_VWAP_FALLBACK_POLICY = "close_price_fallback_when_missing_from_csv"


@dataclass(frozen=True, slots=True)
class ImportWindow:
    """Optional timestamp bounds for local CSV import and replay."""

    start: datetime | None
    end: datetime | None

    def contains(self, interval_begin: datetime) -> bool:
        if self.start is not None and interval_begin < self.start:
            return False
        if self.end is not None and interval_begin >= self.end:
            return False
        return True

    def to_dict(self) -> dict[str, str | None]:
        return {
            "start": None if self.start is None else to_rfc3339(self.start),
            "end": None if self.end is None else to_rfc3339(self.end),
        }


@dataclass(frozen=True, slots=True)
class KrakenCsvFile:
    """One resolved Kraken OHLCVT CSV input file."""

    symbol: str
    pair_code: str
    interval_minutes: int
    path: Path


@dataclass(frozen=True, slots=True)
class RawImportStats:
    """Per-symbol raw OHLC import counts for one local CSV run."""

    symbol: str
    csv_path: str
    parsed_rows: int
    selected_rows: int
    skipped_outside_window_rows: int
    created_rows: int
    updated_rows: int
    unchanged_rows: int
    vwap_fallback_rows: int
    import_start: datetime | None
    import_end: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class FeatureReplaySymbolStats:
    """Per-symbol feature replay counts for one import run."""

    symbol: str
    generated_rows: int
    created_rows: int
    updated_rows: int
    unchanged_rows: int
    skipped_outside_window_rows: int

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class FeatureReplayStats:
    """Aggregate feature replay counts across the selected symbols."""

    generated_rows: int
    created_rows: int
    updated_rows: int
    unchanged_rows: int
    skipped_outside_window_rows: int
    per_symbol: tuple[FeatureReplaySymbolStats, ...]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(
            {
                **asdict(self),
                "per_symbol": [row.to_dict() for row in self.per_symbol],
            }
        )


@dataclass(frozen=True, slots=True)
class _ParsedCsvRow:
    interval_begin: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    vwap: float
    volume: float
    trade_count: int
    used_vwap_fallback: bool


def _build_argument_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import local Kraken OHLCVT CSV files into raw_ohlc and feature_ohlc.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(_default_dataset_root()),
        help="Root directory containing the extracted Kraken OHLCVT CSV files.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(settings.kraken.symbols),
        help="Repo-native symbols to import from the local Kraken dataset.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=settings.kraken.ohlc_interval_minutes,
        help="OHLC interval in minutes. Must match the live pipeline interval.",
    )
    parser.add_argument(
        "--start",
        help="Optional inclusive RFC3339 UTC import start bound.",
    )
    parser.add_argument(
        "--end",
        help="Optional exclusive RFC3339 UTC import end bound.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="How many changed rows to upsert per batch.",
    )
    parser.add_argument(
        "--skip-raw-import",
        action="store_true",
        help="Skip local CSV ingestion and only replay features/report from existing raw_ohlc rows.",
    )
    parser.add_argument(
        "--skip-feature-replay",
        action="store_true",
        help="Skip feature replay and only import raw_ohlc plus the readiness report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip raw and feature writes and only persist the readiness report.",
    )
    parser.add_argument(
        "--training-config",
        default=str(_DEFAULT_TRAINING_CONFIG),
        help="Training config used for the readiness gate and sufficiency report.",
    )
    parser.add_argument(
        "--report-artifact-root",
        default=str(default_data_readiness_artifact_root()),
        help="Artifact root where the readiness report should be written.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the import summary as JSON.",
    )
    return parser


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Datasets" / "master_q4"


def _resolve_import_window(arguments: argparse.Namespace, interval_minutes: int) -> ImportWindow:
    start = None
    end = None
    if arguments.start:
        start = _floor_to_interval(parse_rfc3339(str(arguments.start)), interval_minutes)
    if arguments.end:
        end = _floor_to_interval(parse_rfc3339(str(arguments.end)), interval_minutes)
    if start is not None and end is not None and end <= start:
        raise ValueError("Local Kraken CSV import end must be later than start")
    return ImportWindow(start=start, end=end)


def _floor_to_interval(timestamp: datetime, interval_minutes: int) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    utc_timestamp = timestamp.astimezone(timezone.utc)
    interval_seconds = interval_minutes * 60
    aligned_epoch = int(utc_timestamp.timestamp()) // interval_seconds * interval_seconds
    return datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)


def resolve_kraken_csv_files(
    dataset_root: Path,
    *,
    symbols: tuple[str, ...],
    interval_minutes: int,
) -> tuple[KrakenCsvFile, ...]:
    """Resolve one authoritative local Kraken CSV file per requested symbol."""
    resolved_root = Path(dataset_root).resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Kraken dataset root was not found: {resolved_root}")
    resolved_files: list[KrakenCsvFile] = []
    for symbol in symbols:
        aliases = _KRAKEN_SYMBOL_FILE_ALIASES.get(symbol)
        if aliases is None:
            raise ValueError(f"No Kraken file aliases are defined for symbol {symbol}")
        candidates = [
            resolved_root / f"{pair_code}_{interval_minutes}.csv"
            for pair_code in aliases
            if not pair_code.startswith("._")
        ]
        matches = [path for path in candidates if path.is_file()]
        if not matches:
            raise ValueError(
                "No Kraken OHLCVT CSV file matched "
                f"{symbol} at {interval_minutes}m under {resolved_root}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple Kraken OHLCVT CSV files matched {symbol}: "
                f"{[path.name for path in matches]}"
            )
        matched_path = matches[0]
        resolved_files.append(
            KrakenCsvFile(
                symbol=symbol,
                pair_code=matched_path.stem.rsplit("_", 1)[0],
                interval_minutes=interval_minutes,
                path=matched_path,
            )
        )
    return tuple(resolved_files)


def parse_kraken_csv_row(
    columns: list[str],
    *,
    interval_minutes: int,
    line_number: int,
    source_path: Path,
) -> _ParsedCsvRow:
    """Parse one Kraken downloadable OHLCVT CSV row."""
    if len(columns) not in {7, 8}:
        raise ValueError(
            f"Unsupported Kraken OHLCVT row width {len(columns)} at "
            f"{source_path}:{line_number}"
        )

    try:
        interval_begin = datetime.fromtimestamp(int(float(columns[0])), tz=timezone.utc)
        open_price = float(columns[1])
        high_price = float(columns[2])
        low_price = float(columns[3])
        close_price = float(columns[4])
        used_vwap_fallback = len(columns) == 7
        if used_vwap_fallback:
            vwap = close_price
            volume = float(columns[5])
            trade_count = int(float(columns[6]))
        else:
            vwap = float(columns[5])
            volume = float(columns[6])
            trade_count = int(float(columns[7]))
    except ValueError as error:
        raise ValueError(
            f"Invalid Kraken OHLCVT row at {source_path}:{line_number}: {columns}"
        ) from error

    return _ParsedCsvRow(
        interval_begin=interval_begin,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        vwap=vwap,
        volume=volume,
        trade_count=trade_count,
        used_vwap_fallback=used_vwap_fallback,
    )


def _stable_import_event_id(
    *,
    symbol: str,
    pair_code: str,
    interval_minutes: int,
    row: _ParsedCsvRow,
) -> str:
    payload = "|".join(
        [
            "kraken",
            "downloadable_ohlcvt",
            pair_code,
            symbol,
            str(interval_minutes),
            str(int(row.interval_begin.timestamp())),
            str(row.open_price),
            str(row.high_price),
            str(row.low_price),
            str(row.close_price),
            str(row.vwap),
            str(row.volume),
            str(row.trade_count),
            _VWAP_FALLBACK_POLICY if row.used_vwap_fallback else "csv_vwap",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_from_csv_row(
    *,
    app_name: str,
    symbol: str,
    pair_code: str,
    interval_minutes: int,
    row: _ParsedCsvRow,
) -> OhlcEvent:
    interval_end = row.interval_begin + timedelta(minutes=interval_minutes)
    return OhlcEvent(
        event_id=_stable_import_event_id(
            symbol=symbol,
            pair_code=pair_code,
            interval_minutes=interval_minutes,
            row=row,
        ),
        app_name=app_name,
        source_exchange="kraken",
        channel="ohlc",
        message_type="historical_csv_import",
        symbol=symbol,
        interval_minutes=interval_minutes,
        interval_begin=row.interval_begin,
        interval_end=interval_end,
        open_price=row.open_price,
        high_price=row.high_price,
        low_price=row.low_price,
        close_price=row.close_price,
        vwap=row.vwap,
        trade_count=row.trade_count,
        volume=row.volume,
        received_at=interval_end,
    )


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


def _normalize_feature_row(row: FeatureOhlcRow) -> FeatureOhlcRow:
    return FeatureOhlcRow(
        source_exchange=row.source_exchange,
        symbol=row.symbol,
        interval_minutes=row.interval_minutes,
        interval_begin=row.interval_begin,
        interval_end=row.interval_end,
        as_of_time=row.as_of_time,
        computed_at=row.as_of_time,
        raw_event_id=row.raw_event_id,
        open_price=row.open_price,
        high_price=row.high_price,
        low_price=row.low_price,
        close_price=row.close_price,
        vwap=row.vwap,
        trade_count=row.trade_count,
        volume=row.volume,
        log_return_1=row.log_return_1,
        log_return_3=row.log_return_3,
        momentum_3=row.momentum_3,
        return_mean_12=row.return_mean_12,
        return_std_12=row.return_std_12,
        realized_vol_12=row.realized_vol_12,
        rsi_14=row.rsi_14,
        macd_line_12_26=row.macd_line_12_26,
        volume_mean_12=row.volume_mean_12,
        volume_std_12=row.volume_std_12,
        volume_zscore_12=row.volume_zscore_12,
        close_zscore_12=row.close_zscore_12,
        lag_log_return_1=row.lag_log_return_1,
        lag_log_return_2=row.lag_log_return_2,
        lag_log_return_3=row.lag_log_return_3,
    )


async def _sync_symbol_csv_file(
    *,
    writer: PostgresWriter,
    store: FeatureStore,
    csv_file: KrakenCsvFile,
    app_name: str,
    import_window: ImportWindow,
    batch_size: int,
) -> RawImportStats:
    existing_events = await store.load_raw_candles(
        symbols=(csv_file.symbol,),
        interval_minutes=csv_file.interval_minutes,
        start=import_window.start,
        end=import_window.end,
    )
    existing_by_key = {_ohlc_key(event): event for event in existing_events}
    write_batch: list[OhlcEvent] = []
    parsed_rows = 0
    selected_rows = 0
    skipped_outside_window_rows = 0
    created_rows = 0
    updated_rows = 0
    unchanged_rows = 0
    vwap_fallback_rows = 0
    import_start = None
    import_end = None
    last_timestamp: datetime | None = None

    with csv_file.path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        for line_number, columns in enumerate(reader, start=1):
            if not columns or all(not column.strip() for column in columns):
                continue
            parsed_row = parse_kraken_csv_row(
                [column.strip() for column in columns],
                interval_minutes=csv_file.interval_minutes,
                line_number=line_number,
                source_path=csv_file.path,
            )
            parsed_rows += 1
            if last_timestamp is not None and parsed_row.interval_begin < last_timestamp:
                raise ValueError(
                    "Kraken OHLCVT CSV rows must be ordered by timestamp ascending: "
                    f"{csv_file.path}"
                )
            last_timestamp = parsed_row.interval_begin
            if not import_window.contains(parsed_row.interval_begin):
                skipped_outside_window_rows += 1
                continue
            if parsed_row.used_vwap_fallback:
                vwap_fallback_rows += 1
            selected_rows += 1
            import_start = (
                parsed_row.interval_begin
                if import_start is None
                else min(import_start, parsed_row.interval_begin)
            )
            import_end = (
                parsed_row.interval_begin + timedelta(minutes=csv_file.interval_minutes)
                if import_end is None
                else max(import_end, parsed_row.interval_begin + timedelta(minutes=csv_file.interval_minutes))
            )
            event = _event_from_csv_row(
                app_name=app_name,
                symbol=csv_file.symbol,
                pair_code=csv_file.pair_code,
                interval_minutes=csv_file.interval_minutes,
                row=parsed_row,
            )
            existing_event = existing_by_key.get(_ohlc_key(event))
            if existing_event is None:
                created_rows += 1
                write_batch.append(event)
            elif _ohlc_fingerprint(existing_event) == _ohlc_fingerprint(event):
                unchanged_rows += 1
            else:
                updated_rows += 1
                write_batch.append(event)
            if len(write_batch) >= batch_size:
                await writer.write_ohlc_batch(write_batch)
                write_batch.clear()

    if write_batch:
        await writer.write_ohlc_batch(write_batch)

    return RawImportStats(
        symbol=csv_file.symbol,
        csv_path=str(csv_file.path),
        parsed_rows=parsed_rows,
        selected_rows=selected_rows,
        skipped_outside_window_rows=skipped_outside_window_rows,
        created_rows=created_rows,
        updated_rows=updated_rows,
        unchanged_rows=unchanged_rows,
        vwap_fallback_rows=vwap_fallback_rows,
        import_start=import_start,
        import_end=import_end,
    )


async def _replay_symbol_feature_rows(
    *,
    store: FeatureStore,
    symbol: str,
    interval_minutes: int,
    history_limit: int,
    import_window: ImportWindow,
    batch_size: int,
) -> FeatureReplaySymbolStats:
    raw_events = await store.load_raw_candles(
        symbols=(symbol,),
        interval_minutes=interval_minutes,
    )
    if not raw_events:
        return FeatureReplaySymbolStats(
            symbol=symbol,
            generated_rows=0,
            created_rows=0,
            updated_rows=0,
            unchanged_rows=0,
            skipped_outside_window_rows=0,
        )

    existing_rows = await store.load_feature_rows(
        symbols=(symbol,),
        interval_minutes=interval_minutes,
        start=import_window.start,
        end=import_window.end,
    )
    existing_by_key = {_feature_key(row): row for row in existing_rows}
    state = FeatureStateManager(
        grace_seconds=0,
        history_limit=max(history_limit, MIN_FINALIZED_CANDLES),
    )
    write_batch: list[FeatureOhlcRow] = []
    generated_rows = 0
    created_rows = 0
    updated_rows = 0
    unchanged_rows = 0
    skipped_outside_window_rows = 0

    for event in raw_events:
        emitted_rows = state.apply_event(event, computed_at=event.interval_end)
        for emitted_row in emitted_rows:
            normalized_row = _normalize_feature_row(emitted_row)
            if not import_window.contains(normalized_row.interval_begin):
                skipped_outside_window_rows += 1
                continue
            generated_rows += 1
            existing_row = existing_by_key.get(_feature_key(normalized_row))
            if existing_row is None:
                created_rows += 1
                write_batch.append(normalized_row)
            elif _feature_fingerprint(existing_row) == _feature_fingerprint(normalized_row):
                unchanged_rows += 1
            else:
                updated_rows += 1
                write_batch.append(normalized_row)
            if len(write_batch) >= batch_size:
                await store.upsert_feature_rows_batch(write_batch)
                write_batch.clear()

    final_now = raw_events[-1].interval_end + timedelta(minutes=interval_minutes)
    for emitted_row in state.sweep(now=final_now, computed_at=final_now):
        normalized_row = _normalize_feature_row(emitted_row)
        if not import_window.contains(normalized_row.interval_begin):
            skipped_outside_window_rows += 1
            continue
        generated_rows += 1
        existing_row = existing_by_key.get(_feature_key(normalized_row))
        if existing_row is None:
            created_rows += 1
            write_batch.append(normalized_row)
        elif _feature_fingerprint(existing_row) == _feature_fingerprint(normalized_row):
            unchanged_rows += 1
        else:
            updated_rows += 1
            write_batch.append(normalized_row)
        if len(write_batch) >= batch_size:
            await store.upsert_feature_rows_batch(write_batch)
            write_batch.clear()

    if write_batch:
        await store.upsert_feature_rows_batch(write_batch)

    return FeatureReplaySymbolStats(
        symbol=symbol,
        generated_rows=generated_rows,
        created_rows=created_rows,
        updated_rows=updated_rows,
        unchanged_rows=unchanged_rows,
        skipped_outside_window_rows=skipped_outside_window_rows,
    )


async def _replay_features(
    *,
    store: FeatureStore,
    symbols: tuple[str, ...],
    interval_minutes: int,
    history_limit: int,
    import_window: ImportWindow,
    batch_size: int,
) -> FeatureReplayStats:
    per_symbol = [
        await _replay_symbol_feature_rows(
            store=store,
            symbol=symbol,
            interval_minutes=interval_minutes,
            history_limit=history_limit,
            import_window=import_window,
            batch_size=batch_size,
        )
        for symbol in symbols
    ]
    return FeatureReplayStats(
        generated_rows=sum(row.generated_rows for row in per_symbol),
        created_rows=sum(row.created_rows for row in per_symbol),
        updated_rows=sum(row.updated_rows for row in per_symbol),
        unchanged_rows=sum(row.unchanged_rows for row in per_symbol),
        skipped_outside_window_rows=sum(
            row.skipped_outside_window_rows for row in per_symbol
        ),
        per_symbol=tuple(per_symbol),
    )


async def _run_import(arguments: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    logger = logging.getLogger(f"{settings.app_name}.import_kraken_ohlcvt")
    if arguments.interval != settings.kraken.ohlc_interval_minutes:
        raise ValueError(
            "Local Kraken OHLCVT import must use the same interval as the live pipeline: "
            f"{settings.kraken.ohlc_interval_minutes} minutes."
        )
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    report_only = bool(arguments.report_only)
    skip_raw_import = bool(arguments.skip_raw_import or report_only)
    skip_feature_replay = bool(arguments.skip_feature_replay or report_only)
    import_window = _resolve_import_window(arguments, settings.kraken.ohlc_interval_minutes)
    training_config_path = Path(arguments.training_config).resolve()
    csv_files = (
        tuple()
        if skip_raw_import
        else resolve_kraken_csv_files(
            Path(arguments.dataset_root),
            symbols=tuple(arguments.symbols),
            interval_minutes=arguments.interval,
        )
    )

    postgres_dsn = await _resolve_postgres_dsn(settings.postgres)
    writer = PostgresWriter(postgres_dsn, settings.tables)
    store = FeatureStore(postgres_dsn, settings.tables)
    await writer.connect()
    await store.connect()

    raw_stats: list[RawImportStats] = []
    feature_stats: FeatureReplayStats | None = None
    try:
        if not skip_raw_import:
            for csv_file in csv_files:
                logger.info(
                    "Importing Kraken OHLCVT CSV",
                    extra={
                        "symbol": csv_file.symbol,
                        "csv_path": str(csv_file.path),
                        "interval_minutes": csv_file.interval_minutes,
                        "requested_window": import_window.to_dict(),
                    },
                )
                raw_stats.append(
                    await _sync_symbol_csv_file(
                        writer=writer,
                        store=store,
                        csv_file=csv_file,
                        app_name=settings.app_name,
                        import_window=import_window,
                        batch_size=arguments.batch_size,
                    )
                )
        if not skip_feature_replay:
            feature_stats = await _replay_features(
                store=store,
                symbols=tuple(arguments.symbols),
                interval_minutes=arguments.interval,
                history_limit=settings.features.bootstrap_candles,
                import_window=import_window,
                batch_size=arguments.batch_size,
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
        "dataset_root": str(Path(arguments.dataset_root).resolve()),
        "requested_window": import_window.to_dict(),
        "symbols": list(arguments.symbols),
        "interval_minutes": arguments.interval,
        "skip_raw_import": skip_raw_import,
        "skip_feature_replay": skip_feature_replay,
        "vwap_policy": (
            "CSV provided vwap when present; "
            f"otherwise { _VWAP_FALLBACK_POLICY }"
        ),
        "raw_import": [stats.to_dict() for stats in raw_stats],
        "feature_replay": None if feature_stats is None else feature_stats.to_dict(),
        "readiness_report_artifact_dir": str(artifact_dir),
        "readiness": {
            "ready_for_training": readiness_report.ready_for_training,
            "readiness_detail": readiness_report.readiness_detail,
            "raw_rows_total": readiness_report.raw_rows_total,
            "feature_rows_total": readiness_report.feature_rows_total,
            "labeled_rows_total": readiness_report.labeled_rows_total,
            "earliest_usable_timestamp": (
                None
                if readiness_report.earliest_usable_timestamp is None
                else to_rfc3339(readiness_report.earliest_usable_timestamp)
            ),
            "latest_usable_timestamp": (
                None
                if readiness_report.latest_usable_timestamp is None
                else to_rfc3339(readiness_report.latest_usable_timestamp)
            ),
        },
    }
    (artifact_dir / "import_operation.json").write_text(
        json.dumps(make_json_safe(operation_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return operation_summary


def main() -> None:
    """Run the one-shot local Kraken OHLCVT importer."""
    settings = Settings.from_env()
    parser = _build_argument_parser(settings)
    arguments = parser.parse_args()
    configure_logging(settings.log_level)
    summary = asyncio.run(_run_import(arguments, settings))
    if arguments.json:
        print(json.dumps(make_json_safe(summary), sort_keys=True))
        return
    print(
        "imported_symbols="
        f"{json.dumps(summary['symbols'])}"
    )
    for stats in summary["raw_import"]:
        print(
            "raw_import "
            f"symbol={stats['symbol']} "
            f"parsed={stats['parsed_rows']} "
            f"selected={stats['selected_rows']} "
            f"created={stats['created_rows']} "
            f"updated={stats['updated_rows']} "
            f"unchanged={stats['unchanged_rows']} "
            f"skipped_outside_window={stats['skipped_outside_window_rows']} "
            f"vwap_fallback_rows={stats['vwap_fallback_rows']}"
        )
    if summary["feature_replay"] is not None:
        feature_replay = summary["feature_replay"]
        print(
            "feature_replay "
            f"generated={feature_replay['generated_rows']} "
            f"created={feature_replay['created_rows']} "
            f"updated={feature_replay['updated_rows']} "
            f"unchanged={feature_replay['unchanged_rows']} "
            f"skipped_outside_window={feature_replay['skipped_outside_window_rows']}"
        )
    print(f"readiness_report_artifact_dir={summary['readiness_report_artifact_dir']}")
    print(f"ready_for_training={summary['readiness']['ready_for_training']}")
    print(f"readiness_detail={summary['readiness']['readiness_detail']}")


if __name__ == "__main__":
    main()
