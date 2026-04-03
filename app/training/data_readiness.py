"""Historical data sufficiency and training-readiness diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg

from app.common.config import Settings
from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.regime.live import load_live_regime_runtime
from app.training.dataset import (
    DatasetSample,
    TrainingConfig,
    candidate_dsns,
    load_training_config,
    load_training_dataset_preview,
)
from app.training.splits import minimum_required_unique_timestamps


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


@dataclass(frozen=True, slots=True)
class GapWindow:
    """One contiguous missing-interval window for a symbol/table pair."""

    start: datetime
    end: datetime
    missing_intervals: int

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class SymbolReadinessSnapshot:
    """Coverage and label diagnostics for one configured symbol."""

    symbol: str
    raw_row_count: int
    feature_row_count: int
    labeled_row_count: int
    raw_earliest_interval_begin: datetime | None
    raw_latest_interval_begin: datetime | None
    feature_earliest_interval_begin: datetime | None
    feature_latest_interval_begin: datetime | None
    labeled_earliest_as_of_time: datetime | None
    labeled_latest_as_of_time: datetime | None
    positive_label_rate: float | None
    raw_missing_interval_count: int
    feature_missing_interval_count: int
    raw_gap_windows: tuple[GapWindow, ...]
    feature_gap_windows: tuple[GapWindow, ...]
    regime_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(
            {
                **asdict(self),
                "raw_gap_windows": [window.to_dict() for window in self.raw_gap_windows],
                "feature_gap_windows": [window.to_dict() for window in self.feature_gap_windows],
            }
        )


@dataclass(frozen=True, slots=True)
class DataReadinessReport:
    """Persistable training-readiness report built from the real DB tables."""

    config_path: str
    generated_at: datetime
    artifact_root: str
    source_table: str
    raw_table: str
    feature_table: str
    source_exchange: str
    interval_minutes: int
    postgres_reachable: bool
    postgres_error: str | None
    raw_table_exists: bool | None
    feature_table_exists: bool | None
    raw_rows_total: int
    feature_rows_total: int
    labeled_rows_total: int
    unique_timestamps: int
    required_unique_timestamps: int
    ready_for_training: bool
    readiness_detail: str
    earliest_usable_timestamp: datetime | None
    latest_usable_timestamp: datetime | None
    overall_positive_label_rate: float | None
    label_counts: dict[str, int]
    regime_distribution_available: bool
    regime_distribution_detail: str | None
    regime_distribution: dict[str, int]
    opportunity_density: dict[str, Any]
    warnings: tuple[str, ...]
    symbol_summaries: tuple[SymbolReadinessSnapshot, ...]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(
            {
                **asdict(self),
                "symbol_summaries": [snapshot.to_dict() for snapshot in self.symbol_summaries],
            }
        )


@dataclass(frozen=True, slots=True)
class _TableCoverage:
    exists: bool
    intervals_by_symbol: dict[str, tuple[datetime, ...]]


def default_data_readiness_artifact_root() -> Path:
    """Return the default artifact root for readiness reports."""
    return Path(__file__).resolve().parents[2] / "artifacts" / "training" / "data_readiness"


def build_data_readiness_report_from_path(config_path: Path) -> DataReadinessReport:
    """Load a checked-in training config and build the matching readiness report."""
    resolved_config_path = Path(config_path).resolve()
    config = load_training_config(resolved_config_path)
    return build_data_readiness_report(config, config_path=resolved_config_path)


def build_data_readiness_report(
    config: TrainingConfig,
    *,
    config_path: Path | None = None,
) -> DataReadinessReport:
    """Build a report from the real raw -> feature -> label pipeline tables."""
    settings = Settings.from_env()
    dataset_error: str | None = None
    try:
        dataset = load_training_dataset_preview(config)
    except ValueError as error:
        dataset_error = str(error)
        dataset = _empty_training_dataset(config)
    required_unique_timestamps = minimum_required_unique_timestamps(
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )
    try:
        coverage = asyncio.run(
            _load_table_coverage_with_fallback(
                settings=settings,
                symbols=config.symbols,
                interval_minutes=settings.kraken.ohlc_interval_minutes,
            )
        )
    except ValueError as error:
        return DataReadinessReport(
            config_path=str(config_path.resolve()) if config_path is not None else "<in-memory>",
            generated_at=utc_now(),
            artifact_root=config.artifact_root,
            source_table=config.source_table,
            raw_table=settings.tables.raw_ohlc,
            feature_table=settings.tables.feature_ohlc,
            source_exchange="kraken",
            interval_minutes=settings.kraken.ohlc_interval_minutes,
            postgres_reachable=False,
            postgres_error=str(error),
            raw_table_exists=None,
            feature_table_exists=None,
            raw_rows_total=0,
            feature_rows_total=0,
            labeled_rows_total=len(dataset.samples),
            unique_timestamps=int(dataset.manifest["unique_timestamps"]),
            required_unique_timestamps=required_unique_timestamps,
            ready_for_training=False,
            readiness_detail="PostgreSQL is not reachable for historical readiness checks.",
            earliest_usable_timestamp=_min_sample_as_of_time(dataset.samples),
            latest_usable_timestamp=_max_sample_as_of_time(dataset.samples),
            overall_positive_label_rate=_positive_label_rate(dataset.samples),
            label_counts=dict(dataset.manifest["label_counts"]),
            regime_distribution_available=False,
            regime_distribution_detail=str(error),
            regime_distribution={},
            opportunity_density=_build_opportunity_density(dataset.samples),
            warnings=("PostgreSQL is not reachable for readiness checks.",),
            symbol_summaries=tuple(
                SymbolReadinessSnapshot(
                    symbol=symbol,
                    raw_row_count=0,
                    feature_row_count=0,
                    labeled_row_count=0,
                    raw_earliest_interval_begin=None,
                    raw_latest_interval_begin=None,
                    feature_earliest_interval_begin=None,
                    feature_latest_interval_begin=None,
                    labeled_earliest_as_of_time=None,
                    labeled_latest_as_of_time=None,
                    positive_label_rate=None,
                    raw_missing_interval_count=0,
                    feature_missing_interval_count=0,
                    raw_gap_windows=(),
                    feature_gap_windows=(),
                    regime_distribution={},
                )
                for symbol in config.symbols
            ),
        )

    regime_distribution, per_symbol_regimes, regime_detail = _resolve_regime_distribution(
        dataset.samples
    )
    symbol_summaries = _build_symbol_readiness_snapshots(
        symbols=config.symbols,
        dataset_samples=dataset.samples,
        raw_intervals_by_symbol=coverage["raw"].intervals_by_symbol,
        feature_intervals_by_symbol=coverage["feature"].intervals_by_symbol,
        regime_distribution_by_symbol=per_symbol_regimes,
        interval_minutes=settings.kraken.ohlc_interval_minutes,
    )
    raw_rows_total = sum(snapshot.raw_row_count for snapshot in symbol_summaries)
    feature_rows_total = sum(snapshot.feature_row_count for snapshot in symbol_summaries)
    labeled_rows_total = len(dataset.samples)
    unique_timestamps = int(dataset.manifest["unique_timestamps"])
    warnings = _build_warnings(
        symbol_summaries=symbol_summaries,
        dataset=dataset,
        required_unique_timestamps=required_unique_timestamps,
        feature_table_exists=coverage["feature"].exists,
        dataset_error=dataset_error,
    )
    readiness_detail = _resolve_readiness_detail(
        config=config,
        raw_table_exists=coverage["raw"].exists,
        feature_table_exists=coverage["feature"].exists,
        symbol_summaries=symbol_summaries,
        labeled_rows_total=labeled_rows_total,
        unique_timestamps=unique_timestamps,
        required_unique_timestamps=required_unique_timestamps,
        dataset_error=dataset_error,
    )
    ready_for_training = (
        labeled_rows_total > 0
        and unique_timestamps >= required_unique_timestamps
        and all(snapshot.feature_row_count > 0 for snapshot in symbol_summaries)
    )
    return DataReadinessReport(
        config_path=str(config_path.resolve()) if config_path is not None else "<in-memory>",
        generated_at=utc_now(),
        artifact_root=config.artifact_root,
        source_table=config.source_table,
        raw_table=settings.tables.raw_ohlc,
        feature_table=settings.tables.feature_ohlc,
        source_exchange="kraken",
        interval_minutes=settings.kraken.ohlc_interval_minutes,
        postgres_reachable=True,
        postgres_error=None,
        raw_table_exists=coverage["raw"].exists,
        feature_table_exists=coverage["feature"].exists,
        raw_rows_total=raw_rows_total,
        feature_rows_total=feature_rows_total,
        labeled_rows_total=labeled_rows_total,
        unique_timestamps=unique_timestamps,
        required_unique_timestamps=required_unique_timestamps,
        ready_for_training=ready_for_training,
        readiness_detail=readiness_detail,
        earliest_usable_timestamp=_min_sample_as_of_time(dataset.samples),
        latest_usable_timestamp=_max_sample_as_of_time(dataset.samples),
        overall_positive_label_rate=_positive_label_rate(dataset.samples),
        label_counts=dict(dataset.manifest["label_counts"]),
        regime_distribution_available=regime_detail is None,
        regime_distribution_detail=regime_detail,
        regime_distribution=regime_distribution,
        opportunity_density=_build_opportunity_density(dataset.samples),
        warnings=warnings,
        symbol_summaries=symbol_summaries,
    )


def assert_training_data_ready(config: TrainingConfig, *, config_path: Path | None = None) -> None:
    """Fail early with the shared readiness logic before a training run starts."""
    report = build_data_readiness_report(config, config_path=config_path)
    if report.ready_for_training:
        return
    raise ValueError(report.readiness_detail)


def write_data_readiness_artifacts(
    report: DataReadinessReport,
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Persist the readiness report as JSON/CSV/Markdown artifacts."""
    root = (
        default_data_readiness_artifact_root()
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    run_id = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "readiness_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        artifact_dir / "symbol_coverage.csv",
        [
            {
                "symbol": snapshot.symbol,
                "raw_row_count": snapshot.raw_row_count,
                "feature_row_count": snapshot.feature_row_count,
                "labeled_row_count": snapshot.labeled_row_count,
                "positive_label_rate": snapshot.positive_label_rate,
                "raw_missing_interval_count": snapshot.raw_missing_interval_count,
                "feature_missing_interval_count": snapshot.feature_missing_interval_count,
                "raw_earliest_interval_begin": _maybe_timestamp(
                    snapshot.raw_earliest_interval_begin
                ),
                "raw_latest_interval_begin": _maybe_timestamp(
                    snapshot.raw_latest_interval_begin
                ),
                "feature_earliest_interval_begin": _maybe_timestamp(
                    snapshot.feature_earliest_interval_begin
                ),
                "feature_latest_interval_begin": _maybe_timestamp(
                    snapshot.feature_latest_interval_begin
                ),
                "labeled_earliest_as_of_time": _maybe_timestamp(
                    snapshot.labeled_earliest_as_of_time
                ),
                "labeled_latest_as_of_time": _maybe_timestamp(
                    snapshot.labeled_latest_as_of_time
                ),
                "regime_distribution": json.dumps(snapshot.regime_distribution, sort_keys=True),
            }
            for snapshot in report.symbol_summaries
        ],
    )
    gap_rows: list[dict[str, Any]] = []
    for snapshot in report.symbol_summaries:
        gap_rows.extend(
            {
                "symbol": snapshot.symbol,
                "table_name": report.raw_table,
                "gap_start": to_rfc3339(window.start),
                "gap_end": to_rfc3339(window.end),
                "missing_intervals": window.missing_intervals,
            }
            for window in snapshot.raw_gap_windows
        )
        gap_rows.extend(
            {
                "symbol": snapshot.symbol,
                "table_name": report.feature_table,
                "gap_start": to_rfc3339(window.start),
                "gap_end": to_rfc3339(window.end),
                "missing_intervals": window.missing_intervals,
            }
            for window in snapshot.feature_gap_windows
        )
    _write_csv(artifact_dir / "gap_summary.csv", gap_rows)
    (artifact_dir / "summary.md").write_text(_render_summary(report), encoding="utf-8")
    return artifact_dir


async def _load_table_coverage_with_fallback(
    *,
    settings: Settings,
    symbols: tuple[str, ...],
    interval_minutes: int,
) -> dict[str, _TableCoverage]:
    last_error: Exception | None = None
    for dsn in candidate_dsns(settings.postgres):
        try:
            return await _load_table_coverage(
                dsn=dsn,
                raw_table_name=settings.tables.raw_ohlc,
                feature_table_name=settings.tables.feature_ohlc,
                symbols=symbols,
                interval_minutes=interval_minutes,
            )
        except (OSError, asyncpg.PostgresConnectionError) as error:
            last_error = error
            continue
    if last_error is None:
        raise ValueError("No PostgreSQL DSN candidates were available for readiness checks")
    raise ValueError(f"Could not connect to PostgreSQL for readiness checks: {last_error}") from last_error


async def _load_table_coverage(
    *,
    dsn: str,
    raw_table_name: str,
    feature_table_name: str,
    symbols: tuple[str, ...],
    interval_minutes: int,
) -> dict[str, _TableCoverage]:
    connection = await asyncpg.connect(dsn)
    try:
        raw_table = await _load_symbol_intervals(
            connection=connection,
            table_name=raw_table_name,
            symbols=symbols,
            interval_minutes=interval_minutes,
        )
        feature_table = await _load_symbol_intervals(
            connection=connection,
            table_name=feature_table_name,
            symbols=symbols,
            interval_minutes=interval_minutes,
        )
        return {"raw": raw_table, "feature": feature_table}
    finally:
        await connection.close()


async def _load_symbol_intervals(
    *,
    connection: asyncpg.Connection,
    table_name: str,
    symbols: tuple[str, ...],
    interval_minutes: int,
) -> _TableCoverage:
    quoted_table_name = _quote_table_name(table_name)
    try:
        rows = await connection.fetch(
            f"""
            SELECT symbol, interval_begin
            FROM {quoted_table_name}
            WHERE source_exchange = $1
              AND interval_minutes = $2
              AND symbol = ANY($3::text[])
            ORDER BY symbol ASC, interval_begin ASC
            """,
            "kraken",
            interval_minutes,
            list(symbols),
        )
    except (asyncpg.InvalidSchemaNameError, asyncpg.UndefinedTableError):
        return _TableCoverage(
            exists=False,
            intervals_by_symbol={symbol: tuple() for symbol in symbols},
        )

    intervals_by_symbol: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        intervals_by_symbol[str(row["symbol"])].append(row["interval_begin"])
    return _TableCoverage(
        exists=True,
        intervals_by_symbol={
            symbol: tuple(intervals_by_symbol.get(symbol, []))
            for symbol in symbols
        },
    )


def _build_symbol_readiness_snapshots(
    *,
    symbols: tuple[str, ...],
    dataset_samples: tuple[DatasetSample, ...],
    raw_intervals_by_symbol: dict[str, tuple[datetime, ...]],
    feature_intervals_by_symbol: dict[str, tuple[datetime, ...]],
    regime_distribution_by_symbol: dict[str, dict[str, int]],
    interval_minutes: int,
) -> tuple[SymbolReadinessSnapshot, ...]:
    samples_by_symbol: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in dataset_samples:
        samples_by_symbol[sample.symbol].append(sample)
    ordered_snapshots: list[SymbolReadinessSnapshot] = []
    for symbol in symbols:
        symbol_samples = tuple(sorted(samples_by_symbol.get(symbol, []), key=lambda sample: sample.as_of_time))
        raw_intervals = tuple(sorted(raw_intervals_by_symbol.get(symbol, ())))
        feature_intervals = tuple(sorted(feature_intervals_by_symbol.get(symbol, ())))
        raw_gap_count, raw_gaps = _compute_gap_windows(raw_intervals, interval_minutes=interval_minutes)
        feature_gap_count, feature_gaps = _compute_gap_windows(
            feature_intervals,
            interval_minutes=interval_minutes,
        )
        ordered_snapshots.append(
            SymbolReadinessSnapshot(
                symbol=symbol,
                raw_row_count=len(raw_intervals),
                feature_row_count=len(feature_intervals),
                labeled_row_count=len(symbol_samples),
                raw_earliest_interval_begin=raw_intervals[0] if raw_intervals else None,
                raw_latest_interval_begin=raw_intervals[-1] if raw_intervals else None,
                feature_earliest_interval_begin=feature_intervals[0] if feature_intervals else None,
                feature_latest_interval_begin=feature_intervals[-1] if feature_intervals else None,
                labeled_earliest_as_of_time=(
                    symbol_samples[0].as_of_time if symbol_samples else None
                ),
                labeled_latest_as_of_time=(
                    symbol_samples[-1].as_of_time if symbol_samples else None
                ),
                positive_label_rate=_positive_label_rate(symbol_samples),
                raw_missing_interval_count=raw_gap_count,
                feature_missing_interval_count=feature_gap_count,
                raw_gap_windows=raw_gaps,
                feature_gap_windows=feature_gaps,
                regime_distribution=regime_distribution_by_symbol.get(symbol, {}),
            )
        )
    return tuple(ordered_snapshots)


def _compute_gap_windows(
    intervals: tuple[datetime, ...],
    *,
    interval_minutes: int,
) -> tuple[int, tuple[GapWindow, ...]]:
    if len(intervals) < 2:
        return 0, ()
    step = timedelta(minutes=interval_minutes)
    gap_windows: list[GapWindow] = []
    missing_interval_count = 0
    for previous, current in zip(intervals, intervals[1:]):
        delta = current - previous
        missing_intervals = int(delta.total_seconds() // step.total_seconds()) - 1
        if missing_intervals <= 0:
            continue
        missing_interval_count += missing_intervals
        gap_windows.append(
            GapWindow(
                start=previous + step,
                end=current - step,
                missing_intervals=missing_intervals,
            )
        )
    return missing_interval_count, tuple(gap_windows)


def _resolve_regime_distribution(
    samples: tuple[DatasetSample, ...],
) -> tuple[dict[str, int], dict[str, dict[str, int]], str | None]:
    if not samples:
        return {}, {}, "No labeled rows are available yet for regime distribution."
    try:
        runtime = load_live_regime_runtime(thresholds_path="", signal_policy_path="")
    except ValueError as error:
        return {}, {}, str(error)

    overall_counter: Counter[str] = Counter()
    per_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        resolved = runtime.resolve_feature_row_regime(
            {
                "symbol": sample.symbol,
                "interval_begin": sample.interval_begin,
                "as_of_time": sample.as_of_time,
                "realized_vol_12": sample.features["realized_vol_12"],
                "momentum_3": sample.features["momentum_3"],
                "macd_line_12_26": sample.features["macd_line_12_26"],
            }
        )
        overall_counter[resolved.regime_label] += 1
        per_symbol[sample.symbol][resolved.regime_label] += 1
    return (
        dict(overall_counter),
        {symbol: dict(counter) for symbol, counter in per_symbol.items()},
        None,
    )


def _build_opportunity_density(samples: tuple[DatasetSample, ...]) -> dict[str, Any]:
    thresholds = (0.0, 0.0005, 0.0010, 0.0020)
    overall = {
        f"{int(threshold * 10_000)}bps": sum(
            1 for sample in samples if sample.future_return_3 > threshold
        )
        for threshold in thresholds
    }
    per_symbol: dict[str, dict[str, int]] = {}
    for symbol in sorted({sample.symbol for sample in samples}):
        symbol_samples = tuple(sample for sample in samples if sample.symbol == symbol)
        per_symbol[symbol] = {
            f"{int(threshold * 10_000)}bps": sum(
                1 for sample in symbol_samples if sample.future_return_3 > threshold
            )
            for threshold in thresholds
        }
    return {"overall": overall, "per_symbol": per_symbol}


def _positive_label_rate(samples: tuple[DatasetSample, ...] | list[DatasetSample]) -> float | None:
    if not samples:
        return None
    positive_count = sum(sample.label for sample in samples)
    return positive_count / len(samples)


def _min_sample_as_of_time(samples: tuple[DatasetSample, ...]) -> datetime | None:
    if not samples:
        return None
    return min(sample.as_of_time for sample in samples)


def _max_sample_as_of_time(samples: tuple[DatasetSample, ...]) -> datetime | None:
    if not samples:
        return None
    return max(sample.as_of_time for sample in samples)


def _build_warnings(
    *,
    symbol_summaries: tuple[SymbolReadinessSnapshot, ...],
    dataset,
    required_unique_timestamps: int,
    feature_table_exists: bool,
    dataset_error: str | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not feature_table_exists:
        warnings.append("feature_ohlc does not exist yet.")
    if any(snapshot.raw_row_count == 0 for snapshot in symbol_summaries):
        missing_symbols = [
            snapshot.symbol for snapshot in symbol_summaries if snapshot.raw_row_count == 0
        ]
        warnings.append(f"Raw history is still missing for configured symbols: {missing_symbols}.")
    if any(snapshot.feature_row_count == 0 for snapshot in symbol_summaries):
        missing_symbols = [
            snapshot.symbol
            for snapshot in symbol_summaries
            if snapshot.feature_row_count == 0
        ]
        warnings.append(
            f"Feature rows are still missing for configured symbols: {missing_symbols}."
        )
    if int(dataset.manifest["eligible_rows"]) == 0:
        warnings.append("No eligible labeled rows are available yet.")
    actual_unique_timestamps = int(dataset.manifest["unique_timestamps"])
    if actual_unique_timestamps < required_unique_timestamps:
        warnings.append(
            "Configured walk-forward training is not ready yet: "
            f"{actual_unique_timestamps}/{required_unique_timestamps} eligible timestamps."
        )
    if any(snapshot.raw_missing_interval_count > 0 for snapshot in symbol_summaries):
        warnings.append("Raw OHLC gaps remain in the configured symbols.")
    if any(snapshot.feature_missing_interval_count > 0 for snapshot in symbol_summaries):
        warnings.append("Feature-table gaps remain in the configured symbols.")
    if dataset_error and dataset_error not in warnings:
        warnings.append(dataset_error)
    return tuple(warnings)


def _resolve_readiness_detail(
    *,
    config: TrainingConfig,
    raw_table_exists: bool,
    feature_table_exists: bool,
    symbol_summaries: tuple[SymbolReadinessSnapshot, ...],
    labeled_rows_total: int,
    unique_timestamps: int,
    required_unique_timestamps: int,
    dataset_error: str | None,
) -> str:
    if not raw_table_exists:
        return "raw_ohlc does not exist yet for historical backfill."
    if not feature_table_exists:
        return f"{config.source_table} does not exist yet for training."
    missing_raw_symbols = [snapshot.symbol for snapshot in symbol_summaries if snapshot.raw_row_count == 0]
    if missing_raw_symbols:
        return (
            "raw_ohlc is still missing configured symbols required for training: "
            f"{missing_raw_symbols}."
        )
    missing_feature_symbols = [
        snapshot.symbol for snapshot in symbol_summaries if snapshot.feature_row_count == 0
    ]
    if missing_feature_symbols:
        return (
            f"{config.source_table} is still missing configured symbols required for training: "
            f"{missing_feature_symbols}."
        )
    if labeled_rows_total == 0:
        return (
            "No eligible labeled rows were produced from feature_ohlc. "
            "Backfill more finalized feature history before retraining."
        )
    if unique_timestamps < required_unique_timestamps:
        return (
            "feature_ohlc does not yet satisfy the configured walk-forward timestamp "
            f"requirement ({unique_timestamps}/{required_unique_timestamps})."
        )
    if dataset_error is not None:
        return dataset_error
    return "feature_ohlc satisfies the configured walk-forward timestamp requirement."


def _empty_training_dataset(config: TrainingConfig):
    return type(
        "EmptyTrainingDataset",
        (),
        {
            "samples": tuple(),
            "manifest": {
                "source_table": config.source_table,
                "loaded_rows": 0,
                "eligible_rows": 0,
                "symbols": {symbol: 0 for symbol in config.symbols},
                "label_counts": {"0": 0, "1": 0},
                "dropped_flat_future_return_rows": 0,
                "dropped_persistence_warmup_rows": 0,
                "unique_timestamps": 0,
                "time_range": {"min_as_of_time": None, "max_as_of_time": None},
            },
        },
    )()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _maybe_timestamp(value: datetime | None) -> str | None:
    return None if value is None else to_rfc3339(value)


def _render_summary(report: DataReadinessReport) -> str:
    lines = [
        "# Stream Alpha data readiness",
        "",
        f"- generated_at: {to_rfc3339(report.generated_at)}",
        f"- source_table: `{report.source_table}`",
        f"- raw_table: `{report.raw_table}`",
        f"- feature_table: `{report.feature_table}`",
        f"- raw_rows_total: {report.raw_rows_total}",
        f"- feature_rows_total: {report.feature_rows_total}",
        f"- labeled_rows_total: {report.labeled_rows_total}",
        f"- earliest_usable_timestamp: {_maybe_timestamp(report.earliest_usable_timestamp) or 'n/a'}",
        f"- latest_usable_timestamp: {_maybe_timestamp(report.latest_usable_timestamp) or 'n/a'}",
        (
            f"- walk_forward_ready: {report.ready_for_training} "
            f"({report.unique_timestamps}/{report.required_unique_timestamps} unique timestamps)"
        ),
        f"- readiness_detail: {report.readiness_detail}",
    ]
    if report.overall_positive_label_rate is not None:
        lines.append(
            f"- overall_positive_label_rate: {report.overall_positive_label_rate:.4f}"
        )
    if report.regime_distribution_available:
        lines.append(
            f"- regime_distribution: `{json.dumps(report.regime_distribution, sort_keys=True)}`"
        )
    else:
        lines.append(
            "- regime_distribution: unavailable "
            f"({report.regime_distribution_detail})"
        )
    lines.append("")
    lines.append("## Warnings")
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Symbol coverage")
    lines.append("")
    lines.append(
        "| symbol | raw_rows | feature_rows | labeled_rows | pos_label_rate | raw_gaps | feature_gaps |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for snapshot in report.symbol_summaries:
        positive_label_rate = (
            "n/a"
            if snapshot.positive_label_rate is None
            else f"{snapshot.positive_label_rate:.4f}"
        )
        lines.append(
            "| "
            f"{snapshot.symbol} | "
            f"{snapshot.raw_row_count} | "
            f"{snapshot.feature_row_count} | "
            f"{snapshot.labeled_row_count} | "
            f"{positive_label_rate} | "
            f"{snapshot.raw_missing_interval_count} | "
            f"{snapshot.feature_missing_interval_count} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Build and optionally persist a data readiness report."""
    parser = argparse.ArgumentParser(description="Inspect Stream Alpha historical data readiness")
    parser.add_argument("--config", required=True, help="Path to the JSON training config")
    parser.add_argument(
        "--artifact-root",
        default=str(default_data_readiness_artifact_root()),
        help="Directory where the readiness artifact should be written",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of writing an artifact",
    )
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="Persist the report under the supplied artifact root",
    )
    arguments = parser.parse_args()

    report = build_data_readiness_report_from_path(Path(arguments.config))
    if arguments.json:
        print(json.dumps(report.to_dict(), sort_keys=True))
        return
    if arguments.write_artifact:
        artifact_dir = write_data_readiness_artifacts(
            report,
            artifact_root=Path(arguments.artifact_root),
        )
        print(f"artifact_dir={artifact_dir}")
        return
    print(_render_summary(report))


if __name__ == "__main__":
    main()
