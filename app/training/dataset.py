"""Dataset loading and label construction for the M3 offline training pipeline."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg

from app.common.config import PostgresSettings, Settings
from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339


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


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


LEGACY_ARCHIVED_MODEL_NAMES = frozenset(
    {
        "logistic_regression",
        "hist_gradient_boosting",
    }
)

SEQUENCE_CONTEXT_KEY = "__sequence_context__"


@dataclass(frozen=True, slots=True)
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Checked-in M3 configuration for dataset construction and evaluation."""

    source_table: str
    symbols: tuple[str, ...]
    time_column: str
    interval_column: str
    close_column: str
    categorical_feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    label_horizon_candles: int
    purge_gap_candles: int
    test_folds: int
    first_train_fraction: float
    test_fraction: float
    round_trip_fee_bps: float
    artifact_root: str
    models: dict[str, dict[str, Any]]

    @property
    def round_trip_fee_rate(self) -> float:
        """Return the configured round-trip fee in decimal form."""
        return self.round_trip_fee_bps / 10_000.0

    @property
    def all_feature_columns(self) -> tuple[str, ...]:
        """Return the full ordered feature column list used by the models."""
        return self.categorical_feature_columns + self.numeric_feature_columns

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary form for artifact persistence."""
        return make_json_safe(
            {
                "source_table": self.source_table,
                "symbols": list(self.symbols),
                "time_column": self.time_column,
                "interval_column": self.interval_column,
                "close_column": self.close_column,
                "categorical_feature_columns": list(self.categorical_feature_columns),
                "numeric_feature_columns": list(self.numeric_feature_columns),
                "label_horizon_candles": self.label_horizon_candles,
                "purge_gap_candles": self.purge_gap_candles,
                "test_folds": self.test_folds,
                "first_train_fraction": self.first_train_fraction,
                "test_fraction": self.test_fraction,
                "round_trip_fee_bps": self.round_trip_fee_bps,
                "artifact_root": self.artifact_root,
                "models": {
                    model_name: dict(model_config)
                    for model_name, model_config in sorted(self.models.items())
                },
            }
        )


@dataclass(frozen=True, slots=True)
class DatasetSample:  # pylint: disable=too-many-instance-attributes
    """One labeled training sample derived from a finalized feature row."""

    row_id: str
    symbol: str
    interval_begin: datetime
    as_of_time: datetime
    close_price: float
    future_close_price: float
    future_return_3: float
    label: int
    persistence_prediction: int
    features: dict[str, Any]

    def to_prediction_metadata(self) -> dict[str, Any]:
        """Return the source metadata used in OOF prediction artifacts."""
        return {
            "row_id": self.row_id,
            "symbol": self.symbol,
            "interval_begin": to_rfc3339(self.interval_begin),
            "as_of_time": to_rfc3339(self.as_of_time),
            "close_price": self.close_price,
            "future_close_price": self.future_close_price,
            "future_return_3": self.future_return_3,
            "y_true": self.label,
            "persistence_prediction": self.persistence_prediction,
        }


@dataclass(frozen=True, slots=True)
class SourceFeatureRow:
    """One ordered finalized feature row preserved for sequential training paths."""

    row_id: str
    symbol: str
    interval_begin: datetime
    as_of_time: datetime
    close_price: float
    features: dict[str, Any]

    def to_feature_input(
        self,
        *,
        feature_columns: tuple[str, ...],
    ) -> dict[str, Any]:
        """Return one flat feature input compatible with saved artifact contracts."""
        feature_input: dict[str, Any] = {}
        for column in feature_columns:
            if column == "symbol":
                feature_input[column] = self.symbol
            elif column == "close_price":
                feature_input[column] = self.close_price
            else:
                if column not in self.features:
                    raise ValueError(
                        f"Source feature row {self.row_id} is missing column {column}",
                    )
                feature_input[column] = self.features[column]
        return feature_input


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """Labeled M3 dataset plus a compact manifest describing its construction."""

    samples: tuple[DatasetSample, ...]
    source_rows: tuple[SourceFeatureRow, ...]
    source_schema: tuple[str, ...]
    manifest: dict[str, Any]
    feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        """Return the ordered sample timestamps used for walk-forward splitting."""
        return tuple(sample.as_of_time for sample in self.samples)

    @property
    def frequency_minutes(self) -> int:
        """Return the inferred canonical feature-row cadence in minutes."""
        return infer_source_frequency_minutes(self.source_rows)


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate the checked-in JSON config for the offline training run."""
    config_data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    raw_models = dict(config_data.get("models", {}))
    models = {
        str(model_name): dict(model_config)
        for model_name, model_config in raw_models.items()
    }
    legacy_models = sorted(
        model_name
        for model_name in models
        if model_name in LEGACY_ARCHIVED_MODEL_NAMES
    )
    if legacy_models:
        raise ValueError(
            "Legacy archived sklearn models are no longer allowed in the "
            "authoritative training configs: "
            f"{legacy_models}"
        )
    return TrainingConfig(
        source_table=str(config_data["source_table"]),
        symbols=tuple(str(symbol) for symbol in config_data["symbols"]),
        time_column=str(config_data["time_column"]),
        interval_column=str(config_data["interval_column"]),
        close_column=str(config_data["close_column"]),
        categorical_feature_columns=tuple(
            str(column) for column in config_data["categorical_feature_columns"]
        ),
        numeric_feature_columns=tuple(
            str(column) for column in config_data["numeric_feature_columns"]
        ),
        label_horizon_candles=int(config_data["label_horizon_candles"]),
        purge_gap_candles=int(config_data["purge_gap_candles"]),
        test_folds=int(config_data["test_folds"]),
        first_train_fraction=float(config_data["first_train_fraction"]),
        test_fraction=float(config_data["test_fraction"]),
        round_trip_fee_bps=float(config_data["round_trip_fee_bps"]),
        artifact_root=str(config_data["artifact_root"]),
        models=models,
    )


def load_training_dataset(
    config: TrainingConfig,
    *,
    parquet_dir: Path | None = None,
) -> TrainingDataset:
    """Load the configured source table from PostgreSQL and construct labeled samples.

    If *parquet_dir* is given, read from exported parquet files instead of PostgreSQL.
    """
    if parquet_dir is not None:
        dataset = _load_training_dataset_from_parquet(parquet_dir, config)
    else:
        dataset = load_training_dataset_preview(config)
    return _require_non_empty_training_dataset(dataset, config)


def load_training_dataset_preview(config: TrainingConfig) -> TrainingDataset:
    """Load the configured source table without requiring that labeled rows already exist."""
    settings = Settings.from_env()
    return asyncio.run(_load_training_dataset_with_fallback(settings, config))


def _load_training_dataset_from_parquet(
    parquet_dir: Path,
    config: TrainingConfig,
) -> TrainingDataset:
    """Load training data from exported parquet files instead of PostgreSQL."""
    import pyarrow.parquet as pq  # noqa: E402  # deferred import

    selected_columns = _selected_source_columns(config)
    all_rows: list[dict[str, Any]] = []

    for symbol in config.symbols:
        safe_symbol = symbol.replace("/", "_")
        symbol_dir = parquet_dir / safe_symbol
        if not symbol_dir.is_dir():
            raise ValueError(
                f"Parquet directory for symbol {symbol} not found at {symbol_dir}"
            )
        part_files = sorted(symbol_dir.glob("*.parquet"))
        if not part_files:
            raise ValueError(f"No parquet files found in {symbol_dir}")
        for part_file in part_files:
            table = pq.read_table(part_file, columns=selected_columns)
            df = table.to_pandas()
            all_rows.extend(df.to_dict(orient="records"))

    all_rows.sort(
        key=lambda row: (
            str(row["symbol"]),
            row[config.time_column],
            row[config.interval_column],
        )
    )

    source_schema = list(all_rows[0].keys()) if all_rows else selected_columns
    source_rows = _build_source_feature_rows(all_rows, config)
    samples, manifest = _build_labeled_samples(all_rows, config)
    ordered_samples = tuple(
        sorted(samples, key=lambda sample: (sample.as_of_time, sample.symbol))
    )
    return TrainingDataset(
        samples=ordered_samples,
        source_rows=source_rows,
        source_schema=tuple(source_schema),
        manifest=manifest,
        feature_columns=config.all_feature_columns,
        categorical_feature_columns=config.categorical_feature_columns,
        numeric_feature_columns=config.numeric_feature_columns,
    )


async def _load_training_dataset_with_fallback(
    settings: Settings,
    config: TrainingConfig,
) -> TrainingDataset:
    last_error: Exception | None = None
    for dsn in candidate_dsns(settings.postgres):
        try:
            return await _load_training_dataset(dsn, config)
        except (OSError, asyncpg.PostgresConnectionError) as error:
            last_error = error
            continue
    if last_error is None:
        raise ValueError("No PostgreSQL DSN candidates were available for dataset loading")
    raise ValueError(f"Could not connect to PostgreSQL for training: {last_error}") from last_error


async def _load_training_dataset(dsn: str, config: TrainingConfig) -> TrainingDataset:
    connection = await asyncpg.connect(dsn)
    try:
        source_schema = await _fetch_source_schema(connection, config.source_table)
        selected_columns = _selected_source_columns(config)
        _validate_source_columns(source_schema, selected_columns)
        rows = await _fetch_source_rows(connection, config, selected_columns)
    finally:
        await connection.close()

    source_rows = _build_source_feature_rows(rows, config)
    samples, manifest = _build_labeled_samples(rows, config)
    ordered_samples = tuple(sorted(samples, key=lambda sample: (sample.as_of_time, sample.symbol)))
    return TrainingDataset(
        samples=ordered_samples,
        source_rows=source_rows,
        source_schema=tuple(source_schema),
        manifest=manifest,
        feature_columns=config.all_feature_columns,
        categorical_feature_columns=config.categorical_feature_columns,
        numeric_feature_columns=config.numeric_feature_columns,
    )


def _require_non_empty_training_dataset(
    dataset: TrainingDataset,
    config: TrainingConfig,
) -> TrainingDataset:
    if int(dataset.manifest["loaded_rows"]) == 0:
        raise ValueError(
            f"No source rows found in {config.source_table} for symbols {list(config.symbols)}"
        )
    if dataset.samples:
        return dataset
    raise ValueError(
        "No eligible labeled rows were produced from feature_ohlc. "
        "Check that the table contains enough finalized feature rows."
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
    config: TrainingConfig,
    selected_columns: list[str],
) -> list[dict[str, Any]]:
    table_name = _quote_table_name(config.source_table)
    select_clause = ", ".join(_quote_identifier(column) for column in selected_columns)
    if config.symbols:
        rows = await connection.fetch(
            f"""
            SELECT {select_clause}
            FROM {table_name}
            WHERE symbol = ANY($1::text[])
            ORDER BY symbol ASC, {config.time_column} ASC, {config.interval_column} ASC
            """,
            list(config.symbols),
        )
    else:
        rows = await connection.fetch(
            f"""
            SELECT {select_clause}
            FROM {table_name}
            ORDER BY symbol ASC, {config.time_column} ASC, {config.interval_column} ASC
            """
        )
    return [dict(row) for row in rows]


def _selected_source_columns(config: TrainingConfig) -> list[str]:
    return _deduplicate(
        [
            "symbol",
            config.interval_column,
            config.time_column,
            config.close_column,
            *list(config.all_feature_columns),
        ]
    )


def candidate_dsns(postgres: PostgresSettings) -> tuple[str, ...]:
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


def _validate_source_columns(source_schema: list[str], selected_columns: list[str]) -> None:
    missing_columns = sorted(set(selected_columns) - set(source_schema))
    if missing_columns:
        raise ValueError(
            "Training source schema does not match the configured columns. "
            f"Missing columns: {missing_columns}"
        )


def _build_labeled_samples(
    rows: list[dict[str, Any]],
    config: TrainingConfig,
) -> tuple[list[DatasetSample], dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["symbol"])].append(row)

    samples: list[DatasetSample] = []
    dropped_flat_returns = 0
    dropped_persistence_warmup = 0
    symbol_counts: dict[str, int] = {}
    label_counts = {0: 0, 1: 0}

    for symbol, symbol_rows in sorted(grouped_rows.items()):
        ordered_rows = sorted(
            symbol_rows,
            key=lambda row: (row[config.time_column], row[config.interval_column]),
        )
        symbol_samples = _build_symbol_samples(ordered_rows, config)
        samples.extend(symbol_samples.samples)
        dropped_flat_returns += symbol_samples.dropped_flat_returns
        dropped_persistence_warmup += symbol_samples.dropped_persistence_warmup
        symbol_counts[symbol] = len(symbol_samples.samples)
        label_counts[0] += symbol_samples.label_counts[0]
        label_counts[1] += symbol_samples.label_counts[1]

    manifest = {
        "source_table": config.source_table,
        "loaded_rows": len(rows),
        "eligible_rows": len(samples),
        "symbols": symbol_counts,
        "label_counts": {str(label): count for label, count in label_counts.items()},
        "dropped_flat_future_return_rows": dropped_flat_returns,
        "dropped_persistence_warmup_rows": dropped_persistence_warmup,
        "unique_timestamps": len({sample.as_of_time for sample in samples}),
        "time_range": _time_range_payload(samples),
    }
    return samples, manifest


@dataclass(frozen=True, slots=True)
class SymbolSampleBuild:
    """Intermediate symbol-local sample construction result."""

    samples: tuple[DatasetSample, ...]
    dropped_flat_returns: int
    dropped_persistence_warmup: int
    label_counts: dict[int, int]


# pylint: disable=too-many-locals
def _build_symbol_samples(
    rows: list[dict[str, Any]],
    config: TrainingConfig,
) -> SymbolSampleBuild:
    samples: list[DatasetSample] = []
    dropped_flat_returns = 0
    dropped_persistence_warmup = 0
    label_counts = {0: 0, 1: 0}
    last_nonzero_direction: int | None = None
    horizon = config.label_horizon_candles

    for index, current_row in enumerate(rows):
        future_index = index + horizon
        if future_index >= len(rows):
            break
        if index < horizon:
            dropped_persistence_warmup += 1
            continue

        future_row = rows[future_index]
        current_close = float(current_row[config.close_column])
        future_close = float(future_row[config.close_column])
        future_return_3 = (future_close / current_close) - 1.0
        if future_return_3 == 0.0:
            dropped_flat_returns += 1
            continue

        label = 1 if future_return_3 > 0.0 else 0
        label_counts[label] += 1

        persistence_source = rows[index - horizon]
        realized_return_3 = (current_close / float(persistence_source[config.close_column])) - 1.0
        if realized_return_3 > 0.0:
            persistence_prediction = 1
            last_nonzero_direction = 1
        elif realized_return_3 < 0.0:
            persistence_prediction = 0
            last_nonzero_direction = 0
        else:
            persistence_prediction = 0 if last_nonzero_direction is None else last_nonzero_direction

        features = {
            column: current_row[column]
            for column in config.all_feature_columns
        }
        sample = DatasetSample(
            row_id=f"{current_row['symbol']}|{to_rfc3339(current_row[config.interval_column])}",
            symbol=str(current_row["symbol"]),
            interval_begin=current_row[config.interval_column],
            as_of_time=current_row[config.time_column],
            close_price=current_close,
            future_close_price=future_close,
            future_return_3=future_return_3,
            label=label,
            persistence_prediction=persistence_prediction,
            features=features,
        )
        samples.append(sample)

    return SymbolSampleBuild(
        samples=tuple(samples),
        dropped_flat_returns=dropped_flat_returns,
        dropped_persistence_warmup=dropped_persistence_warmup,
        label_counts=label_counts,
    )


def _time_range_payload(samples: list[DatasetSample]) -> dict[str, str | None]:
    if not samples:
        return {"min_as_of_time": None, "max_as_of_time": None}
    return {
        "min_as_of_time": to_rfc3339(min(sample.as_of_time for sample in samples)),
        "max_as_of_time": to_rfc3339(max(sample.as_of_time for sample in samples)),
    }


def _build_source_feature_rows(
    rows: list[dict[str, Any]],
    config: TrainingConfig,
) -> tuple[SourceFeatureRow, ...]:
    """Return the full ordered feature-table truth used by sequential models."""
    source_rows = [
        SourceFeatureRow(
            row_id=f"{row['symbol']}|{to_rfc3339(row[config.interval_column])}",
            symbol=str(row["symbol"]),
            interval_begin=row[config.interval_column],
            as_of_time=row[config.time_column],
            close_price=float(row[config.close_column]),
            features={
                column: row[column]
                for column in config.all_feature_columns
            },
        )
        for row in rows
    ]
    return tuple(
        sorted(
            source_rows,
            key=lambda source_row: (source_row.as_of_time, source_row.symbol),
        )
    )


def infer_source_frequency_minutes(
    source_rows: tuple[SourceFeatureRow, ...] | list[SourceFeatureRow],
) -> int:
    """Infer the canonical cadence from ordered feature rows."""
    by_symbol: dict[str, list[SourceFeatureRow]] = defaultdict(list)
    for source_row in source_rows:
        by_symbol[source_row.symbol].append(source_row)

    deltas: list[int] = []
    for symbol_rows in by_symbol.values():
        ordered_rows = sorted(symbol_rows, key=lambda row: row.as_of_time)
        for previous_row, current_row in zip(ordered_rows, ordered_rows[1:], strict=False):
            delta_seconds = int(
                (current_row.as_of_time - previous_row.as_of_time).total_seconds()
            )
            if delta_seconds <= 0:
                continue
            deltas.append(delta_seconds // 60)
            break
    if not deltas:
        raise ValueError("Could not infer source frequency from fewer than two source rows")
    return min(deltas)


def build_sequence_context_rows(
    *,
    target_samples: list[DatasetSample],
    source_rows: tuple[SourceFeatureRow, ...] | list[SourceFeatureRow],
    feature_columns: tuple[str, ...],
    lookback_candles: int,
) -> list[dict[str, Any]]:
    """Build ordered per-sample sequence contexts without leaking future rows."""
    if lookback_candles <= 0:
        raise ValueError("lookback_candles must be positive for sequence models")

    by_symbol: dict[str, list[SourceFeatureRow]] = defaultdict(list)
    for source_row in source_rows:
        by_symbol[source_row.symbol].append(source_row)
    for symbol, symbol_rows in by_symbol.items():
        by_symbol[symbol] = sorted(symbol_rows, key=lambda row: row.as_of_time)

    context_rows: list[dict[str, Any]] = []
    for sample in target_samples:
        history_rows = [
            row
            for row in by_symbol.get(sample.symbol, [])
            if row.as_of_time <= sample.as_of_time
        ]
        if len(history_rows) < lookback_candles:
            raise ValueError(
                "Sequence model does not have enough lookback rows for "
                f"{sample.row_id}: required {lookback_candles}, found {len(history_rows)}",
            )
        ordered_context = history_rows[-lookback_candles:]
        context_feature_row = {
            "symbol": sample.symbol,
            "as_of_time": sample.as_of_time,
            **{
                column: (
                    sample.symbol
                    if column == "symbol"
                    else sample.close_price
                    if column == "close_price"
                    else sample.features[column]
                )
                for column in feature_columns
            },
            SEQUENCE_CONTEXT_KEY: [
                {
                    "symbol": history_row.symbol,
                    "as_of_time": history_row.as_of_time,
                    **history_row.to_feature_input(feature_columns=feature_columns),
                }
                for history_row in ordered_context
            ],
        }
        context_rows.append(context_feature_row)
    return context_rows


def future_target_timestamp(
    *,
    as_of_time: datetime,
    horizon_candles: int,
    frequency_minutes: int,
) -> datetime:
    """Return the expected future timestamp for one labeled horizon."""
    return as_of_time + timedelta(minutes=horizon_candles * frequency_minutes)
