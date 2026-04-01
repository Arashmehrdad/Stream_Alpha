"""Operator-friendly M7 training readiness checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import asyncpg

from app.common.config import Settings
from app.common.serialization import make_json_safe
from app.training.dataset import (
    TrainingConfig,
    candidate_dsns,
    load_training_config,
    load_training_dataset,
)
from app.training.splits import minimum_required_unique_timestamps


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class TrainingReadinessReport:
    """Structured local training readiness report for scripts and operators."""

    config_path: str
    config_ok: bool
    config_error: str | None
    artifact_root: str | None
    source_table: str | None
    autogluon_installed: bool
    autogluon_version: str | None
    postgres_reachable: bool
    postgres_error: str | None
    feature_table_exists: bool | None
    row_count: int | None
    eligible_rows: int | None
    unique_timestamps: int | None
    required_unique_timestamps: int | None
    ready_for_training: bool
    readiness_detail: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for scripts."""
        return make_json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class _TrainingSourceProbe:
    postgres_reachable: bool
    postgres_error: str | None
    feature_table_exists: bool | None
    row_count: int | None


def build_training_readiness_report(config_path: Path) -> TrainingReadinessReport:
    """Inspect local M7 training readiness without starting a training run."""
    resolved_config_path = Path(config_path).resolve()
    autogluon_version = _resolve_autogluon_version()
    config: TrainingConfig | None = None
    config_error: str | None = None
    try:
        config = load_training_config(resolved_config_path)
    except (OSError, ValueError) as error:
        config_error = str(error)

    if config is None:
        return TrainingReadinessReport(
            config_path=str(resolved_config_path),
            config_ok=False,
            config_error=config_error,
            artifact_root=None,
            source_table=None,
            autogluon_installed=autogluon_version is not None,
            autogluon_version=autogluon_version,
            postgres_reachable=False,
            postgres_error="Training config must load before PostgreSQL readiness can be checked",
            feature_table_exists=None,
            row_count=None,
            eligible_rows=None,
            unique_timestamps=None,
            required_unique_timestamps=None,
            ready_for_training=False,
            readiness_detail="Training config is not loadable",
        )

    required_unique_timestamps = minimum_required_unique_timestamps(
        first_train_fraction=config.first_train_fraction,
        test_fraction=config.test_fraction,
        test_folds=config.test_folds,
        purge_gap_candles=config.purge_gap_candles,
    )
    try:
        settings = Settings.from_env()
    except ValueError as error:
        return TrainingReadinessReport(
            config_path=str(resolved_config_path),
            config_ok=True,
            config_error=None,
            artifact_root=config.artifact_root,
            source_table=config.source_table,
            autogluon_installed=autogluon_version is not None,
            autogluon_version=autogluon_version,
            postgres_reachable=False,
            postgres_error=str(error),
            feature_table_exists=None,
            row_count=None,
            eligible_rows=None,
            unique_timestamps=None,
            required_unique_timestamps=required_unique_timestamps,
            ready_for_training=False,
            readiness_detail="Environment settings could not be resolved for training",
        )

    probe = asyncio.run(
        _probe_training_source_with_fallback(
            postgres=settings.postgres,
            table_name=config.source_table,
        )
    )
    eligible_rows: int | None = None
    unique_timestamps: int | None = None
    ready_for_training = False
    readiness_detail: str | None = None
    if not probe.postgres_reachable:
        readiness_detail = "PostgreSQL is not reachable for training readiness checks"
    elif probe.feature_table_exists is False:
        readiness_detail = f"Training source table {config.source_table} does not exist"
    elif probe.row_count in {None, 0}:
        readiness_detail = f"Training source table {config.source_table} has no rows yet"
    else:
        try:
            dataset = load_training_dataset(config)
            eligible_rows = int(dataset.manifest["eligible_rows"])
            unique_timestamps = int(dataset.manifest["unique_timestamps"])
            ready_for_training = unique_timestamps >= required_unique_timestamps
            if ready_for_training:
                readiness_detail = (
                    "feature_ohlc satisfies the configured walk-forward timestamp requirement"
                )
            else:
                readiness_detail = (
                    "feature_ohlc does not yet satisfy the configured walk-forward "
                    f"timestamp requirement ({unique_timestamps}/{required_unique_timestamps})"
                )
        except ValueError as error:
            readiness_detail = str(error)

    return TrainingReadinessReport(
        config_path=str(resolved_config_path),
        config_ok=True,
        config_error=None,
        artifact_root=config.artifact_root,
        source_table=config.source_table,
        autogluon_installed=autogluon_version is not None,
        autogluon_version=autogluon_version,
        postgres_reachable=probe.postgres_reachable,
        postgres_error=probe.postgres_error,
        feature_table_exists=probe.feature_table_exists,
        row_count=probe.row_count,
        eligible_rows=eligible_rows,
        unique_timestamps=unique_timestamps,
        required_unique_timestamps=required_unique_timestamps,
        ready_for_training=ready_for_training,
        readiness_detail=readiness_detail,
    )


async def _probe_training_source_with_fallback(
    *,
    postgres,
    table_name: str,
) -> _TrainingSourceProbe:
    last_error: Exception | None = None
    for dsn in candidate_dsns(postgres):
        try:
            return await _probe_training_source(dsn=dsn, table_name=table_name)
        except (OSError, asyncpg.PostgresConnectionError) as error:
            last_error = error
            continue
    if last_error is None:
        return _TrainingSourceProbe(
            postgres_reachable=False,
            postgres_error="No PostgreSQL DSN candidates were available for readiness checks",
            feature_table_exists=None,
            row_count=None,
        )
    return _TrainingSourceProbe(
        postgres_reachable=False,
        postgres_error=str(last_error),
        feature_table_exists=None,
        row_count=None,
    )


async def _probe_training_source(
    *,
    dsn: str,
    table_name: str,
) -> _TrainingSourceProbe:
    connection = await asyncpg.connect(dsn)
    try:
        schema_name, relation_name = _split_table_name(table_name)
        exists = bool(
            await connection.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = $1 AND table_name = $2
                )
                """,
                schema_name,
                relation_name,
            )
        )
        if not exists:
            return _TrainingSourceProbe(
                postgres_reachable=True,
                postgres_error=None,
                feature_table_exists=False,
                row_count=0,
            )
        row_count = int(
            await connection.fetchval(
                f"SELECT COUNT(*) FROM {_quote_table_name(table_name)}"
            )
        )
        return _TrainingSourceProbe(
            postgres_reachable=True,
            postgres_error=None,
            feature_table_exists=True,
            row_count=row_count,
        )
    finally:
        await connection.close()


def _resolve_autogluon_version() -> str | None:
    """Return the installed AutoGluon version when available."""
    for distribution_name in ("autogluon.tabular", "autogluon"):
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _split_table_name(table_name: str) -> tuple[str, str]:
    parts = table_name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {table_name}")
    if len(parts) == 1:
        return "public", parts[0]
    return parts[0], parts[1]


def _quote_table_name(name: str) -> str:
    schema_name, relation_name = _split_table_name(name)
    return ".".join((_quote_identifier(schema_name), _quote_identifier(relation_name)))


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def main() -> None:
    """Run one local training readiness check."""
    parser = argparse.ArgumentParser(description="Inspect Stream Alpha local training readiness")
    parser.add_argument("--config", required=True, help="Path to the JSON training config")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the readiness report as JSON",
    )
    arguments = parser.parse_args()

    report = build_training_readiness_report(Path(arguments.config))
    if arguments.json:
        print(json.dumps(report.to_dict(), sort_keys=True))
        return

    print(f"config_ok={report.config_ok}")
    print(f"autogluon_version={report.autogluon_version or 'missing'}")
    print(f"postgres_reachable={report.postgres_reachable}")
    print(f"feature_table_exists={report.feature_table_exists}")
    print(f"row_count={report.row_count}")
    print(f"ready_for_training={report.ready_for_training}")
    if report.readiness_detail:
        print(f"detail={report.readiness_detail}")


if __name__ == "__main__":
    main()
