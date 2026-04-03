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
from app.training.data_readiness import build_data_readiness_report
from app.training.dataset import (
    TrainingConfig,
    candidate_dsns,
    load_training_config,
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
    fastai_installed: bool
    fastai_version: str | None
    fastai_usable: bool
    fastai_detail: str | None
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


@dataclass(frozen=True, slots=True)
class _OptionalBreadthStatus:
    installed: bool
    version: str | None
    usable: bool
    detail: str | None


def build_training_readiness_report(config_path: Path) -> TrainingReadinessReport:
    """Inspect local M7 training readiness without starting a training run."""
    resolved_config_path = Path(config_path).resolve()
    autogluon_version = _resolve_autogluon_version()
    fastai_status = _resolve_fastai_status()
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
            fastai_installed=fastai_status.installed,
            fastai_version=fastai_status.version,
            fastai_usable=fastai_status.usable,
            fastai_detail=fastai_status.detail,
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
        data_readiness = build_data_readiness_report(
            config,
            config_path=resolved_config_path,
        )
    except ValueError as error:
        return TrainingReadinessReport(
            config_path=str(resolved_config_path),
            config_ok=True,
            config_error=None,
            artifact_root=config.artifact_root,
            source_table=config.source_table,
            autogluon_installed=autogluon_version is not None,
            autogluon_version=autogluon_version,
            fastai_installed=fastai_status.installed,
            fastai_version=fastai_status.version,
            fastai_usable=fastai_status.usable,
            fastai_detail=fastai_status.detail,
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

    return TrainingReadinessReport(
        config_path=str(resolved_config_path),
        config_ok=True,
        config_error=None,
        artifact_root=config.artifact_root,
        source_table=config.source_table,
        autogluon_installed=autogluon_version is not None,
        autogluon_version=autogluon_version,
        fastai_installed=fastai_status.installed,
        fastai_version=fastai_status.version,
        fastai_usable=fastai_status.usable,
        fastai_detail=fastai_status.detail,
        postgres_reachable=data_readiness.postgres_reachable,
        postgres_error=data_readiness.postgres_error,
        feature_table_exists=data_readiness.feature_table_exists,
        row_count=data_readiness.feature_rows_total,
        eligible_rows=data_readiness.labeled_rows_total,
        unique_timestamps=data_readiness.unique_timestamps,
        required_unique_timestamps=required_unique_timestamps,
        ready_for_training=data_readiness.ready_for_training,
        readiness_detail=data_readiness.readiness_detail,
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
        version = _resolve_package_version(distribution_name)
        if version is not None:
            return version
    return None


def _resolve_package_version(distribution_name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _resolve_fastai_status() -> _OptionalBreadthStatus:
    """Return whether optional FastAI breadth is both installed and actually usable."""
    fastai_version = _resolve_package_version("fastai")
    if fastai_version is None:
        return _OptionalBreadthStatus(
            installed=False,
            version=None,
            usable=False,
            detail="missing optional breadth only, not a blocker",
        )
    try:
        from autogluon.common.utils.try_import import try_import_fastai

        try_import_fastai()
    except (ImportError, ModuleNotFoundError) as error:
        return _OptionalBreadthStatus(
            installed=True,
            version=fastai_version,
            usable=False,
            detail=_summarize_fastai_failure(error),
        )
    return _OptionalBreadthStatus(
        installed=True,
        version=fastai_version,
        usable=True,
        detail="optional breadth available",
    )


def _summarize_fastai_failure(error: BaseException) -> str:
    """Surface the real FastAI import blocker instead of a generic installed/missing guess."""
    missing_module = _first_missing_module(error)
    if missing_module == "IPython":
        return "installed but unusable for AutoGluon because IPython is missing"
    summary = _first_non_empty_message(error)
    if summary is None:
        return type(error).__name__
    return f"installed but unusable for AutoGluon: {summary}"


def _first_missing_module(error: BaseException) -> str | None:
    """Return the first missing module name seen in the exception chain."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name:
            return current.name
        current = _next_exception_in_chain(current)
    return None


def _first_non_empty_message(error: BaseException) -> str | None:
    """Return the first non-empty message from the exception chain."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).strip()
        if message:
            return message
        current = _next_exception_in_chain(current)
    return None


def _next_exception_in_chain(error: BaseException) -> BaseException | None:
    """Walk the causal chain without looping forever."""
    return error.__cause__ or error.__context__


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
    print(f"fastai_version={report.fastai_version or 'missing_optional'}")
    print(f"fastai_usable={report.fastai_usable}")
    if report.fastai_detail:
        print(f"fastai_detail={report.fastai_detail}")
    print(f"postgres_reachable={report.postgres_reachable}")
    print(f"feature_table_exists={report.feature_table_exists}")
    print(f"row_count={report.row_count}")
    print(f"ready_for_training={report.ready_for_training}")
    if report.readiness_detail:
        print(f"detail={report.readiness_detail}")


if __name__ == "__main__":
    main()
