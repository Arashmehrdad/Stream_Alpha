"""CLI entrypoint for one local M18 evaluation run."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.common.config import Settings
from app.common.time import parse_rfc3339, utc_now
from app.evaluation.config import default_evaluation_config_path
from app.evaluation.repository import EvaluationRepository
from app.evaluation.schemas import COMPARISON_FAMILIES, EvaluationRequest
from app.evaluation.service import EvaluationService, default_evaluation_run_id
from app.runtime.config import resolve_trading_config_path
from app.trading.config import load_paper_trading_config


def main() -> None:
    """Generate one canonical M18 evaluation run."""
    parser = argparse.ArgumentParser(description="Generate one Stream Alpha M18 evaluation run")
    parser.add_argument("--start", required=True, help="RFC3339 start time")
    parser.add_argument("--end", required=True, help="RFC3339 end time")
    parser.add_argument(
        "--config",
        default="",
        help="Trading config path used for service_name, symbols, and source exchange",
    )
    parser.add_argument(
        "--modes",
        default="paper,shadow,live",
        help="Comma-separated execution modes to include",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional explicit evaluation run id",
    )
    parser.add_argument(
        "--evaluation-config",
        default="",
        help="Optional M18 evaluation config path",
    )
    arguments = parser.parse_args()
    generated_at = utc_now()
    config_path = resolve_trading_config_path(
        arguments.config or None,
        use_profile_default=not bool(arguments.config),
    )
    trading_config = load_paper_trading_config(Path(config_path))
    settings = Settings.from_env()
    requested_modes = tuple(
        part.strip().lower()
        for part in arguments.modes.split(",")
        if part.strip()
    )
    request = EvaluationRequest(
        service_name=trading_config.service_name,
        source_exchange=trading_config.source_exchange,
        interval_minutes=trading_config.interval_minutes,
        symbols=trading_config.symbols,
        execution_modes=requested_modes,
        comparison_families=COMPARISON_FAMILIES,
        window_start=parse_rfc3339(arguments.start),
        window_end=parse_rfc3339(arguments.end),
        trading_config_path=str(Path(config_path).resolve()),
        evaluation_run_id=arguments.run_id or default_evaluation_run_id(generated_at=generated_at),
        generated_at=generated_at,
    )
    evaluation_config_path = (
        default_evaluation_config_path()
        if not arguments.evaluation_config
        else Path(arguments.evaluation_config)
    )
    asyncio.run(
        _run(
            request=request,
            dsn=settings.postgres.dsn,
            evaluation_config_path=evaluation_config_path,
        )
    )


async def _run(
    *,
    request: EvaluationRequest,
    dsn: str,
    evaluation_config_path: Path,
) -> None:
    repository = EvaluationRepository(dsn)
    await repository.connect()
    try:
        service = EvaluationService(
            repository=repository,
            evaluation_config_path=evaluation_config_path,
        )
        result = await service.generate_run(request)
    finally:
        await repository.close()
    print(result["artifact_paths"]["evaluation_report_json"])


if __name__ == "__main__":
    main()
