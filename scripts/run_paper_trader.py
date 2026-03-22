"""CLI entrypoint for the Stream Alpha M5 paper trader."""

# pylint: disable=duplicate-code,wrong-import-position

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.config import Settings
from app.common.logging import configure_logging
from app.runtime.config import resolve_trading_config_path
from app.trading.config import load_paper_trading_config
from app.trading.repository import TradingRepository
from app.trading.runner import PaperTradingRunner
from app.trading.signal_client import SignalClient


def main() -> None:
    """Run the M5 paper trader once or in polling mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="Path to the checked-in M5 paper trading config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single catch-up cycle and exit",
    )
    args = parser.parse_args()
    asyncio.run(
        _main_async(
            resolve_trader_config_path(args.config),
            run_once=args.once,
        )
    )


def resolve_trader_config_path(config_value: str | None) -> Path:
    """Resolve the trader config path from CLI override or runtime env."""
    if config_value is not None and config_value.strip():
        return Path(config_value).expanduser().resolve()
    return resolve_trading_config_path()


async def _main_async(config_path: Path, *, run_once: bool) -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    config = load_paper_trading_config(config_path)
    repository = TradingRepository(settings.postgres.dsn, config.source_table)
    signal_client = SignalClient(config.inference_base_url)
    runner = PaperTradingRunner(
        config=config,
        repository=repository,
        signal_client=signal_client,
    )
    await runner.startup()
    try:
        if run_once:
            await runner.run_once()
        else:
            await runner.run_forever()
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    main()
