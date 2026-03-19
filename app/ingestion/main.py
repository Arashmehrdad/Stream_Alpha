"""Main entrypoint for the Stream Alpha producer service."""

from __future__ import annotations

import asyncio
import signal

from app.common.config import Settings
from app.common.logging import configure_logging
from app.ingestion.service import ProducerService


async def _run_service() -> None:
    """Initialize configuration, logging, and signal handling."""
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    service = ProducerService(settings)
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, service.request_stop)
        except NotImplementedError:
            break
    await service.run()


def main() -> None:
    """Run the producer service until shutdown."""
    asyncio.run(_run_service())
