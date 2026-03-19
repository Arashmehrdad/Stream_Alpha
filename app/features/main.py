"""Main entrypoint for the Stream Alpha M2 feature consumer."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import signal

from app.common.config import Settings
from app.common.logging import configure_logging
from app.features.service import FeatureConsumerService


async def _run_service() -> None:
    """Initialize settings, logging, and signal handling for the feature service."""
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    service = FeatureConsumerService(settings)
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, service.request_stop)
        except NotImplementedError:
            break
    await service.run()


def main() -> None:
    """Run the feature consumer until shutdown."""
    asyncio.run(_run_service())


if __name__ == "__main__":
    main()
