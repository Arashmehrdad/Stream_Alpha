"""Long-running Kraken ingestion service orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from app.common.config import Settings
from app.common.models import HealthEvent, generate_event_id
from app.common.time import to_rfc3339, utc_now
from app.ingestion.db import PostgresWriter
from app.ingestion.kraken import (
    ack_key,
    build_public_subscription_requests,
    expected_subscription_keys,
)
from app.ingestion.normalizers import (
    NormalizationError,
    normalize_ohlc_payload,
    normalize_trade_payload,
)
from app.ingestion.publisher import KafkaEventPublisher


@dataclass(slots=True)
class ExponentialBackoff:
    """Simple exponential backoff with bounded jitter."""

    initial_delay_seconds: float
    max_delay_seconds: float
    multiplier: float
    jitter_seconds: float
    _attempts: int = 0

    def next_delay(self) -> float:
        """Return the next retry delay and advance the attempt counter."""
        base_delay = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * (self.multiplier ** self._attempts),
        )
        self._attempts += 1
        jitter = random.uniform(0.0, self.jitter_seconds)
        return base_delay + jitter

    def reset(self) -> None:
        """Reset the retry sequence after a successful connection."""
        self._attempts = 0


class ProducerService:  # pylint: disable=too-many-instance-attributes
    """Coordinate websocket ingestion, publishing, persistence, and health."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("streamalpha.ingestion")
        self.stop_event = asyncio.Event()
        self.publisher = KafkaEventPublisher(
            bootstrap_servers=settings.kafka.bootstrap_servers,
            client_id=settings.kafka.client_id,
        )
        self.db = PostgresWriter(
            dsn=settings.postgres.dsn,
            tables=settings.tables,
        )
        self._backoff = ExponentialBackoff(
            initial_delay_seconds=settings.retry.initial_delay_seconds,
            max_delay_seconds=settings.retry.max_delay_seconds,
            multiplier=settings.retry.multiplier,
            jitter_seconds=settings.retry.jitter_seconds,
        )
        self._kraken_connected = False
        self._payload_errors = 0
        self._last_exchange_activity_at = None
        self._session_counter = 0

    def request_stop(self) -> None:
        """Trigger a graceful service shutdown."""
        self.logger.info("Stop requested")
        self.stop_event.set()

    async def run(self) -> None:  # pylint: disable=broad-exception-caught
        """Run the service until shutdown is requested."""
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await self._ensure_dependencies()
        await self._emit_health(
            status="starting",
            component="producer",
            message="Producer service starting",
        )

        try:
            while not self.stop_event.is_set():
                try:
                    await self._run_stream_session()
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError("Kraken websocket session ended unexpectedly")
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    delay_seconds = self._backoff.next_delay()
                    self.logger.warning(
                        "Producer session failed; reconnect scheduled",
                        extra={
                            "error": str(exc),
                            "retry_in_seconds": round(delay_seconds, 3),
                        },
                        exc_info=True,
                    )
                    await self._emit_health(
                        status="reconnecting",
                        component="producer",
                        message="Producer session failed; reconnect scheduled",
                        details={
                            "error": str(exc),
                            "retry_in_seconds": round(delay_seconds, 3),
                            "session_counter": self._session_counter,
                        },
                    )
                    await self._reset_dependencies()
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), timeout=delay_seconds)
                    except asyncio.TimeoutError:
                        await self._ensure_dependencies()
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            await self._emit_health(
                status="stopping",
                component="producer",
                message="Producer service stopping",
            )
            await self._reset_dependencies()

    async def _ensure_dependencies(self) -> None:  # pylint: disable=broad-exception-caught
        """Wait for PostgreSQL and Redpanda dependencies to become available."""
        dependency_backoff = ExponentialBackoff(
            initial_delay_seconds=self.settings.retry.initial_delay_seconds,
            max_delay_seconds=self.settings.retry.max_delay_seconds,
            multiplier=self.settings.retry.multiplier,
            jitter_seconds=self.settings.retry.jitter_seconds,
        )
        while not self.stop_event.is_set():
            try:
                await self.db.connect()
                await self.publisher.start()
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                delay_seconds = dependency_backoff.next_delay()
                self.logger.warning(
                    "Dependency startup failed; retrying",
                    extra={
                        "error": str(exc),
                        "retry_in_seconds": round(delay_seconds, 3),
                    },
                    exc_info=True,
                )
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=delay_seconds)
                except asyncio.TimeoutError:
                    continue

    async def _reset_dependencies(self) -> None:
        """Close external clients without failing shutdown paths."""
        with suppress(Exception):
            await self.publisher.stop()
        with suppress(Exception):
            await self.db.close()

    async def _run_stream_session(self) -> None:
        """Open one Kraken websocket session and process messages until it ends."""
        self._session_counter += 1
        self.logger.info(
            "Connecting to Kraken websocket",
            extra={
                "url": self.settings.kraken.ws_url,
                "symbols": list(self.settings.kraken.symbols),
                "ohlc_interval_minutes": self.settings.kraken.ohlc_interval_minutes,
                "session_counter": self._session_counter,
            },
        )

        expected_acks = expected_subscription_keys(
            symbols=self.settings.kraken.symbols,
            ohlc_interval_minutes=self.settings.kraken.ohlc_interval_minutes,
        )
        received_acks: set[str] = set()

        try:
            async with websockets.connect(
                self.settings.kraken.ws_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1000,
            ) as websocket:
                self._kraken_connected = True
                self._backoff.reset()
                self._mark_exchange_activity()
                await self._emit_health(
                    status="connected",
                    component="kraken",
                    message="Connected to Kraken websocket",
                    details={"session_counter": self._session_counter},
                )
                await self._send_subscriptions(websocket)

                ack_deadline = asyncio.get_running_loop().time() + 20

                while not self.stop_event.is_set():
                    timed_out_waiting_for_ack = (
                        received_acks != expected_acks
                        and asyncio.get_running_loop().time() > ack_deadline
                    )
                    if timed_out_waiting_for_ack:
                        missing_acks = sorted(expected_acks - received_acks)
                        raise TimeoutError(
                            "Timed out waiting for subscription acknowledgements: "
                            f"{missing_acks}"
                        )

                    try:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except ConnectionClosed as exc:
                        raise ConnectionError(
                            f"Kraken websocket closed with code={exc.code} reason={exc.reason}"
                        ) from exc

                    await self._handle_message(raw_message, received_acks)
        finally:
            self._kraken_connected = False

    async def _send_subscriptions(self, websocket: Any) -> None:
        """Send the fixed public M1 subscription set to Kraken."""
        for index, request in enumerate(
            build_public_subscription_requests(
                symbols=self.settings.kraken.symbols,
                ohlc_interval_minutes=self.settings.kraken.ohlc_interval_minutes,
            ),
            start=1,
        ):
            payload = request.as_message(req_id=index)
            await websocket.send(json.dumps(payload))
            self.logger.info("Subscription request sent", extra={"request": payload})

    async def _handle_message(self, raw_message: str, received_acks: set[str]) -> None:
        """Parse and route one inbound websocket message."""
        received_at = utc_now()
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            await self._handle_bad_payload(
                channel="unknown",
                received_at=received_at,
                raw_message=raw_message,
                reason=f"JSON decode error: {exc}",
            )
            return

        if not isinstance(message, dict):
            await self._handle_bad_payload(
                channel="unknown",
                received_at=received_at,
                raw_message=raw_message,
                reason="Top-level websocket payload is not an object",
            )
            return

        if message.get("method") == "subscribe":
            if not message.get("success", False):
                raise RuntimeError(message.get("error", "Subscription failed"))
            key = ack_key(message)
            received_acks.add(key)
            self._mark_exchange_activity()
            self.logger.info(
                "Subscription acknowledged",
                extra={"ack_key": key, "ws_message": message},
            )
            return

        if message.get("success") is False:
            raise RuntimeError(message.get("error", "Kraken request failed"))

        await self._dispatch_channel_message(message, received_at)

    async def _dispatch_channel_message(self, message: dict[str, Any], received_at) -> None:
        """Dispatch a parsed websocket payload by Kraken channel name."""
        channel = message.get("channel")
        self._mark_exchange_activity()
        if channel == "heartbeat":
            return
        if channel == "status":
            await self._handle_status_message(message, received_at)
            return
        if channel == "trade":
            await self._handle_trade_message(message, received_at)
            return
        if channel == "ohlc":
            await self._handle_ohlc_message(message, received_at)
            return
        self.logger.debug("Ignoring websocket message", extra={"ws_message": message})

    async def _handle_trade_message(self, message: dict[str, Any], received_at) -> None:
        """Normalize, publish, and persist trade events from one message."""
        message_type = str(message.get("type", "unknown"))
        data = message.get("data")
        if not isinstance(data, list):
            raise NormalizationError("Trade message data field must be a list")
        for payload in data:
            if not isinstance(payload, dict):
                await self._handle_bad_payload(
                    channel="trade",
                    received_at=received_at,
                    raw_message=message,
                    reason="Trade entry is not an object",
                )
                continue
            try:
                event = normalize_trade_payload(
                    payload,
                    app_name=self.settings.app_name,
                    message_type=message_type,
                    received_at=received_at,
                )
            except NormalizationError as exc:
                await self._handle_bad_payload(
                    channel="trade",
                    received_at=received_at,
                    raw_message=payload,
                    reason=str(exc),
                )
                continue

            await self.publisher.publish(
                topic=self.settings.topics.raw_trades,
                key=f"{event.symbol}:{event.trade_id}",
                event=event,
            )
            await self.db.write_trade(event)

    async def _handle_ohlc_message(self, message: dict[str, Any], received_at) -> None:
        """Normalize, publish, and persist OHLC events from one message."""
        message_type = str(message.get("type", "unknown"))
        data = message.get("data")
        if not isinstance(data, list):
            raise NormalizationError("OHLC message data field must be a list")
        for payload in data:
            if not isinstance(payload, dict):
                await self._handle_bad_payload(
                    channel="ohlc",
                    received_at=received_at,
                    raw_message=message,
                    reason="OHLC entry is not an object",
                )
                continue
            try:
                event = normalize_ohlc_payload(
                    payload,
                    app_name=self.settings.app_name,
                    message_type=message_type,
                    received_at=received_at,
                )
            except NormalizationError as exc:
                await self._handle_bad_payload(
                    channel="ohlc",
                    received_at=received_at,
                    raw_message=payload,
                    reason=str(exc),
                )
                continue

            await self.publisher.publish(
                topic=self.settings.topics.raw_ohlc,
                key=f"{event.symbol}:{event.interval_begin.isoformat()}",
                event=event,
            )
            await self.db.write_ohlc(event)

    async def _handle_status_message(self, message: dict[str, Any], received_at) -> None:
        """Convert Kraken status updates into health events."""
        status_payload = {}
        data = message.get("data")
        if isinstance(data, list) and data:
            first_entry = data[0]
            if isinstance(first_entry, dict):
                status_payload = first_entry

        await self._emit_health(
            status="exchange_status",
            component="kraken",
            message="Received Kraken status update",
            observed_at=received_at,
            details=status_payload,
            source_exchange="kraken",
        )

    async def _handle_bad_payload(
        self,
        *,
        channel: str,
        received_at,
        raw_message: Any,
        reason: str,
    ) -> None:
        """Record a malformed payload without terminating the process."""
        self._payload_errors += 1
        self.logger.warning(
            "Bad payload isolated",
            extra={
                "channel": channel,
                "reason": reason,
                "payload_errors": self._payload_errors,
                "raw_message": raw_message,
            },
        )
        await self._emit_health(
            status="payload_error",
            component=channel,
            message="Bad payload isolated",
            observed_at=received_at,
            source_exchange="kraken",
            details={
                "reason": reason,
                "payload_errors": self._payload_errors,
            },
        )

    async def _heartbeat_loop(self) -> None:
        """Emit periodic producer heartbeat events until shutdown."""
        while not self.stop_event.is_set():
            observed_at = utc_now()
            status = "healthy" if self._kraken_connected else "degraded"
            details = {
                "kraken_connected": self._kraken_connected,
                "payload_errors": self._payload_errors,
                "last_exchange_activity_at": (
                    to_rfc3339(self._last_exchange_activity_at)
                    if self._last_exchange_activity_at is not None
                    else None
                ),
            }
            await self._emit_health(
                status=status,
                component="producer",
                message="Producer heartbeat",
                observed_at=observed_at,
                details=details,
            )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.settings.heartbeat_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def _emit_health(  # pylint: disable=too-many-arguments
        self,
        *,
        status: str,
        component: str,
        message: str,
        details: dict[str, Any] | None = None,
        observed_at=None,
        source_exchange: str | None = None,
    ) -> None:
        """Best-effort publish and persist a producer health event."""
        event = HealthEvent(
            event_id=generate_event_id(),
            app_name=self.settings.app_name,
            service_name=self.settings.service_name,
            status=status,
            component=component,
            message=message,
            observed_at=observed_at or utc_now(),
            source_exchange=source_exchange,
            details=details or {},
        )

        with suppress(Exception):
            await self.publisher.publish(
                topic=self.settings.topics.raw_health,
                key=f"{event.service_name}:{event.status}",
                event=event,
            )
        with suppress(Exception):
            await self.db.write_heartbeat(event)

    def _mark_exchange_activity(self) -> None:
        """Track the latest successful message or status activity time."""
        self._last_exchange_activity_at = utc_now()
