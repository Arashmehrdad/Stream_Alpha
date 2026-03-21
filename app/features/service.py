"""Kafka-backed OHLC feature consumer for Stream Alpha M2."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from aiokafka import AIOKafkaConsumer

from app.common.config import Settings
from app.common.time import utc_now
from app.features.db import FeatureStore
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.models import deserialize_ohlc_event
from app.features.state import FeatureStateManager
from app.reliability.config import default_reliability_config_path, load_reliability_config
from app.reliability.schemas import ServiceHeartbeat
from app.reliability.service import SERVICE_HEARTBEAT_DEGRADED, SERVICE_HEARTBEAT_HEALTHY
from app.reliability.store import ReliabilityStore


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
        """Reset the retry sequence after a successful run."""
        self._attempts = 0


class FeatureConsumerService:  # pylint: disable=too-many-instance-attributes
    """Consume raw OHLC events, finalize candles, and persist rolling features."""

    def __init__(self, settings: Settings) -> None:
        bootstrap_candles = max(
            settings.features.bootstrap_candles,
            MIN_FINALIZED_CANDLES,
        )
        self.settings = settings
        self.logger = logging.getLogger(f"{settings.app_name}.features")
        self.stop_event = asyncio.Event()
        self.db = FeatureStore(settings.postgres.dsn, settings.tables)
        self.reliability_config = load_reliability_config(
            default_reliability_config_path()
        )
        self.reliability_store = ReliabilityStore(settings.postgres.dsn)
        self.state = FeatureStateManager(
            grace_seconds=settings.features.finalization_grace_seconds,
            history_limit=bootstrap_candles,
        )
        self._bootstrap_candles = bootstrap_candles
        self._consumer: AIOKafkaConsumer | None = None
        self._backoff = ExponentialBackoff(
            initial_delay_seconds=settings.retry.initial_delay_seconds,
            max_delay_seconds=settings.retry.max_delay_seconds,
            multiplier=settings.retry.multiplier,
            jitter_seconds=settings.retry.jitter_seconds,
        )
        self._sweep_interval_seconds = max(
            1,
            min(settings.features.finalization_grace_seconds, 5),
        )
        self._last_feature_write_at = None
        self._last_event_received_at = None
        self._persisted_feature_rows = 0

    def request_stop(self) -> None:
        """Trigger a graceful service shutdown."""
        self.logger.info("Stop requested")
        self.stop_event.set()

    async def run(self) -> None:  # pylint: disable=broad-exception-caught
        """Run bootstrap, consumption, and stale-candle sweeps until shutdown."""
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            while not self.stop_event.is_set():
                try:
                    await self._connect_database()
                    await self._bootstrap_state()
                    await self._start_consumer()
                    self._backoff.reset()
                    await self._consume_loop()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if self.stop_event.is_set():
                        break
                    delay_seconds = self._backoff.next_delay()
                    self.logger.warning(
                        "Feature consumer failed; retrying",
                        extra={
                            "error": str(exc),
                            "retry_in_seconds": round(delay_seconds, 3),
                        },
                        exc_info=True,
                    )
                    await self._reset_dependencies()
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), timeout=delay_seconds)
                    except asyncio.TimeoutError:
                        continue
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            await self._reset_dependencies()

    async def _connect_database(self) -> None:
        await self.db.connect()
        await self.reliability_store.connect()

    async def _bootstrap_state(self) -> None:
        bootstrap_now = utc_now()
        candles = await self.db.load_bootstrap_candles(self._bootstrap_candles)
        feature_rows = self.state.bootstrap(
            candles,
            now=bootstrap_now,
            computed_at=bootstrap_now,
        )
        for row in feature_rows:
            await self.db.upsert_feature_row(row)
            self._last_feature_write_at = row.as_of_time
            self._persisted_feature_rows += 1
        self.logger.info(
            "Feature bootstrap complete",
            extra={
                "loaded_raw_candles": len(candles),
                "upserted_feature_rows": len(feature_rows),
                "bootstrap_candles": self._bootstrap_candles,
            },
        )

    async def _start_consumer(self) -> None:
        if self._consumer is not None:
            return
        self._consumer = AIOKafkaConsumer(
            self.settings.topics.raw_ohlc,
            bootstrap_servers=self.settings.kafka.bootstrap_servers,
            client_id=f"{self.settings.features.service_name}-consumer",
            group_id=self.settings.features.consumer_group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda value: value,
        )
        await self._consumer.start()
        self.logger.info(
            "Feature consumer started",
            extra={
                "topic": self.settings.topics.raw_ohlc,
                "group_id": self.settings.features.consumer_group_id,
            },
        )

    async def _consume_loop(self) -> None:
        consumer = self._require_consumer()
        sweep_task = asyncio.create_task(self._sweep_loop())
        try:
            while not self.stop_event.is_set():
                batches = await consumer.getmany(timeout_ms=1000, max_records=100)
                await self._process_batch(batches)
        finally:
            sweep_task.cancel()
            with suppress(asyncio.CancelledError):
                await sweep_task

    async def _process_batch(
        self,
        batches: Mapping[Any, Sequence[Any]],
    ) -> None:
        record_count = 0
        for records in batches.values():
            record_count += len(records)
            for record in records:
                await self._handle_record(record.value)
        if record_count > 0:
            await self._require_consumer().commit()

    async def _handle_record(self, value: bytes) -> None:
        try:
            event = deserialize_ohlc_event(value)
        except ValueError as exc:
            self.logger.warning(
                "Skipping malformed raw OHLC payload",
                extra={"error": str(exc)},
            )
            return

        self._last_event_received_at = event.received_at
        feature_rows = self.state.apply_event(event, computed_at=utc_now())
        for row in feature_rows:
            await self.db.upsert_feature_row(row)
            self._last_feature_write_at = row.as_of_time
            self._persisted_feature_rows += 1

    async def _sweep_loop(self) -> None:
        while not self.stop_event.is_set():
            current_time = utc_now()
            feature_rows = self.state.sweep(now=current_time, computed_at=current_time)
            for row in feature_rows:
                await self.db.upsert_feature_row(row)
                self._last_feature_write_at = row.as_of_time
                self._persisted_feature_rows += 1

            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self._sweep_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def _reset_dependencies(self) -> None:
        consumer = self._consumer
        self._consumer = None
        if consumer is not None:
            with suppress(Exception):
                await consumer.stop()
        with suppress(Exception):
            await self.db.close()
        with suppress(Exception):
            await self.reliability_store.close()

    def _require_consumer(self) -> AIOKafkaConsumer:
        if self._consumer is None:
            raise RuntimeError("Feature consumer has not been started")
        return self._consumer

    async def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            observed_at = utc_now()
            detail = {
                "last_event_received_at": (
                    None
                    if self._last_event_received_at is None
                    else self._last_event_received_at.isoformat().replace("+00:00", "Z")
                ),
                "last_feature_write_at": (
                    None
                    if self._last_feature_write_at is None
                    else self._last_feature_write_at.isoformat().replace("+00:00", "Z")
                ),
                "persisted_feature_rows": self._persisted_feature_rows,
            }
            with suppress(Exception):
                await self.reliability_store.save_service_heartbeat(
                    ServiceHeartbeat(
                        service_name=self.settings.features.service_name,
                        component_name="features",
                        heartbeat_at=observed_at,
                        health_overall_status=(
                            "HEALTHY"
                            if self._last_feature_write_at is not None
                            else "DEGRADED"
                        ),
                        reason_code=(
                            SERVICE_HEARTBEAT_HEALTHY
                            if self._last_feature_write_at is not None
                            else SERVICE_HEARTBEAT_DEGRADED
                        ),
                        detail=str(detail),
                    )
                )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.reliability_config.heartbeat.write_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue
