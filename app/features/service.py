"""Kafka-backed OHLC feature consumer for Stream Alpha M2."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aiokafka import AIOKafkaConsumer

from app.common.config import Settings
from app.common.time import to_rfc3339, utc_now
from app.features.db import FeatureStore
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.models import FeatureOhlcRow
from app.features.models import deserialize_ohlc_event
from app.features.state import FeatureStateManager
from app.reliability.artifacts import append_jsonl_artifact, write_json_artifact
from app.reliability.config import default_reliability_config_path, load_reliability_config
from app.reliability.schemas import FeatureLagSnapshot, RecoveryEvent, ServiceHeartbeat
from app.reliability.service import (
    FEATURE_LAG_BREACH,
    FEATURE_LAG_BREACH_CLEARED,
    FEATURE_LAG_BREACH_DETECTED,
    FEATURE_LAG_OK,
    SERVICE_HEARTBEAT_DEGRADED,
    SERVICE_HEARTBEAT_HEALTHY,
    evaluate_feature_consumer_lag,
)
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
        self._lag_component_name = "features"
        self._last_feature_write_at = None
        self._last_event_received_at = None
        self._persisted_feature_rows = 0
        self._latest_raw_event_received_at_by_symbol: dict[str, datetime] = {}
        self._latest_feature_interval_begin_by_symbol: dict[str, datetime] = {}
        self._latest_feature_as_of_time_by_symbol: dict[str, datetime] = {}
        self._lag_breach_by_symbol: dict[str, bool] = {}

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
        for candle in candles:
            self._record_raw_event_received(
                symbol=candle.symbol,
                received_at=candle.received_at,
            )
        feature_rows = self.state.bootstrap(
            candles,
            now=bootstrap_now,
            computed_at=bootstrap_now,
        )
        for row in feature_rows:
            await self.db.upsert_feature_row(row)
            self._record_feature_row_persisted(row)
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

        self._record_raw_event_received(
            symbol=event.symbol,
            received_at=event.received_at,
        )
        feature_rows = self.state.apply_event(event, computed_at=utc_now())
        for row in feature_rows:
            await self.db.upsert_feature_row(row)
            self._record_feature_row_persisted(row)

    async def _sweep_loop(self) -> None:
        while not self.stop_event.is_set():
            current_time = utc_now()
            feature_rows = self.state.sweep(now=current_time, computed_at=current_time)
            for row in feature_rows:
                await self.db.upsert_feature_row(row)
                self._record_feature_row_persisted(row)

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
            lag_snapshots: list[FeatureLagSnapshot] = []
            with suppress(Exception):
                lag_snapshots = await self._sync_feature_lag_state(observed_at)
            lag_breach_active = any(snapshot.lag_breach for snapshot in lag_snapshots)
            lag_breach_symbols = [
                snapshot.symbol
                for snapshot in lag_snapshots
                if snapshot.lag_breach
            ]
            if lag_breach_active:
                heartbeat_status = "DEGRADED"
                heartbeat_reason_code = FEATURE_LAG_BREACH
            elif self._last_feature_write_at is not None:
                heartbeat_status = "HEALTHY"
                heartbeat_reason_code = SERVICE_HEARTBEAT_HEALTHY
            else:
                heartbeat_status = "DEGRADED"
                heartbeat_reason_code = SERVICE_HEARTBEAT_DEGRADED
            detail = {
                "last_event_received_at": (
                    None
                    if self._last_event_received_at is None
                    else to_rfc3339(self._last_event_received_at)
                ),
                "last_feature_write_at": (
                    None
                    if self._last_feature_write_at is None
                    else to_rfc3339(self._last_feature_write_at)
                ),
                "persisted_feature_rows": self._persisted_feature_rows,
                "lag_breach_active": lag_breach_active,
                "lag_breach_symbols": lag_breach_symbols,
            }
            with suppress(Exception):
                await self.reliability_store.save_service_heartbeat(
                    ServiceHeartbeat(
                        service_name=self.settings.features.service_name,
                        component_name=self._lag_component_name,
                        heartbeat_at=observed_at,
                        health_overall_status=heartbeat_status,
                        reason_code=heartbeat_reason_code,
                        detail=json.dumps(detail, sort_keys=True),
                    )
                )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.reliability_config.heartbeat.write_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    def _record_raw_event_received(
        self,
        *,
        symbol: str,
        received_at: datetime,
    ) -> None:
        existing = self._latest_raw_event_received_at_by_symbol.get(symbol)
        if existing is None or received_at > existing:
            self._latest_raw_event_received_at_by_symbol[symbol] = received_at
        if self._last_event_received_at is None or received_at > self._last_event_received_at:
            self._last_event_received_at = received_at

    def _record_feature_row_persisted(self, row: FeatureOhlcRow) -> None:
        self._latest_feature_interval_begin_by_symbol[row.symbol] = row.interval_begin
        self._latest_feature_as_of_time_by_symbol[row.symbol] = row.as_of_time
        if self._last_feature_write_at is None or row.as_of_time > self._last_feature_write_at:
            self._last_feature_write_at = row.as_of_time
        self._persisted_feature_rows += 1

    def _build_feature_lag_snapshots(
        self,
        evaluated_at: datetime,
    ) -> list[FeatureLagSnapshot]:
        return [
            evaluate_feature_consumer_lag(
                service_name=self.settings.features.service_name,
                component_name=self._lag_component_name,
                symbol=symbol,
                evaluated_at=evaluated_at,
                latest_raw_event_received_at=self._latest_raw_event_received_at_by_symbol.get(
                    symbol
                ),
                latest_feature_interval_begin=self._latest_feature_interval_begin_by_symbol.get(
                    symbol
                ),
                latest_feature_as_of_time=self._latest_feature_as_of_time_by_symbol.get(
                    symbol
                ),
                feature_time_lag_max_seconds=(
                    self.reliability_config.lag.feature_time_lag_max_seconds
                ),
                consumer_processing_lag_max_seconds=(
                    self.reliability_config.lag.consumer_processing_lag_max_seconds
                ),
            )
            for symbol in self.settings.kraken.symbols
        ]

    async def _sync_feature_lag_state(
        self,
        evaluated_at: datetime,
    ) -> list[FeatureLagSnapshot]:
        lag_snapshots = self._build_feature_lag_snapshots(evaluated_at)
        for lag_snapshot in lag_snapshots:
            await self.reliability_store.save_feature_lag_state(lag_snapshot)
        await self._record_lag_transition_events(lag_snapshots)
        self._write_lag_summary_artifact(
            evaluated_at=evaluated_at,
            lag_snapshots=lag_snapshots,
        )
        return lag_snapshots

    async def _record_lag_transition_events(
        self,
        lag_snapshots: Sequence[FeatureLagSnapshot],
    ) -> None:
        for lag_snapshot in lag_snapshots:
            previously_in_breach = self._lag_breach_by_symbol.get(lag_snapshot.symbol)
            self._lag_breach_by_symbol[lag_snapshot.symbol] = lag_snapshot.lag_breach
            symbol_observed = (
                lag_snapshot.latest_raw_event_received_at is not None
                or lag_snapshot.latest_feature_as_of_time is not None
            )
            if not symbol_observed:
                continue
            if previously_in_breach is None and not lag_snapshot.lag_breach:
                continue
            if previously_in_breach == lag_snapshot.lag_breach:
                continue
            await self._record_reliability_event(
                RecoveryEvent(
                    service_name=self.settings.features.service_name,
                    component_name=lag_snapshot.symbol,
                    event_type="FEATURE_LAG_TRANSITION",
                    event_time=lag_snapshot.evaluated_at,
                    reason_code=(
                        FEATURE_LAG_BREACH_DETECTED
                        if lag_snapshot.lag_breach
                        else FEATURE_LAG_BREACH_CLEARED
                    ),
                    health_overall_status=lag_snapshot.health_overall_status,
                    freshness_status="STALE" if lag_snapshot.lag_breach else "FRESH",
                    breaker_state=None,
                    detail=lag_snapshot.detail,
                )
            )

    async def _record_reliability_event(self, event: RecoveryEvent) -> None:
        await self.reliability_store.insert_reliability_event(event)
        append_jsonl_artifact(
            self.reliability_config.artifacts.recovery_events_path,
            {
                "service_name": event.service_name,
                "component_name": event.component_name,
                "event_type": event.event_type,
                "event_time": to_rfc3339(event.event_time),
                "reason_code": event.reason_code,
                "health_overall_status": event.health_overall_status,
                "freshness_status": event.freshness_status,
                "breaker_state": event.breaker_state,
                "detail": event.detail,
            },
        )

    def _write_lag_summary_artifact(
        self,
        *,
        evaluated_at: datetime,
        lag_snapshots: Sequence[FeatureLagSnapshot],
    ) -> None:
        active_reason_codes = _unique_reason_codes(
            [
                lag_snapshot.reason_code
                for lag_snapshot in lag_snapshots
                if lag_snapshot.lag_breach
            ]
        )
        write_json_artifact(
            self.reliability_config.artifacts.lag_summary_path,
            {
                "generated_at": to_rfc3339(evaluated_at),
                "service_name": self.settings.features.service_name,
                "component_name": self._lag_component_name,
                "lag_breach_active": any(
                    lag_snapshot.lag_breach for lag_snapshot in lag_snapshots
                ),
                "reason_codes": (
                    active_reason_codes if active_reason_codes else [FEATURE_LAG_OK]
                ),
                "thresholds": {
                    "feature_time_lag_max_seconds": (
                        self.reliability_config.lag.feature_time_lag_max_seconds
                    ),
                    "consumer_processing_lag_max_seconds": (
                        self.reliability_config.lag.consumer_processing_lag_max_seconds
                    ),
                },
                "symbols": [
                    {
                        "symbol": lag_snapshot.symbol,
                        "evaluated_at": to_rfc3339(lag_snapshot.evaluated_at),
                        "latest_raw_event_received_at": (
                            None
                            if lag_snapshot.latest_raw_event_received_at is None
                            else to_rfc3339(lag_snapshot.latest_raw_event_received_at)
                        ),
                        "latest_feature_interval_begin": (
                            None
                            if lag_snapshot.latest_feature_interval_begin is None
                            else to_rfc3339(lag_snapshot.latest_feature_interval_begin)
                        ),
                        "latest_feature_as_of_time": (
                            None
                            if lag_snapshot.latest_feature_as_of_time is None
                            else to_rfc3339(lag_snapshot.latest_feature_as_of_time)
                        ),
                        "time_lag_seconds": lag_snapshot.time_lag_seconds,
                        "processing_lag_seconds": lag_snapshot.processing_lag_seconds,
                        "time_lag_reason_code": lag_snapshot.time_lag_reason_code,
                        "processing_lag_reason_code": (
                            lag_snapshot.processing_lag_reason_code
                        ),
                        "lag_breach": lag_snapshot.lag_breach,
                        "health_overall_status": lag_snapshot.health_overall_status,
                        "reason_code": lag_snapshot.reason_code,
                        "detail": lag_snapshot.detail,
                    }
                    for lag_snapshot in lag_snapshots
                ],
            },
        )


def _unique_reason_codes(reason_codes: Sequence[str]) -> list[str]:
    unique_codes: list[str] = []
    for reason_code in reason_codes:
        if reason_code not in unique_codes:
            unique_codes.append(reason_code)
    return unique_codes
