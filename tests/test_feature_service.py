"""Service-level tests for M2 Kafka consumer safety behavior."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.common.config import Settings
from app.features.service import FeatureConsumerService
from app.reliability.service import (
    FEATURE_LAG_BREACH,
    FEATURE_LAG_BREACH_CLEARED,
    FEATURE_LAG_BREACH_DETECTED,
)


class FakeKafkaConsumer:
    """Minimal Kafka consumer stub used for config and commit-path tests."""

    last_instance = None

    def __init__(self, *topics: str, **kwargs: object) -> None:
        self.topics = topics
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.commit_calls = 0
        FakeKafkaConsumer.last_instance = self

    async def start(self) -> None:
        """Record that the fake consumer started."""
        self.started = True

    async def stop(self) -> None:
        """Record that the fake consumer stopped."""
        self.stopped = True

    async def commit(self) -> None:
        """Record one explicit offset commit."""
        self.commit_calls += 1


class FakeReliabilityStore:
    """Minimal reliability store stub for feature lag tests."""

    def __init__(self) -> None:
        self.saved_lag_states = []
        self.inserted_events = []

    async def save_feature_lag_state(self, lag_state) -> None:
        self.saved_lag_states.append(lag_state)

    async def insert_reliability_event(self, event) -> None:
        self.inserted_events.append(event)


def test_start_consumer_uses_earliest_and_manual_commits(monkeypatch) -> None:
    """The feature consumer should use restart-safe Kafka offset settings."""
    monkeypatch.setattr("app.features.service.AIOKafkaConsumer", FakeKafkaConsumer)
    service = FeatureConsumerService(Settings.from_env())

    async def run_test() -> None:
        await service._start_consumer()
        await service._reset_dependencies()

    asyncio.run(run_test())

    consumer = FakeKafkaConsumer.last_instance
    assert consumer is not None
    assert consumer.topics == ("raw.ohlc",)
    assert consumer.kwargs["auto_offset_reset"] == "earliest"
    assert consumer.kwargs["enable_auto_commit"] is False


def test_process_batch_commits_offsets_after_success(monkeypatch) -> None:
    """Processed batches should commit offsets only after record handling succeeds."""
    service = FeatureConsumerService(Settings.from_env())
    fake_consumer = FakeKafkaConsumer("raw.ohlc")
    handled_values: list[bytes] = []

    async def fake_handle_record(value: bytes) -> None:
        handled_values.append(value)

    service._consumer = fake_consumer
    monkeypatch.setattr(service, "_handle_record", fake_handle_record)

    async def run_test() -> None:
        await service._process_batch(
            {
                "partition-0": [
                    SimpleNamespace(value=b"first"),
                    SimpleNamespace(value=b"second"),
                ]
            }
        )

    asyncio.run(run_test())

    assert handled_values == [b"first", b"second"]
    assert fake_consumer.commit_calls == 1


def test_process_batch_does_not_commit_offsets_on_failure(monkeypatch) -> None:
    """A failing batch should not advance offsets before retry."""
    service = FeatureConsumerService(Settings.from_env())
    fake_consumer = FakeKafkaConsumer("raw.ohlc")

    async def fake_handle_record(_: bytes) -> None:
        raise RuntimeError("upsert failed")

    service._consumer = fake_consumer
    monkeypatch.setattr(service, "_handle_record", fake_handle_record)

    async def run_test() -> None:
        await service._process_batch({"partition-0": [SimpleNamespace(value=b"boom")]})

    try:
        asyncio.run(run_test())
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the batch failure to propagate")

    assert fake_consumer.commit_calls == 0


def test_feature_lag_state_persists_breach_and_clear_transitions(monkeypatch) -> None:
    """Feature lag sync should persist per-symbol lag rows and transition events."""
    service = FeatureConsumerService(Settings.from_env())
    fake_store = FakeReliabilityStore()
    service.reliability_store = fake_store
    monkeypatch.setattr("app.features.service.write_json_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "app.features.service.append_jsonl_artifact",
        lambda *_args, **_kwargs: None,
    )
    now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    service._latest_raw_event_received_at_by_symbol["BTC/USD"] = now
    service._latest_feature_interval_begin_by_symbol["BTC/USD"] = now - timedelta(minutes=10)
    service._latest_feature_as_of_time_by_symbol["BTC/USD"] = now - timedelta(minutes=15)

    async def run_test() -> None:
        await service._sync_feature_lag_state(now)
        service._latest_feature_interval_begin_by_symbol["BTC/USD"] = now - timedelta(minutes=5)
        service._latest_feature_as_of_time_by_symbol["BTC/USD"] = now - timedelta(minutes=1)
        await service._sync_feature_lag_state(now)

    asyncio.run(run_test())

    assert fake_store.saved_lag_states[0].reason_code == FEATURE_LAG_BREACH
    assert fake_store.inserted_events[0].reason_code == FEATURE_LAG_BREACH_DETECTED
    assert fake_store.inserted_events[1].reason_code == FEATURE_LAG_BREACH_CLEARED
