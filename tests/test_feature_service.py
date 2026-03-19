"""Service-level tests for M2 Kafka consumer safety behavior."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.common.config import Settings
from app.features.service import FeatureConsumerService


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
