"""Smoke test for Kafka publish serialization behavior."""

# pylint: disable=duplicate-code

import asyncio
import json

from app.common.models import TradeEvent, generate_event_id
from app.common.time import parse_rfc3339
from app.ingestion.publisher import KafkaEventPublisher


class FakeProducerClient:
    """Minimal async producer stub used by the smoke test."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.messages: list[dict[str, bytes | None | str]] = []

    async def start(self) -> None:
        """Record that the fake producer has started."""
        self.started = True

    async def stop(self) -> None:
        """Record that the fake producer has stopped."""
        self.stopped = True

    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        key: bytes | None = None,
    ) -> None:
        """Capture one published message payload."""
        self.messages.append({"topic": topic, "value": value, "key": key})


def test_publish_smoke_path_serializes_expected_event() -> None:
    """Publishing should emit the expected topic, key, and JSON payload."""
    fake_producer = FakeProducerClient()
    publisher = KafkaEventPublisher(
        bootstrap_servers="unused",
        client_id="test-client",
        producer_client=fake_producer,
    )

    event = TradeEvent(
        event_id=generate_event_id(),
        app_name="streamalpha",
        source_exchange="kraken",
        channel="trade",
        message_type="update",
        symbol="BTC/USD",
        trade_id=99,
        side="buy",
        order_type="market",
        price=66000.0,
        quantity=0.25,
        event_time=parse_rfc3339("2025-01-02T03:04:05.123456Z"),
        received_at=parse_rfc3339("2025-01-02T03:04:06.000000Z"),
    )

    async def run_test() -> None:
        """Drive the publisher start, publish, and stop lifecycle."""
        await publisher.start()
        await publisher.publish("raw.trades", "BTC/USD:99", event)
        await publisher.stop()

    asyncio.run(run_test())

    assert fake_producer.started is True
    assert fake_producer.stopped is True
    assert len(fake_producer.messages) == 1

    published = fake_producer.messages[0]
    assert published["topic"] == "raw.trades"
    assert published["key"] == b"BTC/USD:99"

    payload = json.loads(published["value"].decode("utf-8"))
    assert payload["symbol"] == "BTC/USD"
    assert payload["trade_id"] == 99
    assert payload["channel"] == "trade"
