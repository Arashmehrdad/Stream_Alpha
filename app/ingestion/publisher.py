"""Kafka publishing utilities for normalized events."""

from __future__ import annotations

from typing import Any

from aiokafka import AIOKafkaProducer

from app.common.serialization import serialize_model


class KafkaEventPublisher:
    """Thin wrapper around an async Kafka producer client."""

    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str,
        producer_client: Any | None = None,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._client_id = client_id
        self._producer = producer_client
        self._owns_client = producer_client is None
        self._started = False

    async def start(self) -> None:
        """Start the underlying producer client."""
        if self._started:
            return
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                client_id=self._client_id,
            )
        await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the underlying producer client."""
        if not self._started:
            return
        await self._producer.stop()
        self._started = False
        if self._owns_client:
            self._producer = None

    async def publish(self, topic: str, key: str | None, event: Any) -> None:
        """Publish a normalized event to a target topic."""
        if not self._started or self._producer is None:
            raise RuntimeError("KafkaEventPublisher has not been started")
        payload = serialize_model(event)
        encoded_key = key.encode("utf-8") if key is not None else None
        await self._producer.send_and_wait(topic, payload, key=encoded_key)
