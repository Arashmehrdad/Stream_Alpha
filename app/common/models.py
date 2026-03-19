"""Typed internal event models shared across services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


def generate_event_id() -> str:
    """Create a unique event identifier for normalized events."""
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class TradeEvent:  # pylint: disable=too-many-instance-attributes
    """Normalized trade event published and persisted by the producer."""

    event_id: str
    app_name: str
    source_exchange: str
    channel: str
    message_type: str
    symbol: str
    trade_id: int
    side: str
    order_type: str
    price: float
    quantity: float
    event_time: datetime
    received_at: datetime


@dataclass(frozen=True, slots=True)
class OhlcEvent:  # pylint: disable=too-many-instance-attributes
    """Normalized OHLC event published and persisted by the producer."""

    event_id: str
    app_name: str
    source_exchange: str
    channel: str
    message_type: str
    symbol: str
    interval_minutes: int
    interval_begin: datetime
    interval_end: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    vwap: float
    trade_count: int
    volume: float
    received_at: datetime


@dataclass(frozen=True, slots=True)
class HealthEvent:  # pylint: disable=too-many-instance-attributes
    """Health and observability event for producer state transitions."""

    event_id: str
    app_name: str
    service_name: str
    status: str
    component: str
    message: str
    observed_at: datetime
    source_exchange: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
