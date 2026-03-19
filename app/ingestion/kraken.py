"""Kraken WebSocket v2 subscription helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SubscriptionRequest:
    """A single Kraken public channel subscription request."""

    channel: str
    symbols: tuple[str, ...]
    interval: int | None = None
    snapshot: bool = True

    def as_message(self, req_id: int) -> dict[str, Any]:
        """Render the request as a Kraken WebSocket payload."""
        params: dict[str, Any] = {
            "channel": self.channel,
            "symbol": list(self.symbols),
            "snapshot": self.snapshot,
        }
        if self.interval is not None:
            params["interval"] = self.interval
        return {
            "method": "subscribe",
            "params": params,
            "req_id": req_id,
        }


def build_public_subscription_requests(
    symbols: tuple[str, ...],
    ohlc_interval_minutes: int,
) -> list[SubscriptionRequest]:
    """Build the fixed M1 public subscriptions for Kraken."""
    return [
        SubscriptionRequest(channel="trade", symbols=symbols, snapshot=True),
        SubscriptionRequest(
            channel="ohlc",
            symbols=symbols,
            interval=ohlc_interval_minutes,
            snapshot=True,
        ),
    ]


def expected_subscription_keys(
    symbols: tuple[str, ...],
    ohlc_interval_minutes: int,
) -> set[str]:
    """Return the acknowledgement keys expected for a full subscription set."""
    keys = {f"trade:{symbol}" for symbol in symbols}
    keys.update(f"ohlc:{symbol}:{ohlc_interval_minutes}" for symbol in symbols)
    return keys


def ack_key(message: dict[str, Any]) -> str:
    """Convert a Kraken subscribe acknowledgement into a stable lookup key."""
    result = message.get("result", {})
    channel = result.get("channel")
    symbol = result.get("symbol")
    interval = result.get("interval")
    if channel == "ohlc":
        return f"{channel}:{symbol}:{interval}"
    return f"{channel}:{symbol}"
