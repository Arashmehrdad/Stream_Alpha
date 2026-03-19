"""Normalization functions for Kraken trade and OHLC payloads."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.common.models import OhlcEvent, TradeEvent, generate_event_id
from app.common.time import parse_rfc3339


class NormalizationError(ValueError):
    """Raised when an inbound payload cannot be normalized."""


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise NormalizationError(f"Missing required field: {key}")
    return payload[key]


def normalize_trade_payload(
    payload: dict[str, Any],
    *,
    app_name: str,
    message_type: str,
    received_at,
) -> TradeEvent:
    """Normalize a Kraken trade payload into the internal trade model."""
    try:
        trade_id = int(_require(payload, "trade_id"))
        symbol = str(_require(payload, "symbol"))
        side = str(_require(payload, "side"))
        order_type = str(_require(payload, "ord_type"))
        price = float(_require(payload, "price"))
        quantity = float(_require(payload, "qty"))
        event_time = parse_rfc3339(str(_require(payload, "timestamp")))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"Invalid trade payload: {exc}") from exc

    return TradeEvent(
        event_id=generate_event_id(),
        app_name=app_name,
        source_exchange="kraken",
        channel="trade",
        message_type=message_type,
        symbol=symbol,
        trade_id=trade_id,
        side=side,
        order_type=order_type,
        price=price,
        quantity=quantity,
        event_time=event_time,
        received_at=received_at,
    )


def normalize_ohlc_payload(  # pylint: disable=too-many-locals
    payload: dict[str, Any],
    *,
    app_name: str,
    message_type: str,
    received_at,
) -> OhlcEvent:
    """Normalize a Kraken OHLC payload into the internal candle model."""
    try:
        symbol = str(_require(payload, "symbol"))
        interval_minutes = int(_require(payload, "interval"))
        interval_begin = parse_rfc3339(str(_require(payload, "interval_begin")))
        open_price = float(_require(payload, "open"))
        high_price = float(_require(payload, "high"))
        low_price = float(_require(payload, "low"))
        close_price = float(_require(payload, "close"))
        vwap = float(_require(payload, "vwap"))
        trade_count = int(_require(payload, "trades"))
        volume = float(_require(payload, "volume"))
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"Invalid OHLC payload: {exc}") from exc

    interval_end = interval_begin + timedelta(minutes=interval_minutes)
    return OhlcEvent(
        event_id=generate_event_id(),
        app_name=app_name,
        source_exchange="kraken",
        channel="ohlc",
        message_type=message_type,
        symbol=symbol,
        interval_minutes=interval_minutes,
        interval_begin=interval_begin,
        interval_end=interval_end,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        vwap=vwap,
        trade_count=trade_count,
        volume=volume,
        received_at=received_at,
    )
