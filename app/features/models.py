"""Typed models and deserialization helpers for finalized OHLC features."""

# pylint: disable=duplicate-code

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from app.common.models import OhlcEvent
from app.common.time import parse_rfc3339


def _require_mapping_value(payload: Mapping[str, Any], field_name: str) -> Any:
    value = payload.get(field_name)
    if value is None:
        raise ValueError(f"Missing required OHLC field: {field_name}")
    return value


def _require_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = _require_mapping_value(payload, field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field {field_name} must be a non-empty string")
    return value.strip()


def _require_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = _require_mapping_value(payload, field_name)
    return int(value)


def _require_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = _require_mapping_value(payload, field_name)
    return float(value)


def _require_datetime(payload: Mapping[str, Any], field_name: str) -> datetime:
    return parse_rfc3339(_require_string(payload, field_name))


def ohlc_event_from_mapping(payload: Mapping[str, Any]) -> OhlcEvent:
    """Build an `OhlcEvent` from a JSON-style mapping."""
    return OhlcEvent(
        event_id=_require_string(payload, "event_id"),
        app_name=_require_string(payload, "app_name"),
        source_exchange=_require_string(payload, "source_exchange"),
        channel=_require_string(payload, "channel"),
        message_type=_require_string(payload, "message_type"),
        symbol=_require_string(payload, "symbol"),
        interval_minutes=_require_int(payload, "interval_minutes"),
        interval_begin=_require_datetime(payload, "interval_begin"),
        interval_end=_require_datetime(payload, "interval_end"),
        open_price=_require_float(payload, "open_price"),
        high_price=_require_float(payload, "high_price"),
        low_price=_require_float(payload, "low_price"),
        close_price=_require_float(payload, "close_price"),
        vwap=_require_float(payload, "vwap"),
        trade_count=_require_int(payload, "trade_count"),
        volume=_require_float(payload, "volume"),
        received_at=_require_datetime(payload, "received_at"),
    )


def deserialize_ohlc_event(payload: bytes | str | dict[str, Any]) -> OhlcEvent:
    """Deserialize a raw Kafka or PostgreSQL payload into an `OhlcEvent`."""
    parsed_payload: Any
    if isinstance(payload, bytes):
        parsed_payload = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        parsed_payload = json.loads(payload)
    else:
        parsed_payload = payload

    if not isinstance(parsed_payload, dict):
        raise ValueError("OHLC payload must deserialize into an object")
    return ohlc_event_from_mapping(parsed_payload)


@dataclass(frozen=True, slots=True)
class FeatureOhlcRow:  # pylint: disable=too-many-instance-attributes
    """Typed feature row emitted only for finalized OHLC candles."""

    source_exchange: str
    symbol: str
    interval_minutes: int
    interval_begin: datetime
    interval_end: datetime
    as_of_time: datetime
    computed_at: datetime
    raw_event_id: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    vwap: float
    trade_count: int
    volume: float
    log_return_1: float
    log_return_3: float
    momentum_3: float
    return_mean_12: float
    return_std_12: float
    realized_vol_12: float
    rsi_14: float
    macd_line_12_26: float
    volume_mean_12: float
    volume_std_12: float
    volume_zscore_12: float
    close_zscore_12: float
    lag_log_return_1: float
    lag_log_return_2: float
    lag_log_return_3: float
