"""Unit tests for payload normalization."""

from datetime import timezone

import pytest

from app.common.time import parse_rfc3339
from app.ingestion.normalizers import (
    NormalizationError,
    normalize_ohlc_payload,
    normalize_trade_payload,
)


def test_normalize_trade_payload_maps_expected_fields() -> None:
    """Trade payloads should map into the expected internal event fields."""
    event = normalize_trade_payload(
        {
            "symbol": "BTC/USD",
            "side": "buy",
            "price": 67000.1,
            "qty": 0.125,
            "ord_type": "market",
            "trade_id": 42,
            "timestamp": "2025-01-02T03:04:05.123456789Z",
        },
        app_name="streamalpha",
        message_type="update",
        received_at=parse_rfc3339("2025-01-02T03:04:06.000000Z"),
    )

    assert event.source_exchange == "kraken"
    assert event.channel == "trade"
    assert event.trade_id == 42
    assert event.symbol == "BTC/USD"
    assert event.price == 67000.1
    assert event.quantity == 0.125
    assert event.event_time.microsecond == 123456
    assert event.event_time.tzinfo == timezone.utc


def test_normalize_ohlc_payload_computes_interval_end() -> None:
    """OHLC payloads should compute an interval end from begin plus interval."""
    event = normalize_ohlc_payload(
        {
            "symbol": "ETH/USD",
            "open": 3500.0,
            "high": 3510.0,
            "low": 3495.0,
            "close": 3502.5,
            "vwap": 3501.2,
            "trades": 18,
            "volume": 12.75,
            "interval_begin": "2025-01-02T03:00:00.111111111Z",
            "interval": 5,
        },
        app_name="streamalpha",
        message_type="snapshot",
        received_at=parse_rfc3339("2025-01-02T03:05:00.000000Z"),
    )

    assert event.channel == "ohlc"
    assert event.symbol == "ETH/USD"
    assert event.interval_minutes == 5
    assert event.interval_begin.isoformat() == "2025-01-02T03:00:00.111111+00:00"
    assert event.interval_end.isoformat() == "2025-01-02T03:05:00.111111+00:00"
    assert event.trade_count == 18


def test_trade_normalizer_rejects_missing_required_field() -> None:
    """Normalization should fail fast when required trade fields are missing."""
    with pytest.raises(NormalizationError):
        normalize_trade_payload(
            {
                "symbol": "SOL/USD",
                "side": "sell",
                "price": 140.5,
                "qty": 10,
                "ord_type": "limit",
                "timestamp": "2025-01-02T03:04:05Z",
            },
            app_name="streamalpha",
            message_type="update",
            received_at=parse_rfc3339("2025-01-02T03:04:06Z"),
        )
