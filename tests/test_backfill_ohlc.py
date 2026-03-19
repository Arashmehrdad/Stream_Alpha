"""Focused tests for the Kraken REST OHLC backfill helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.ingestion.backfill_ohlc import _closed_rest_rows, _event_from_rest_row


def test_closed_rest_rows_drop_final_open_candle_and_trim_lookback() -> None:
    """The final REST row is excluded before the lookback window is applied."""
    rows = [
        [1, "10", "11", "9", "10.5", "10.2", "100", 5],
        [2, "11", "12", "10", "11.5", "11.2", "110", 6],
        [3, "12", "13", "11", "12.5", "12.2", "120", 7],
        [4, "13", "14", "12", "13.5", "13.2", "130", 8],
    ]

    closed_rows = _closed_rest_rows(rows, lookback_candles=2)

    assert closed_rows == [
        [2, "11", "12", "10", "11.5", "11.2", "110", 6],
        [3, "12", "13", "11", "12.5", "12.2", "120", 7],
    ]


def test_event_from_rest_row_matches_internal_ohlc_shape() -> None:
    """REST OHLC rows should map cleanly into the shared `OhlcEvent` model."""
    received_at = datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc)
    row = [
        1_763_958_500,
        "70449.1",
        "70452.8",
        "70385.0",
        "70442.4",
        "70386.8",
        "10.42287647",
        136,
    ]

    event = _event_from_rest_row(
        app_name="streamalpha",
        symbol="BTC/USD",
        interval_minutes=5,
        row=row,
        received_at=received_at,
    )

    assert event.message_type == "backfill"
    assert event.symbol == "BTC/USD"
    assert event.interval_begin == datetime.fromtimestamp(row[0], tz=timezone.utc)
    assert event.interval_end == datetime.fromtimestamp(row[0], tz=timezone.utc) + timedelta(
        minutes=5,
    )
    assert event.close_price == 70442.4
    assert event.volume == 10.42287647
    assert event.trade_count == 136
