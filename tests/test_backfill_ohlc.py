"""Focused tests for the Kraken REST OHLC historical backfill helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.common.models import OhlcEvent
from app.features.engine import MIN_FINALIZED_CANDLES
from app.features.state import FeatureStateManager
from app.ingestion import backfill_ohlc as backfill_module
from app.ingestion.backfill_ohlc import (
    BackfillWindow,
    _closed_rest_rows,
    _event_from_rest_row,
    _normalize_backfill_feature_row,
    _regenerate_feature_rows,
    _sync_symbol_raw_window,
)


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


def test_closed_rest_rows_return_all_closed_rows_when_lookback_is_none() -> None:
    """Historical windows should preserve every closed row when no lookback trim is requested."""
    rows = [
        [1, "10", "11", "9", "10.5", "10.2", "100", 5],
        [2, "11", "12", "10", "11.5", "11.2", "110", 6],
        [3, "12", "13", "11", "12.5", "12.2", "120", 7],
    ]

    closed_rows = _closed_rest_rows(rows, lookback_candles=None)

    assert closed_rows == [
        [1, "10", "11", "9", "10.5", "10.2", "100", 5],
        [2, "11", "12", "10", "11.5", "11.2", "110", 6],
    ]


def test_event_from_rest_row_uses_deterministic_backfill_identity() -> None:
    """Historical reruns should produce the same raw-event identity for the same candle."""
    interval_begin = datetime(2026, 3, 19, 22, 0, tzinfo=timezone.utc)
    row = [
        int(interval_begin.timestamp()),
        "70449.1",
        "70452.8",
        "70385.0",
        "70442.4",
        "70386.8",
        "10.42287647",
        136,
    ]

    first = _event_from_rest_row(
        app_name="streamalpha",
        symbol="BTC/USD",
        interval_minutes=5,
        row=row,
        received_at=interval_begin + timedelta(minutes=5),
    )
    second = _event_from_rest_row(
        app_name="streamalpha",
        symbol="BTC/USD",
        interval_minutes=5,
        row=row,
        received_at=interval_begin + timedelta(minutes=5),
    )

    assert first.event_id == second.event_id
    assert first.interval_begin == interval_begin
    assert first.received_at == interval_begin + timedelta(minutes=5)


class _FakeWriter:
    def __init__(self) -> None:
        self.written_events: list[OhlcEvent] = []

    async def write_ohlc(self, event: OhlcEvent) -> None:
        self.written_events.append(event)


class _FakeFeatureStore:
    def __init__(
        self,
        *,
        raw_events: list[OhlcEvent],
        feature_rows,
    ) -> None:
        self._raw_events = raw_events
        self._feature_rows = list(feature_rows)
        self.upserted_rows = []

    async def load_raw_candles(self, **kwargs):
        start = kwargs.get("start")
        end = kwargs.get("end")
        if start is None and end is None:
            return list(self._raw_events)
        return [
            event
            for event in self._raw_events
            if (start is None or event.interval_begin >= start)
            and (end is None or event.interval_begin < end)
        ]

    async def load_feature_rows(self, **kwargs):
        start = kwargs.get("start")
        end = kwargs.get("end")
        return [
            row
            for row in self._feature_rows
            if (start is None or row.interval_begin >= start)
            and (end is None or row.interval_begin < end)
        ]

    async def upsert_feature_row(self, row) -> None:
        self.upserted_rows.append(row)


def _build_raw_events(symbol: str, *, count: int) -> list[OhlcEvent]:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    events: list[OhlcEvent] = []
    for index in range(count):
        interval_begin = start + timedelta(minutes=index * 5)
        close_price = 100.0 + index
        events.append(
            OhlcEvent(
                event_id=f"{symbol}-{index}",
                app_name="streamalpha",
                source_exchange="kraken",
                channel="ohlc",
                message_type="backfill",
                symbol=symbol,
                interval_minutes=5,
                interval_begin=interval_begin,
                interval_end=interval_begin + timedelta(minutes=5),
                open_price=close_price - 0.5,
                high_price=close_price + 0.5,
                low_price=close_price - 1.0,
                close_price=close_price,
                vwap=close_price,
                trade_count=10 + index,
                volume=1.0 + index,
                received_at=interval_begin + timedelta(minutes=5),
            )
        )
    return events


def _build_feature_rows(raw_events: list[OhlcEvent]):
    state = FeatureStateManager(grace_seconds=0, history_limit=MIN_FINALIZED_CANDLES + 16)
    rebuilt_rows = state.bootstrap(
        raw_events,
        now=raw_events[-1].interval_end + timedelta(minutes=5),
        computed_at=raw_events[-1].interval_end + timedelta(minutes=5),
    )
    return [_normalize_backfill_feature_row(row) for row in rebuilt_rows]


def test_sync_symbol_raw_window_skips_unchanged_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerunning the same raw backfill window should not rewrite identical candles."""
    interval_begin = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    row = [
        int(interval_begin.timestamp()),
        "100.0",
        "101.0",
        "99.0",
        "100.5",
        "100.4",
        "2.5",
        12,
    ]
    existing_event = _event_from_rest_row(
        app_name="streamalpha",
        symbol="BTC/USD",
        interval_minutes=5,
        row=row,
        received_at=interval_begin + timedelta(minutes=5),
    )
    store = _FakeFeatureStore(raw_events=[existing_event], feature_rows=[])
    writer = _FakeWriter()
    monkeypatch.setattr(
        backfill_module,
        "_fetch_window_rows_with_retries",
        lambda *args, **kwargs: ([row], False),
    )

    stats = asyncio.run(
        _sync_symbol_raw_window(
            writer=writer,
            store=store,
            rest_ohlc_url="https://example.invalid",
            app_name="streamalpha",
            symbol="BTC/USD",
            interval_minutes=5,
            window=BackfillWindow(
                start=interval_begin,
                end=interval_begin + timedelta(minutes=5),
            ),
            request_retries=1,
        )
    )

    assert stats.created_rows == 0
    assert stats.updated_rows == 0
    assert stats.unchanged_rows == 1
    assert writer.written_events == []


def test_regenerate_feature_rows_are_idempotent_on_rerun() -> None:
    """Feature replay should skip unchanged rows when the same raw history is replayed again."""
    raw_events = _build_raw_events("BTC/USD", count=MIN_FINALIZED_CANDLES + 6)
    rebuilt_rows = _build_feature_rows(raw_events)
    window = BackfillWindow(
        start=raw_events[0].interval_begin,
        end=raw_events[-1].interval_end + timedelta(minutes=5),
    )

    first_store = _FakeFeatureStore(raw_events=raw_events, feature_rows=[])
    first_stats = asyncio.run(
        _regenerate_feature_rows(
            store=first_store,
            symbols=("BTC/USD",),
            interval_minutes=5,
            history_limit=MIN_FINALIZED_CANDLES + 8,
            window=window,
        )
    )

    second_store = _FakeFeatureStore(
        raw_events=raw_events,
        feature_rows=first_store.upserted_rows,
    )
    second_stats = asyncio.run(
        _regenerate_feature_rows(
            store=second_store,
            symbols=("BTC/USD",),
            interval_minutes=5,
            history_limit=MIN_FINALIZED_CANDLES + 8,
            window=window,
        )
    )

    assert first_stats.created_rows == len(first_store.upserted_rows)
    assert first_stats.updated_rows == 0
    assert first_stats.generated_rows > 0
    assert second_stats.created_rows == 0
    assert second_stats.updated_rows == 0
    assert second_stats.unchanged_rows == first_stats.generated_rows
    assert second_store.upserted_rows == []
