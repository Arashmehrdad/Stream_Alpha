"""State manager tests for finalized OHLC feature emission behavior."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.common.models import OhlcEvent
from app.features.state import FeatureStateManager


def _build_candle(index: int, close_offset: float = 0.0) -> OhlcEvent:
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    interval_begin = base_time + timedelta(minutes=5 * index)
    close_price = 100.0 + (index * 1.5) + close_offset
    volume = 25.0 + (index * 0.8)
    return OhlcEvent(
        event_id=f"evt-{index}-{close_offset}",
        app_name="streamalpha",
        source_exchange="kraken",
        channel="ohlc",
        message_type="update",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        open_price=close_price - 0.6,
        high_price=close_price + 0.9,
        low_price=close_price - 1.0,
        close_price=close_price,
        vwap=close_price - 0.1,
        trade_count=20 + index,
        volume=volume,
        received_at=interval_begin + timedelta(minutes=5, seconds=1),
    )


def test_repeated_updates_to_same_candle_do_not_emit_duplicate_finalized_rows() -> None:
    """Same-interval updates should replace the open candle without double-finalizing it."""
    manager = FeatureStateManager(grace_seconds=30, history_limit=64)
    bootstrap_events = [_build_candle(index) for index in range(26)]
    bootstrap_now = bootstrap_events[-1].interval_end + timedelta(seconds=10)

    bootstrap_rows = manager.bootstrap(
        bootstrap_events,
        now=bootstrap_now,
        computed_at=bootstrap_now,
    )
    update_one = _build_candle(25, close_offset=0.3)
    update_two = _build_candle(25, close_offset=0.7)
    next_interval = _build_candle(26)

    rows_first_update = manager.apply_event(
        update_one,
        computed_at=update_one.interval_end + timedelta(seconds=5),
    )
    rows_second_update = manager.apply_event(
        update_two,
        computed_at=update_two.interval_end + timedelta(seconds=6),
    )
    finalized_rows = manager.apply_event(
        next_interval,
        computed_at=next_interval.interval_begin + timedelta(seconds=1),
    )
    rows_after_finalization = manager.apply_event(
        _build_candle(26, close_offset=0.5),
        computed_at=next_interval.interval_begin + timedelta(seconds=2),
    )

    assert not bootstrap_rows
    assert not rows_first_update
    assert not rows_second_update
    assert len(finalized_rows) == 1
    assert finalized_rows[0].interval_begin == update_two.interval_begin
    assert finalized_rows[0].close_price == update_two.close_price
    assert not rows_after_finalization


def test_next_interval_arrival_finalizes_prior_candle() -> None:
    """A later interval should safely finalize the prior candle and emit its features."""
    manager = FeatureStateManager(grace_seconds=30, history_limit=64)
    for index in range(26):
        manager.apply_event(
            _build_candle(index),
            computed_at=_build_candle(index).interval_end,
        )

    next_interval = _build_candle(26)
    feature_rows = manager.apply_event(
        next_interval,
        computed_at=next_interval.interval_begin + timedelta(seconds=1),
    )

    assert len(feature_rows) == 1
    assert feature_rows[0].interval_begin == _build_candle(25).interval_begin


def test_grace_based_sweep_finalizes_a_stale_current_candle() -> None:
    """The periodic sweep should finalize a stale open candle once grace has passed."""
    manager = FeatureStateManager(grace_seconds=30, history_limit=64)
    for index in range(26):
        candle = _build_candle(index)
        manager.apply_event(candle, computed_at=candle.interval_end)

    stale_rows = manager.sweep(
        now=_build_candle(25).interval_end + timedelta(seconds=31),
        computed_at=_build_candle(25).interval_end + timedelta(seconds=31),
    )

    assert len(stale_rows) == 1
    assert stale_rows[0].interval_begin == _build_candle(25).interval_begin


def test_bootstrap_state_rebuild_backfills_features_and_restores_current_candle() -> None:
    """Bootstrap should rebuild finalized history and preserve the latest open candle."""
    manager = FeatureStateManager(grace_seconds=30, history_limit=64)
    bootstrap_events = [_build_candle(index) for index in range(31)]
    bootstrap_now = bootstrap_events[-1].interval_end + timedelta(seconds=10)

    bootstrap_rows = manager.bootstrap(
        bootstrap_events,
        now=bootstrap_now,
        computed_at=bootstrap_now,
    )
    state = manager.get_state("kraken", "BTC/USD", 5)

    assert state is not None
    assert len(bootstrap_rows) == 5
    assert len(state.finalized_candles) == 30
    assert state.current_candle is not None
    assert state.current_candle.interval_begin == bootstrap_events[-1].interval_begin

    next_interval = _build_candle(31)
    finalized_rows = manager.apply_event(
        next_interval,
        computed_at=next_interval.interval_begin + timedelta(seconds=1),
    )

    assert len(finalized_rows) == 1
    assert finalized_rows[0].interval_begin == bootstrap_events[-1].interval_begin
