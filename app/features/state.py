"""Symbol-local candle finalization state for M2 feature generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.common.models import OhlcEvent
from app.features.engine import MIN_FINALIZED_CANDLES, compute_feature_row
from app.features.models import FeatureOhlcRow


@dataclass(frozen=True, slots=True)
class CandleStateKey:
    """Stable key for one exchange-symbol-interval OHLC stream."""

    source_exchange: str
    symbol: str
    interval_minutes: int


@dataclass(slots=True)
class SymbolFeatureState:
    """Mutable finalized and current candle state for one OHLC stream."""

    finalized_candles: list[OhlcEvent] = field(default_factory=list)
    current_candle: OhlcEvent | None = None


class FeatureStateManager:
    """Finalize candles safely and emit feature rows only for closed intervals."""

    def __init__(self, grace_seconds: int, history_limit: int) -> None:
        self._grace_period = timedelta(seconds=grace_seconds)
        self._history_limit = max(history_limit, MIN_FINALIZED_CANDLES)
        self._states: dict[CandleStateKey, SymbolFeatureState] = {}

    def apply_event(
        self,
        event: OhlcEvent,
        *,
        computed_at: datetime,
    ) -> list[FeatureOhlcRow]:
        """Apply one raw OHLC update and return any newly finalized feature rows."""
        key = self._key_for(event)
        state = self._states.setdefault(key, SymbolFeatureState())
        current = state.current_candle

        if current is None:
            if self._is_older_or_duplicate(event, state):
                return []
            state.current_candle = event
            return []

        if event.interval_begin == current.interval_begin:
            state.current_candle = event
            return []

        if event.interval_begin > current.interval_begin:
            finalized_row = self._finalize_current(state, computed_at)
            state.current_candle = event
            return [] if finalized_row is None else [finalized_row]

        return []

    def sweep(self, *, now: datetime, computed_at: datetime) -> list[FeatureOhlcRow]:
        """Finalize any stale current candles once their grace period has elapsed."""
        emitted_rows: list[FeatureOhlcRow] = []
        for state in self._states.values():
            current = state.current_candle
            if current is None:
                continue
            if now >= current.interval_end + self._grace_period:
                finalized_row = self._finalize_current(state, computed_at)
                if finalized_row is not None:
                    emitted_rows.append(finalized_row)
        return emitted_rows

    def bootstrap(
        self,
        events: Iterable[OhlcEvent],
        *,
        now: datetime,
        computed_at: datetime,
    ) -> list[FeatureOhlcRow]:
        """Rebuild in-memory state from persisted raw OHLC rows and backfill features."""
        grouped_events: dict[CandleStateKey, list[OhlcEvent]] = defaultdict(list)
        self._states.clear()

        for event in events:
            grouped_events[self._key_for(event)].append(event)

        emitted_rows: list[FeatureOhlcRow] = []
        for key in sorted(grouped_events, key=self._sort_key):
            ordered_events = sorted(
                grouped_events[key],
                key=lambda candle: candle.interval_begin,
            )
            state = self._states.setdefault(key, SymbolFeatureState())
            if not ordered_events:
                continue

            finalized_events = ordered_events
            latest_event = ordered_events[-1]
            if now < latest_event.interval_end + self._grace_period:
                finalized_events = ordered_events[:-1]
                state.current_candle = latest_event

            for candle in finalized_events:
                feature_row = self._append_finalized_candle(state, candle, computed_at)
                if feature_row is not None:
                    emitted_rows.append(feature_row)

        return emitted_rows

    def get_state(
        self,
        source_exchange: str,
        symbol: str,
        interval_minutes: int,
    ) -> SymbolFeatureState | None:
        """Return the current state for tests and diagnostics."""
        return self._states.get(CandleStateKey(source_exchange, symbol, interval_minutes))

    def _finalize_current(
        self,
        state: SymbolFeatureState,
        computed_at: datetime,
    ) -> FeatureOhlcRow | None:
        current = state.current_candle
        if current is None:
            return None

        state.current_candle = None
        return self._append_finalized_candle(state, current, computed_at)

    def _append_finalized_candle(
        self,
        state: SymbolFeatureState,
        candle: OhlcEvent,
        computed_at: datetime,
    ) -> FeatureOhlcRow | None:
        state.finalized_candles.append(candle)
        if len(state.finalized_candles) > self._history_limit:
            state.finalized_candles = state.finalized_candles[-self._history_limit :]
        return compute_feature_row(state.finalized_candles, computed_at=computed_at)

    @staticmethod
    def _key_for(event: OhlcEvent) -> CandleStateKey:
        return CandleStateKey(
            source_exchange=event.source_exchange,
            symbol=event.symbol,
            interval_minutes=event.interval_minutes,
        )

    @staticmethod
    def _sort_key(key: CandleStateKey) -> tuple[str, str, int]:
        return (key.source_exchange, key.symbol, key.interval_minutes)

    @staticmethod
    def _is_older_or_duplicate(event: OhlcEvent, state: SymbolFeatureState) -> bool:
        if not state.finalized_candles:
            return False
        return event.interval_begin <= state.finalized_candles[-1].interval_begin
