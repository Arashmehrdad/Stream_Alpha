"""Runner idempotency tests for Stream Alpha M5."""

# pylint: disable=duplicate-code,line-too-long,missing-class-docstring
# pylint: disable=missing-function-docstring,too-few-public-methods
# pylint: disable=too-many-public-methods,too-many-arguments

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.adaptation.config import default_adaptation_config_path, load_adaptation_config
from app.alerting.config import (
    AlertingArtifactConfig,
    AlertingConfig,
    OrderFailureSpikeConfig,
    SignalAlertConfig,
)
from app.alerting.service import OperationalAlertService
from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat
from app.trading.config import PaperTradingConfig, RiskConfig
from app.trading.runner import PaperTradingRunner
from app.trading.schemas import (
    CanonicalFeatureLag,
    CanonicalServiceHealth,
    CanonicalSystemReliability,
    DecisionTraceRecord,
    FeatureCandle,
    OrderLifecycleEvent,
    OrderRequest,
    PaperEngineState,
    PendingSignalState,
    ServiceRiskState,
    SignalDecision,
)


def _config(tmp_path: Path) -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir=str(tmp_path / "artifacts"),
        risk=RiskConfig(
            initial_cash=10_000.0,
            position_fraction=0.25,
            fee_bps=20.0,
            slippage_bps=5.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cooldown_candles=1,
            max_open_positions=1,
            max_exposure_per_asset=0.25,
        ),
    )


def _candle(index: int) -> FeatureCandle:
    interval_begin = datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
    return FeatureCandle(
        id=index + 1,
        source_exchange="kraken",
        symbol="BTC/USD",
        interval_minutes=5,
        interval_begin=interval_begin,
        interval_end=interval_begin + timedelta(minutes=5),
        as_of_time=interval_begin + timedelta(minutes=5),
        raw_event_id=f"evt-{index}",
        open_price=100.0 + index,
        high_price=101.0 + index,
        low_price=99.0 + index,
        close_price=100.5 + index,
    )


def _system_reliability(evaluated_at: datetime) -> CanonicalSystemReliability:
    return CanonicalSystemReliability(
        service_name="stream-alpha",
        checked_at=evaluated_at,
        health_overall_status="HEALTHY",
        reason_codes=("FEED_FRESH",),
        lag_breach_active=False,
        services=(
            CanonicalServiceHealth(
                service_name="stream-alpha",
                component_name="producer",
                checked_at=evaluated_at,
                heartbeat_at=evaluated_at,
                heartbeat_age_seconds=0.0,
                heartbeat_freshness_status="FRESH",
                health_overall_status="HEALTHY",
                reason_code="FEED_FRESH",
                feed_freshness_status="FRESH",
                feed_reason_code="FEED_FRESH",
                feed_age_seconds=0.0,
            ),
        ),
        lag_by_symbol=(
            CanonicalFeatureLag(
                service_name="feature-builder",
                component_name="features",
                symbol="BTC/USD",
                evaluated_at=evaluated_at,
                latest_raw_event_received_at=evaluated_at,
                latest_feature_interval_begin=_candle(0).interval_begin,
                latest_feature_as_of_time=evaluated_at,
                time_lag_seconds=0.0,
                processing_lag_seconds=0.0,
                time_lag_reason_code="FEATURE_TIME_LAG_OK",
                processing_lag_reason_code="FEATURE_PROCESSING_LAG_OK",
                lag_breach=False,
                health_overall_status="HEALTHY",
                reason_code="FEATURE_LAG_OK",
            ),
        ),
    )


class FakeRepository:  # pylint: disable=too-many-instance-attributes
    def __init__(self, candles: list[FeatureCandle]) -> None:
        self.candles = candles
        self.states = {"BTC/USD": PaperEngineState(service_name="paper-trader", symbol="BTC/USD")}
        self.open_positions = {}
        self.positions = []
        self.ledger = []
        self.saved_adaptive_drift_states = []
        self.saved_adaptive_performance_windows = []
        self.saved_continual_learning_drift_caps = []
        self.continual_learning_profiles = []
        self.continual_learning_experiments = []
        self.continual_learning_events = []
        self.continual_learning_promotion_decisions = []
        self.risk_state = None
        self.risk_decisions = []
        self.reliability_states = {}
        self.reliability_events = []
        self.heartbeats = []
        self.order_requests: dict[str, OrderRequest] = {}
        self.order_events: dict[tuple[int, str], OrderLifecycleEvent] = {}
        self.decision_traces: dict[int, DecisionTraceRecord] = {}
        self.position_id = 0
        self.order_request_id = 0
        self.decision_trace_id = 0

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_engine_states(
        self,
        *,
        service_name: str,
        execution_mode: str,
        symbols: tuple[str, ...],
    ):
        del service_name, execution_mode, symbols
        return dict(self.states)

    async def load_open_positions(self, service_name: str, *, execution_mode: str):
        del service_name, execution_mode
        return dict(self.open_positions)

    async def load_cash_balance(
        self,
        *,
        service_name: str,
        execution_mode: str,
        initial_cash: float,
    ) -> float:
        del service_name, execution_mode
        return initial_cash + sum(entry.cash_flow for entry in self.ledger)

    async def fetch_new_feature_rows(
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
        last_processed_interval_begin,
    ):
        del symbol, source_exchange, interval_minutes
        if last_processed_interval_begin is None:
            return list(self.candles)
        return [row for row in self.candles if row.interval_begin > last_processed_interval_begin]

    async def insert_position(self, position):
        self.position_id += 1
        stored = replace(position, position_id=self.position_id)
        self.positions.append(stored)
        return self.position_id

    async def close_position(self, position) -> None:
        self.positions = [
            position if row.position_id == position.position_id else row
            for row in self.positions
        ]

    async def insert_ledger_entry(self, entry) -> None:
        self.ledger.append(entry)

    async def save_engine_state(self, state) -> None:
        self.states[state.symbol] = state

    async def fetch_latest_feature_row(
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
    ):
        del source_exchange, interval_minutes
        matching = [row for row in self.candles if row.symbol == symbol]
        return None if not matching else matching[-1]

    async def load_service_risk_state(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return self.risk_state

    async def save_service_risk_state(self, state: ServiceRiskState) -> None:
        self.risk_state = state

    async def insert_risk_decision(self, entry) -> None:
        self.risk_decisions.append(entry)

    async def ensure_decision_trace(self, trace):
        for existing in self.decision_traces.values():
            if (
                existing.service_name == trace.service_name
                and existing.execution_mode == trace.execution_mode
                and existing.signal_row_id == trace.signal_row_id
            ):
                return existing
        self.decision_trace_id += 1
        stored = replace(trace, decision_trace_id=self.decision_trace_id)
        self.decision_traces[stored.decision_trace_id] = stored
        return stored

    async def update_decision_trace(self, trace):
        self.decision_traces[trace.decision_trace_id] = trace
        return trace

    async def load_decision_trace(self, *, decision_trace_id: int):
        return self.decision_traces.get(decision_trace_id)

    async def load_reliability_state(self, *, service_name: str, component_name: str):
        del service_name
        return self.reliability_states.get(component_name)

    async def save_reliability_state(self, state: ReliabilityState) -> None:
        self.reliability_states[state.component_name] = state

    async def insert_reliability_event(self, event: RecoveryEvent) -> RecoveryEvent:
        stored = replace(event, event_id=len(self.reliability_events) + 1)
        self.reliability_events.append(stored)
        return stored

    async def save_service_heartbeat(self, heartbeat: ServiceHeartbeat) -> ServiceHeartbeat:
        stored = replace(heartbeat, heartbeat_id=len(self.heartbeats) + 1)
        self.heartbeats.append(stored)
        return stored

    async def load_positions(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return list(self.positions)

    async def ensure_order_request(self, order_request: OrderRequest) -> OrderRequest:
        existing = self.order_requests.get(order_request.idempotency_key)
        if existing is not None:
            return existing
        self.order_request_id += 1
        stored = replace(order_request, order_request_id=self.order_request_id)
        self.order_requests[stored.idempotency_key] = stored
        return stored

    async def load_order_request_by_idempotency_key(self, *, idempotency_key: str):
        return self.order_requests.get(idempotency_key)

    async def insert_order_event_if_absent(self, event: OrderLifecycleEvent) -> OrderLifecycleEvent:
        key = (event.order_request_id, event.lifecycle_state)
        self.order_events.setdefault(key, event)
        return self.order_events[key]

    async def load_order_events_since(
        self,
        *,
        service_name: str,
        execution_mode: str,
        since,
    ):
        del service_name, execution_mode
        return [
            event
            for event in sorted(
                self.order_events.values(),
                key=lambda item: (item.event_time, item.event_id or 0),
            )
            if event.event_time >= since
        ]

    async def load_decision_traces_since(
        self,
        *,
        service_name: str,
        execution_mode: str,
        since,
    ):
        del service_name, execution_mode
        return [
            trace
            for trace in sorted(
                self.decision_traces.values(),
                key=lambda item: (item.signal_as_of_time, item.decision_trace_id or 0),
            )
            if trace.signal_as_of_time >= since
        ]

    async def load_latest_prices(self, *, source_exchange: str, interval_minutes: int, symbols: tuple[str, ...]):
        del source_exchange, interval_minutes, symbols
        return {"BTC/USD": self.candles[-1].close_price}

    async def load_feature_rows_for_adaptation(
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
        feature_columns,
        limit: int,
    ):
        del source_exchange, interval_minutes
        rows = []
        for candle in self.candles:
            if candle.symbol != symbol:
                continue
            row = {
                "symbol": candle.symbol,
                "interval_begin": candle.interval_begin,
                "as_of_time": candle.as_of_time,
            }
            for feature_name in feature_columns:
                row[feature_name] = getattr(candle, feature_name, None)
            rows.append(row)
        return rows[-limit:]

    async def save_adaptive_drift_state(self, record) -> None:
        self.saved_adaptive_drift_states.append(record)

    async def load_trade_ledger_entries(
        self,
        *,
        service_name: str,
        execution_mode: str,
        since,
    ):
        del service_name, execution_mode
        return [entry for entry in self.ledger if entry.fill_time >= since]

    async def save_adaptive_performance_window(self, record) -> None:
        self.saved_adaptive_performance_windows.append(record)

    async def load_latest_adaptive_drift_state(self, *, symbol: str, regime_label: str):
        matches = [
            item
            for item in self.saved_adaptive_drift_states
            if item.symbol == symbol and item.regime_label == regime_label
        ]
        if not matches:
            return None
        return matches[-1]

    async def save_continual_learning_drift_cap(self, record) -> None:
        self.saved_continual_learning_drift_caps.append(record)

    async def load_continual_learning_profiles(self, *, limit: int):
        del limit
        return list(self.continual_learning_profiles)

    async def load_continual_learning_experiments(self, *, limit: int):
        del limit
        return list(self.continual_learning_experiments)

    async def load_continual_learning_events(self, *, limit: int):
        del limit
        return list(self.continual_learning_events)

    async def load_continual_learning_promotion_decisions(self, *, limit: int):
        del limit
        return list(self.continual_learning_promotion_decisions)

    async def load_active_continual_learning_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ):
        del execution_mode, symbol, regime_label
        for profile in self.continual_learning_profiles:
            if getattr(profile, "status", None) == "ACTIVE":
                return profile
        return None

    async def load_continual_learning_profile(self, *, profile_id: str):
        for profile in self.continual_learning_profiles:
            if getattr(profile, "profile_id", None) == profile_id:
                return profile
        return None

    async def load_latest_continual_learning_drift_cap(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ):
        matches = [
            item
            for item in self.saved_continual_learning_drift_caps
            if item.execution_mode_scope == execution_mode
            and item.symbol_scope == symbol
            and item.regime_scope == regime_label
        ]
        if not matches:
            return None
        return matches[-1]

    async def load_continual_learning_drift_caps(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ):
        matches = [
            item
            for item in self.saved_continual_learning_drift_caps
            if item.execution_mode_scope == execution_mode
            and item.symbol_scope == symbol
            and item.regime_scope == regime_label
        ]
        return matches[:limit]

    async def load_all_continual_learning_drift_caps(self, *, limit: int):
        return self.saved_continual_learning_drift_caps[:limit]


class FakeSignalClient:
    def __init__(self) -> None:
        self.calls = 0

    async def close(self) -> None:
        return None

    async def fetch_signal(self, *, symbol: str, interval_begin):
        del symbol
        self.calls += 1
        first_candle = _candle(0).interval_begin
        signal = "BUY" if interval_begin == first_candle else "HOLD"
        return SignalDecision(
            symbol="BTC/USD",
            signal=signal,
            reason=signal.lower(),
            prob_up=0.7 if signal == "BUY" else 0.5,
            prob_down=0.3 if signal == "BUY" else 0.5,
            confidence=0.7 if signal == "BUY" else 0.5,
            predicted_class="UP" if signal == "BUY" else "DOWN",
            row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
            as_of_time=_candle(0).as_of_time if signal == "BUY" else _candle(1).as_of_time,
            model_name="logistic_regression",
            model_version="m3-20260320T090000Z",
        )


class StaticSignalClient:
    def __init__(
        self,
        *,
        signal: str,
        trade_allowed: bool | None = True,
        decision_source: str | None = "model",
        signal_status: str | None = "MODEL_SIGNAL",
        reason_code: str | None = None,
    ) -> None:  # pylint: disable=too-many-arguments
        self.signal = signal
        self.trade_allowed = trade_allowed
        self.decision_source = decision_source
        self.signal_status = signal_status
        self.reason_code = reason_code

    async def close(self) -> None:
        return None

    async def fetch_signal(self, *, symbol: str, interval_begin):
        return SignalDecision(
            symbol=symbol,
            signal=self.signal,
            reason=self.signal.lower(),
            prob_up=0.7 if self.signal == "BUY" else 0.5,
            prob_down=0.3 if self.signal == "BUY" else 0.5,
            confidence=0.7 if self.signal == "BUY" else 0.0,
            predicted_class="UP" if self.signal == "BUY" else "UNKNOWN",
            row_id=f"{symbol}|{interval_begin.isoformat().replace('+00:00', 'Z')}",
            as_of_time=interval_begin + timedelta(minutes=5),
            model_name="logistic_regression",
            model_version="m3-20260320T090000Z",
            trade_allowed=self.trade_allowed,
            decision_source=self.decision_source,
            signal_status=self.signal_status,
            reason_code=self.reason_code,
        )


class FakeAlertRepository:
    def __init__(self) -> None:
        self.events = []
        self.states = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_state(self, *, fingerprint: str):
        return self.states.get(fingerprint)

    async def save_state(self, state) -> None:
        self.states[state.fingerprint] = state

    async def insert_event(self, event):
        stored = replace(
            event,
            event_id=len(self.events) + 1,
            created_at=event.event_time,
        )
        self.events.append(stored)
        return stored

    async def load_active_states(self, *, service_name: str, execution_mode: str):
        return [
            state
            for state in self.states.values()
            if state.service_name == service_name
            and state.execution_mode == execution_mode
            and state.is_active
        ]

    async def load_events_for_day(
        self,
        *,
        service_name: str,
        execution_mode: str,
        summary_date,
    ):
        return [
            event
            for event in self.events
            if event.service_name == service_name
            and event.execution_mode == execution_mode
            and event.event_time.date() == summary_date
        ]


def test_runner_does_not_duplicate_processed_candles(tmp_path: Path) -> None:
    """Re-running the same cycle should not duplicate fills after state persistence."""
    repository = FakeRepository([_candle(0), _candle(1)])
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient(),
    )

    asyncio.run(runner.run_once())
    first_ledger_count = len(repository.ledger)
    first_order_request_count = len(repository.order_requests)
    first_filled_event_count = len(
        [event for event in repository.order_events.values() if event.lifecycle_state == "FILLED"]
    )
    asyncio.run(runner.run_once())

    assert first_ledger_count == 1
    assert len(repository.ledger) == 1
    assert first_order_request_count == 1
    assert len(repository.order_requests) == 1
    assert first_filled_event_count == 1
    assert len([event for event in repository.order_events.values() if event.lifecycle_state == "FILLED"]) == 1
    assert repository.states["BTC/USD"].last_processed_interval_begin == _candle(1).interval_begin


def test_runner_persists_one_risk_decision_per_processed_signal(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0), _candle(1)])
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient(),
    )

    asyncio.run(runner.run_once())

    assert len(repository.risk_decisions) == 2
    assert all(entry.outcome in {"APPROVED", "MODIFIED", "BLOCKED"} for entry in repository.risk_decisions)
    assert all(entry.reason_codes for entry in repository.risk_decisions)
    assert len(repository.order_requests) == 1

    asyncio.run(runner.run_once())

    assert len(repository.risk_decisions) == 2
    assert len(repository.order_requests) == 1


def test_runner_startup_clears_stale_pending_signal_and_records_recovery_event(
    tmp_path: Path,
) -> None:
    repository = FakeRepository([_candle(0), _candle(1), _candle(2)])
    repository.states["BTC/USD"] = PaperEngineState(
        service_name="paper-trader",
        symbol="BTC/USD",
        pending_signal=PendingSignalState(
            signal="BUY",
            signal_interval_begin=_candle(0).interval_begin,
            signal_as_of_time=_candle(0).as_of_time,
            row_id=f"BTC/USD|{_candle(0).interval_begin.isoformat().replace('+00:00', 'Z')}",
            reason="buy",
            prob_up=0.7,
            prob_down=0.3,
            confidence=0.7,
            predicted_class="UP",
            model_name="logistic_regression",
            regime_label="TREND_UP",
            approved_notional=1000.0,
            risk_outcome="APPROVED",
            risk_reason_codes=("BUY_APPROVED",),
        ),
    )
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient(),
    )

    asyncio.run(runner.startup())

    assert repository.states["BTC/USD"].pending_signal is None
    assert repository.reliability_events
    assert repository.reliability_events[-1].reason_code == (
        "RECOVERY_STALE_PENDING_SIGNAL_CLEARED"
    )
    assert repository.heartbeats
    asyncio.run(runner.shutdown())


def test_runner_skips_signal_fetch_when_breaker_is_open(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0)])
    signal_client = FakeSignalClient()
    repository.reliability_states["signal_client"] = ReliabilityState(
        service_name="paper-trader",
        component_name="signal_client",
        health_overall_status="UNAVAILABLE",
        freshness_status="STALE",
        breaker_state="OPEN",
        failure_count=3,
        success_count=0,
        opened_at=datetime.now(timezone.utc),
        reason_code="SIGNAL_FETCH_FAILED",
        detail="breaker open",
    )
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=signal_client,
    )

    asyncio.run(runner.run_once())

    assert signal_client.calls == 0
    assert len(repository.risk_decisions) == 0
    assert repository.reliability_events
    assert repository.reliability_events[-1].reason_code == (
        "SIGNAL_FETCH_SKIPPED_BREAKER_OPEN"
    )


def test_runner_blocked_buy_writes_clear_risk_rationale(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0)])
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=StaticSignalClient(
            signal="BUY",
            trade_allowed=False,
        ),
    )

    asyncio.run(runner.run_once())

    assert len(repository.decision_traces) == 1
    trace = next(iter(repository.decision_traces.values()))
    assert trace.payload.risk is not None
    assert trace.payload.risk.outcome == "BLOCKED"
    assert trace.payload.risk.primary_reason_code == "TRADE_NOT_ALLOWED"
    assert trace.payload.blocked_trade is not None
    assert trace.payload.blocked_trade.blocked_stage == "risk"
    assert len(repository.order_requests) == 0


def test_runner_modified_buy_links_order_request_to_same_decision_trace(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        risk=replace(
            config.risk,
            max_exposure_per_asset=0.05,
            volatility_target_realized_vol=0.03,
            min_volatility_size_multiplier=0.40,
        ),
    )
    repository = FakeRepository([replace(_candle(0), realized_vol_12=0.10), _candle(1)])
    runner = PaperTradingRunner(
        config=config,
        repository=repository,
        signal_client=StaticSignalClient(signal="BUY"),
    )

    asyncio.run(runner.run_once())

    assert len(repository.decision_traces) == 2
    first_trace = sorted(
        repository.decision_traces.values(),
        key=lambda trace: trace.signal_interval_begin,
    )[0]
    assert first_trace.payload.risk is not None
    assert first_trace.payload.risk.outcome == "MODIFIED"
    assert first_trace.payload.risk.ordered_adjustments
    order_request = next(iter(repository.order_requests.values()))
    assert order_request.decision_trace_id == first_trace.decision_trace_id
    assert order_request.model_version == "m3-20260320T090000Z"


def test_runner_sell_exit_links_closed_position_and_ledger_to_exit_trace(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0), _candle(1), _candle(2)])
    signal_client = StaticSignalClient(signal="HOLD")

    async def _fetch_signal(*, symbol: str, interval_begin):
        del symbol
        signal = "BUY" if interval_begin == _candle(0).interval_begin else "SELL" if interval_begin == _candle(1).interval_begin else "HOLD"
        return SignalDecision(
            symbol="BTC/USD",
            signal=signal,
            reason=signal.lower(),
            prob_up=0.7 if signal == "BUY" else 0.3 if signal == "SELL" else 0.5,
            prob_down=0.3 if signal == "BUY" else 0.7 if signal == "SELL" else 0.5,
            confidence=0.7 if signal != "HOLD" else 0.5,
            predicted_class="UP" if signal == "BUY" else "DOWN",
            row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
            as_of_time=interval_begin + timedelta(minutes=5),
            model_name="logistic_regression",
            model_version="m3-20260320T090000Z",
        )

    signal_client.fetch_signal = _fetch_signal  # type: ignore[method-assign]
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=signal_client,
    )

    asyncio.run(runner.run_once())

    traces = sorted(
        repository.decision_traces.values(),
        key=lambda trace: trace.signal_interval_begin,
    )
    buy_trace = traces[0]
    sell_trace = traces[1]
    closed_position = next(position for position in repository.positions if position.status == "CLOSED")
    sell_ledger = next(entry for entry in repository.ledger if entry.action == "SELL")
    sell_events = [
        event
        for event in repository.order_events.values()
        if event.action == "SELL"
    ]

    assert closed_position.entry_decision_trace_id == buy_trace.decision_trace_id
    assert closed_position.exit_decision_trace_id == sell_trace.decision_trace_id
    assert sell_ledger.decision_trace_id == sell_trace.decision_trace_id
    assert all(
        event.decision_trace_id == sell_trace.decision_trace_id
        for event in sell_events
    )


def test_runner_reliability_hold_path_stays_non_actionable(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0)])
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=StaticSignalClient(
            signal="HOLD",
            trade_allowed=False,
            decision_source="reliability",
            signal_status="RELIABILITY_HOLD",
            reason_code="RELIABILITY_HOLD_STALE_FEATURE_ROW",
        ),
    )

    asyncio.run(runner.run_once())

    assert len(repository.order_requests) == 0
    assert len(repository.risk_decisions) == 1
    assert repository.risk_decisions[0].reason_codes == ("HOLD_NO_OP",)
    trace = next(iter(repository.decision_traces.values()))
    assert trace.payload.signal.decision_source == "reliability"
    assert trace.payload.risk is not None
    assert trace.payload.risk.outcome == "APPROVED"


def test_runner_writes_research_only_live_policy_challenger_artifacts_without_extra_orders(
    tmp_path: Path,
) -> None:
    repository = FakeRepository([_candle(0), _candle(1), _candle(2), _candle(3)])
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient(),
    )

    asyncio.run(runner.run_once())

    challenger_dir = (
        Path(runner.config.artifact_dir)
        / "research"
        / "policy_challengers"
    )
    summary_path = challenger_dir / "latest_scoreboard.json"
    observations_path = challenger_dir / "challenger_observations.jsonl"
    settlements_path = challenger_dir / "challenger_settlements.jsonl"

    assert len(repository.order_requests) == 1
    assert summary_path.is_file()
    assert observations_path.is_file()
    assert settlements_path.is_file()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["production_baseline"]["hypothetical_trade_count"] == 1
    assert any(
        candidate["candidate_name"] == "m7_research_long_only_v1"
        for candidate in summary_payload["candidate_results"]
    )


def test_runner_alerting_smoke_writes_daily_summary_and_alert_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    startup_report_path = tmp_path / "startup_report.json"
    startup_report_path.write_text(
        '{"startup_validation_passed": true, "errors": []}',
        encoding="utf-8",
    )
    evaluated_at = _candle(1).as_of_time
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "paper")
    monkeypatch.setenv("STREAMALPHA_STARTUP_REPORT_PATH", str(startup_report_path))
    monkeypatch.setattr("app.trading.runner.utc_now", lambda: evaluated_at)
    monkeypatch.setattr("app.alerting.service.utc_now", lambda: evaluated_at)

    repository = FakeRepository([_candle(0), _candle(1)])
    alert_repository = FakeAlertRepository()
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient(),
        alert_repository=alert_repository,
    )
    alerting_config = AlertingConfig(
        schema_version="m17_alerting_v1",
        order_failure_spike=OrderFailureSpikeConfig(
            window_minutes=60,
            warning_count=2,
            critical_count=4,
        ),
        signals=SignalAlertConfig(
            silence_window_intervals=3,
            flood_window_intervals=6,
            flood_warning_count=1,
            flood_critical_count=2,
        ),
        artifacts=AlertingArtifactConfig(
            daily_summary_dir=str(tmp_path / "operations" / "daily"),
            startup_safety_path=str(tmp_path / "operations" / "startup_safety.json"),
        ),
    )
    runner.alerting_config = alerting_config
    runner.alerting_service = OperationalAlertService(
        config=alerting_config,
        repository=alert_repository,
    )
    runner._last_system_reliability_snapshot = _system_reliability(evaluated_at)  # pylint: disable=protected-access

    asyncio.run(runner.startup())
    asyncio.run(runner.run_once())
    asyncio.run(runner.shutdown())

    categories = [event.category for event in alert_repository.events]
    summary_path = (
        Path(alerting_config.artifacts.daily_summary_dir)
        / f"{evaluated_at.date().isoformat()}.json"
    )

    assert "STARTUP_SAFETY" in categories
    assert any(
        event.category == "SIGNAL_FLOOD" and event.event_state == "OPEN"
        for event in alert_repository.events
    )
    assert summary_path.is_file()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["counts_by_category"]["SIGNAL_FLOOD"] == 1
    assert summary_payload["startup_safety_status"]["startup_safety_passed"] is True


def test_runner_persists_runtime_m19_drift_and_performance_truth(tmp_path: Path) -> None:
    candles = [
        replace(
            _candle(index),
            realized_vol_12=(0.02 if index < 3 else 0.11),
        )
        for index in range(6)
    ]
    repository = FakeRepository(candles)
    signal_client = StaticSignalClient(signal="HOLD")

    async def _fetch_signal(*, symbol: str, interval_begin):
        del symbol
        if interval_begin == candles[0].interval_begin:
            signal = "BUY"
        elif interval_begin == candles[1].interval_begin:
            signal = "SELL"
        else:
            signal = "HOLD"
        return SignalDecision(
            symbol="BTC/USD",
            signal=signal,
            reason=signal.lower(),
            prob_up=0.7 if signal == "BUY" else 0.3 if signal == "SELL" else 0.5,
            prob_down=0.3 if signal == "BUY" else 0.7 if signal == "SELL" else 0.5,
            confidence=0.7 if signal != "HOLD" else 0.5,
            predicted_class="UP" if signal == "BUY" else "DOWN",
            row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
            as_of_time=interval_begin + timedelta(minutes=5),
            model_name="logistic_regression",
            model_version="m3-20260320T090000Z",
            regime_label="RANGE",
        )

    signal_client.fetch_signal = _fetch_signal  # type: ignore[method-assign]
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=signal_client,
    )
    base_adaptation_config = load_adaptation_config(default_adaptation_config_path())
    runner.adaptation_service.config = replace(
        base_adaptation_config,
        rolling_windows=replace(
            base_adaptation_config.rolling_windows,
            trade_counts=(1,),
            day_windows=(30,),
        ),
        drift=replace(
            base_adaptation_config.drift,
            minimum_reference_samples=3,
            minimum_live_samples=3,
            features=("realized_vol_12",),
        ),
    )
    runner._last_system_reliability_snapshot = _system_reliability(candles[-1].as_of_time)  # pylint: disable=protected-access

    asyncio.run(runner.run_once())

    drift_keys = {
        (item.symbol, item.regime_label)
        for item in repository.saved_adaptive_drift_states
    }
    performance_keys = {
        (item.execution_mode, item.symbol, item.regime_label, item.window_id)
        for item in repository.saved_adaptive_performance_windows
    }

    assert ("BTC/USD", "ALL") in drift_keys
    assert ("ALL", "ALL") in drift_keys
    assert ("paper", "ALL", "ALL", "last_1_trades") in performance_keys
    assert ("paper", "ALL", "ALL", "last_30d") in performance_keys
    aggregate_window = next(
        item
        for item in repository.saved_adaptive_performance_windows
        if item.execution_mode == "paper"
        and item.symbol == "ALL"
        and item.regime_label == "ALL"
        and item.window_id == "last_30d"
    )
    assert aggregate_window.health_context["system_reliability_available"] is True


def test_runner_persists_runtime_m21_drift_caps_from_adaptive_drift_truth(
    tmp_path: Path,
) -> None:
    candles = [
        replace(
            _candle(index),
            realized_vol_12=(0.02 if index < 3 else 0.11),
        )
        for index in range(6)
    ]
    repository = FakeRepository(candles)
    signal_client = StaticSignalClient(signal="HOLD")

    async def _fetch_signal(*, symbol: str, interval_begin):
        del symbol
        signal = "BUY" if interval_begin == candles[0].interval_begin else "HOLD"
        return SignalDecision(
            symbol="BTC/USD",
            signal=signal,
            reason=signal.lower(),
            prob_up=0.7 if signal == "BUY" else 0.5,
            prob_down=0.3 if signal == "BUY" else 0.5,
            confidence=0.7 if signal == "BUY" else 0.5,
            predicted_class="UP" if signal == "BUY" else "DOWN",
            row_id=f"BTC/USD|{interval_begin.isoformat().replace('+00:00', 'Z')}",
            as_of_time=interval_begin + timedelta(minutes=5),
            model_name="logistic_regression",
            model_version="m3-20260320T090000Z",
            regime_label="RANGE",
        )

    signal_client.fetch_signal = _fetch_signal  # type: ignore[method-assign]
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=signal_client,
    )
    base_adaptation_config = load_adaptation_config(default_adaptation_config_path())
    runner.adaptation_service.config = replace(
        base_adaptation_config,
        rolling_windows=replace(
            base_adaptation_config.rolling_windows,
            trade_counts=(1,),
            day_windows=(30,),
        ),
        drift=replace(
            base_adaptation_config.drift,
            minimum_reference_samples=3,
            minimum_live_samples=3,
            features=("realized_vol_12",),
        ),
    )

    asyncio.run(runner.run_once())

    cap_keys = {
        (item.execution_mode_scope, item.symbol_scope, item.regime_scope, item.candidate_type)
        for item in repository.saved_continual_learning_drift_caps
    }

    assert ("paper", "ALL", "ALL", "CALIBRATION_OVERLAY") in cap_keys
    assert ("paper", "ALL", "ALL", "INCREMENTAL_SHADOW_CHALLENGER") in cap_keys
    assert ("paper", "BTC/USD", "ALL", "CALIBRATION_OVERLAY") in cap_keys
