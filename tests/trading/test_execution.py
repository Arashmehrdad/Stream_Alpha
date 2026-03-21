"""Focused M11 execution abstraction tests."""

# pylint: disable=duplicate-code,missing-function-docstring,missing-class-docstring
# pylint: disable=too-few-public-methods

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.common.time import to_rfc3339
from app.trading.config import ExecutionConfig, PaperTradingConfig, RiskConfig
from app.trading.execution import build_idempotency_key, build_order_request
from app.trading.runner import PaperTradingRunner
from app.trading.schemas import (
    FeatureCandle,
    OrderLifecycleEvent,
    OrderRequest,
    PaperEngineState,
    RiskDecision,
    ServiceRiskState,
    SignalDecision,
)


def _config(tmp_path: Path, *, execution_mode: str = "paper") -> PaperTradingConfig:
    return PaperTradingConfig(
        service_name="paper-trader",
        source_exchange="kraken",
        source_table="feature_ohlc",
        interval_minutes=5,
        symbols=("BTC/USD",),
        inference_base_url="http://127.0.0.1:8000",
        poll_interval_seconds=5.0,
        artifact_dir=str(tmp_path / execution_mode),
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
            max_total_exposure=0.60,
            max_daily_loss_amount=250.0,
            max_drawdown_pct=0.15,
            loss_streak_limit=3,
            loss_streak_cooldown_candles=3,
            kill_switch_enabled=False,
            min_trade_notional=50.0,
            volatility_target_realized_vol=0.03,
            min_volatility_size_multiplier=0.40,
            enable_confidence_weighted_sizing=False,
            min_confidence_size_multiplier=0.50,
            regime_position_fraction_caps={
                "TREND_UP": 0.25,
                "RANGE": 0.20,
                "TREND_DOWN": 0.15,
                "HIGH_VOL": 0.10,
            },
        ),
        execution=ExecutionConfig(
            mode=execution_mode,
            idempotency_key_version=1,
        ),
    )


def _candle(index: int) -> FeatureCandle:
    interval_begin = datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index)
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
        realized_vol_12=0.01,
    )


class FakeRepository:  # pylint: disable=too-many-instance-attributes
    def __init__(self, candles: list[FeatureCandle]) -> None:
        self.candles = candles
        self.states = {"BTC/USD": PaperEngineState(service_name="paper-trader", symbol="BTC/USD")}
        self.open_positions = {}
        self.positions = []
        self.ledger = []
        self.risk_state = None
        self.risk_decisions = []
        self.order_requests: dict[str, OrderRequest] = {}
        self.order_events: dict[tuple[int, str], OrderLifecycleEvent] = {}
        self.position_id = 0
        self.order_request_id = 0

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

    async def load_service_risk_state(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return self.risk_state

    async def save_service_risk_state(self, state: ServiceRiskState) -> None:
        self.risk_state = state

    async def insert_risk_decision(self, entry) -> None:
        self.risk_decisions.append(entry)

    async def load_positions(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return list(self.positions)

    async def load_latest_prices(
        self,
        *,
        source_exchange: str,
        interval_minutes: int,
        symbols: tuple[str, ...],
    ):
        del source_exchange, interval_minutes, symbols
        return {"BTC/USD": self.candles[-1].close_price}

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


class FakeSignalClient:
    def __init__(self, actions: dict[datetime, str], *, trade_allowed: bool = True) -> None:
        self.actions = actions
        self.trade_allowed = trade_allowed

    async def close(self) -> None:
        return None

    async def fetch_signal(self, *, symbol: str, interval_begin):
        signal = self.actions.get(interval_begin, "HOLD")
        return SignalDecision(
            symbol=symbol,
            signal=signal,
            reason=signal.lower(),
            prob_up=0.7 if signal == "BUY" else 0.3 if signal == "SELL" else 0.5,
            prob_down=0.3 if signal == "BUY" else 0.7 if signal == "SELL" else 0.5,
            confidence=0.7 if signal != "HOLD" else 0.5,
            predicted_class="UP" if signal == "BUY" else "DOWN",
            row_id=f"{symbol}|{to_rfc3339(interval_begin)}",
            as_of_time=interval_begin + timedelta(minutes=5),
            model_name="logistic_regression",
            regime_label="TREND_UP",
            regime_run_id="20260321T090000Z",
            trade_allowed=self.trade_allowed,
        )


def test_idempotency_key_is_deterministic() -> None:
    candle = _candle(1)
    first_key = build_idempotency_key(
        service_name="paper-trader",
        execution_mode="paper",
        symbol="BTC/USD",
        action="BUY",
        signal_row_id="BTC/USD|2026-03-21T09:00:00Z",
        target_fill_interval_begin=candle.interval_begin,
        approved_notional=2495.00998003992,
        version=1,
    )
    second_key = build_idempotency_key(
        service_name="paper-trader",
        execution_mode="paper",
        symbol="BTC/USD",
        action="BUY",
        signal_row_id="BTC/USD|2026-03-21T09:00:00Z",
        target_fill_interval_begin=candle.interval_begin,
        approved_notional=2495.00998003992,
        version=1,
    )

    assert first_key == second_key


def test_duplicate_key_does_not_create_duplicate_request(tmp_path: Path) -> None:
    repository = FakeRepository([_candle(0)])
    config = _config(tmp_path)
    signal = asyncio.run(
        FakeSignalClient({_candle(0).interval_begin: "BUY"}).fetch_signal(
            symbol="BTC/USD",
            interval_begin=_candle(0).interval_begin,
        )
    )
    decision = RiskDecision(
        service_name=config.service_name,
        symbol="BTC/USD",
        signal="BUY",
        outcome="APPROVED",
        approved_notional=1000.0,
        requested_notional=1000.0,
        reason_codes=("BUY_APPROVED",),
        regime_label="TREND_UP",
        regime_run_id="20260321T090000Z",
        trade_allowed=True,
    )
    order_request = build_order_request(
        config=config,
        candle=_candle(0),
        signal=signal,
        decision=decision,
    )

    assert order_request is not None

    first_request = asyncio.run(repository.ensure_order_request(order_request))
    second_request = asyncio.run(repository.ensure_order_request(order_request))

    assert first_request.order_request_id == second_request.order_request_id
    assert len(repository.order_requests) == 1


def test_paper_mode_still_writes_positions_and_ledger(tmp_path: Path) -> None:
    candles = [_candle(0), _candle(1)]
    repository = FakeRepository(candles)
    runner = PaperTradingRunner(
        config=_config(tmp_path, execution_mode="paper"),
        repository=repository,
        signal_client=FakeSignalClient({_candle(0).interval_begin: "BUY"}),
    )

    asyncio.run(runner.run_once())

    assert len(repository.positions) == 1
    assert len(repository.ledger) == 1
    assert repository.positions[0].execution_mode == "paper"
    assert repository.ledger[0].execution_mode == "paper"
    assert len(repository.order_requests) == 1
    assert {event.lifecycle_state for event in repository.order_events.values()} == {
        "CREATED",
        "ACCEPTED",
        "FILLED",
    }


def test_shadow_mode_writes_order_audit_and_shadow_state(tmp_path: Path) -> None:
    candles = [_candle(0), _candle(1)]
    repository = FakeRepository(candles)
    runner = PaperTradingRunner(
        config=_config(tmp_path, execution_mode="shadow"),
        repository=repository,
        signal_client=FakeSignalClient({_candle(0).interval_begin: "BUY"}),
    )

    asyncio.run(runner.run_once())

    assert repository.states["BTC/USD"].execution_mode == "shadow"
    assert len(repository.order_requests) == 1
    assert all(request.execution_mode == "shadow" for request in repository.order_requests.values())
    assert all(event.execution_mode == "shadow" for event in repository.order_events.values())


def test_blocked_buy_creates_no_order_request(tmp_path: Path) -> None:
    candles = [_candle(0)]
    repository = FakeRepository(candles)
    runner = PaperTradingRunner(
        config=_config(tmp_path),
        repository=repository,
        signal_client=FakeSignalClient({_candle(0).interval_begin: "BUY"}, trade_allowed=False),
    )

    asyncio.run(runner.run_once())

    assert len(repository.order_requests) == 0
    assert len(repository.order_events) == 0
    assert len(repository.positions) == 0
    assert len(repository.ledger) == 0
