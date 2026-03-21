"""Focused M12 guarded-live tests."""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=line-too-long,too-many-lines,use-implicit-booleaness-not-comparison

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.common.time import to_rfc3339
from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat
from app.trading.alpaca import AlpacaOrderConstraintError, AlpacaResponseError
from app.trading.config import (
    ExecutionConfig,
    LiveConfig,
    PaperProbeConfig,
    PaperTradingConfig,
    RiskConfig,
)
from app.trading.execution import LiveExecutionAdapter, build_order_request
from app.trading.live import (
    LIVE_CONFIRMATION_PHRASE,
    LIVE_MANUAL_DISABLE_ACTIVE,
    LIVE_MAX_ORDER_NOTIONAL_EXCEEDED,
    LIVE_PAPER_PROBE_INTEGER_QTY_REQUIRED,
    LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED,
    LIVE_PAPER_PROBE_MIN_ORDER_VALUE_REQUIRED,
    LIVE_STARTUP_CHECKS_NOT_PASSED,
    LIVE_SYMBOL_NOT_WHITELISTED,
    validate_live_startup,
    write_live_status_artifact,
    write_startup_checklist_artifact,
)
from app.trading.runner import PaperTradingRunner
from app.trading.schemas import (
    BrokerAccount,
    BrokerSubmitResult,
    FeatureCandle,
    LiveSafetyState,
    OrderLifecycleEvent,
    OrderRequest,
    PaperEngineState,
    PendingSignalState,
    PortfolioContext,
    RiskDecision,
    ServiceRiskState,
    SignalDecision,
)


def _live_config(  # pylint: disable=too-many-arguments
    tmp_path: Path,
    *,
    symbol_whitelist: tuple[str, ...] = ("BTC/USD",),
    max_order_notional: float = 25.0,
    failure_hard_stop_threshold: int = 2,
    paper_probe_enabled: bool = False,
    paper_probe_symbol_whitelist: tuple[str, ...] = ("DOGE/USD",),
    paper_probe_min_order_value_usd: float = 10.0,
    paper_probe_fixed_qty: int = 500,
    paper_probe_max_orders_per_run: int = 1,
) -> PaperTradingConfig:
    live_dir = tmp_path / "live"
    return PaperTradingConfig(
        service_name="live-trader",
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
            max_total_exposure=0.60,
            max_daily_loss_amount=250.0,
            max_drawdown_pct=0.15,
            loss_streak_limit=3,
            loss_streak_cooldown_candles=3,
            kill_switch_enabled=False,
            min_trade_notional=5.0,
            volatility_target_realized_vol=0.03,
            min_volatility_size_multiplier=0.40,
            enable_confidence_weighted_sizing=False,
            min_confidence_size_multiplier=0.50,
            regime_position_fraction_caps={"TREND_UP": 0.25},
        ),
        execution=ExecutionConfig(
            mode="live",
            idempotency_key_version=1,
            live=LiveConfig(
                enabled=True,
                expected_account_id="PA12345",
                expected_environment="paper",
                symbol_whitelist=symbol_whitelist,
                max_order_notional=max_order_notional,
                failure_hard_stop_threshold=failure_hard_stop_threshold,
                manual_disable_path=str(live_dir / "manual_disable.flag"),
                startup_checklist_path=str(live_dir / "startup_checklist.json"),
                live_status_path=str(live_dir / "live_status.json"),
                paper_probe=PaperProbeConfig(
                    enabled=paper_probe_enabled,
                    symbol_whitelist=paper_probe_symbol_whitelist,
                    integer_qty_only=True,
                    min_order_value_usd=paper_probe_min_order_value_usd,
                    fixed_qty=paper_probe_fixed_qty,
                    max_probe_orders_per_run=paper_probe_max_orders_per_run,
                ),
            ),
        ),
    )


def _candle(index: int = 0, *, symbol: str = "BTC/USD") -> FeatureCandle:
    interval_begin = datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc) + timedelta(
        minutes=5 * index
    )
    return FeatureCandle(
        id=index + 1,
        source_exchange="kraken",
        symbol=symbol,
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


def _signal(interval_begin: datetime, *, signal: str = "BUY") -> SignalDecision:
    return SignalDecision(
        symbol="BTC/USD",
        signal=signal,
        reason=signal.lower(),
        prob_up=0.7 if signal == "BUY" else 0.3 if signal == "SELL" else 0.5,
        prob_down=0.3 if signal == "BUY" else 0.7 if signal == "SELL" else 0.5,
        confidence=0.7 if signal != "HOLD" else 0.5,
        predicted_class="UP" if signal == "BUY" else "DOWN",
        row_id=f"BTC/USD|{to_rfc3339(interval_begin)}",
        as_of_time=interval_begin + timedelta(minutes=5),
        model_name="logistic_regression",
        regime_label="TREND_UP",
        regime_run_id="20260321T090000Z",
        trade_allowed=True,
    )


def _hold_signal(interval_begin: datetime) -> SignalDecision:
    return _signal(interval_begin, signal="HOLD")


def _risk_decision(*, signal: str = "BUY", approved_notional: float = 10.0) -> RiskDecision:
    return RiskDecision(
        service_name="live-trader",
        symbol="BTC/USD",
        signal=signal,
        outcome="APPROVED",
        approved_notional=approved_notional,
        requested_notional=approved_notional,
        reason_codes=("BUY_APPROVED",) if signal == "BUY" else ("SELL_EXIT_APPROVED",),
        regime_label="TREND_UP",
        regime_run_id="20260321T090000Z",
        trade_allowed=True,
    )


def _portfolio(*, available_cash: float = 10_000.0) -> PortfolioContext:
    return PortfolioContext(
        available_cash=available_cash,
        open_position_count=0,
        current_equity=available_cash,
        total_open_exposure_notional=0.0,
        current_symbol_exposure_notional=0.0,
    )


def _pending_state(
    *,
    candle: FeatureCandle,
    order_request: OrderRequest,
) -> PaperEngineState:
    pending_signal = PendingSignalState(
        signal=order_request.action,
        signal_interval_begin=order_request.signal_interval_begin,
        signal_as_of_time=order_request.signal_as_of_time,
        row_id=order_request.signal_row_id,
        reason=order_request.action.lower(),
        prob_up=0.7 if order_request.action == "BUY" else 0.3,
        prob_down=0.3 if order_request.action == "BUY" else 0.7,
        confidence=0.7,
        predicted_class="UP" if order_request.action == "BUY" else "DOWN",
        model_name="logistic_regression",
        regime_label=order_request.regime_label,
        regime_run_id=order_request.regime_run_id,
        approved_notional=order_request.approved_notional,
        risk_outcome=order_request.risk_outcome,
        order_request_id=order_request.order_request_id,
        order_request_idempotency_key=order_request.idempotency_key,
        risk_reason_codes=order_request.risk_reason_codes,
    )
    return PaperEngineState(
        service_name="live-trader",
        symbol=candle.symbol,
        execution_mode="live",
        pending_signal=pending_signal,
    )


def _live_safety_state(
    *,
    startup_checks_passed: bool = True,
    manual_disable_active: bool = False,
    consecutive_live_failures: int = 0,
    failure_hard_stop_active: bool = False,
    environment_name: str = "paper",
) -> LiveSafetyState:
    return LiveSafetyState(
        service_name="live-trader",
        execution_mode="live",
        broker_name="alpaca",
        live_enabled=True,
        startup_checks_passed=startup_checks_passed,
        startup_checks_passed_at=datetime(2026, 3, 21, 8, 55, tzinfo=timezone.utc),
        account_validated=True,
        account_id="PA12345",
        environment_name=environment_name,
        manual_disable_active=manual_disable_active,
        consecutive_live_failures=consecutive_live_failures,
        failure_hard_stop_active=failure_hard_stop_active,
        last_failure_reason=None,
        updated_at=datetime(2026, 3, 21, 8, 55, tzinfo=timezone.utc),
    )


def _order_request(config: PaperTradingConfig, *, approved_notional: float = 10.0) -> OrderRequest:
    signal_candle = _candle(0)
    due_candle = _candle(1)
    order_request = build_order_request(
        config=config,
        candle=signal_candle,
        signal=_signal(signal_candle.interval_begin),
        decision=_risk_decision(approved_notional=approved_notional),
    )
    assert order_request is not None
    assert order_request.target_fill_interval_begin == due_candle.interval_begin
    return replace(order_request, order_request_id=1)


class FakeBrokerClient:
    broker_name = "alpaca"

    def __init__(
        self,
        *,
        account: BrokerAccount | None = None,
        submit_results: list[BrokerSubmitResult] | None = None,
        submit_error: Exception | None = None,
    ) -> None:
        self.account = account or BrokerAccount(
            broker_name="alpaca",
            account_id="PA12345",
            environment_name="paper",
            status="ACTIVE",
        )
        self.submit_results = list(submit_results or [])
        self.submit_error = submit_error
        self.validate_calls = 0
        self.submit_calls: list[tuple[OrderRequest, object, FeatureCandle, dict[str, object]]] = []

    async def validate_account(self) -> BrokerAccount:
        self.validate_calls += 1
        return self.account

    async def submit_order(  # pylint: disable=too-many-arguments
        self,
        *,
        order_request: OrderRequest,
        open_position,
        candle: FeatureCandle,
        probe_policy_active: bool = False,
        probe_symbol: str | None = None,
        probe_qty: int | None = None,
    ) -> BrokerSubmitResult:
        self.submit_calls.append(
            (
                order_request,
                open_position,
                candle,
                {
                    "probe_policy_active": probe_policy_active,
                    "probe_symbol": probe_symbol,
                    "probe_qty": probe_qty,
                },
            )
        )
        if self.submit_error is not None:
            raise self.submit_error
        if self.submit_results:
            return self.submit_results.pop(0)
        return BrokerSubmitResult(
            broker_name="alpaca",
            external_order_id="ord-1",
            external_status="filled",
            account_id="PA12345",
            environment_name="paper",
            details='{"type":"market"}',
            probe_policy_active=probe_policy_active,
            probe_symbol=probe_symbol,
            probe_qty=probe_qty,
        )

    async def close(self) -> None:
        return None


class FakeRepository:  # pylint: disable=too-many-instance-attributes
    def __init__(self, candles: list[FeatureCandle]) -> None:
        self.candles = candles
        self.states = {
            "BTC/USD": PaperEngineState(
                service_name="live-trader",
                symbol="BTC/USD",
                execution_mode="live",
            )
        }
        self.open_positions = {}
        self.positions = []
        self.ledger = []
        self.risk_state = None
        self.risk_decisions = []
        self.reliability_states = {}
        self.reliability_events = []
        self.heartbeats = []
        self.live_safety_state = None
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
        return [
            row
            for row in self.candles
            if row.interval_begin > last_processed_interval_begin
        ]

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

    async def load_live_safety_state(self, *, service_name: str, execution_mode: str):
        del service_name, execution_mode
        return self.live_safety_state

    async def save_live_safety_state(self, state: LiveSafetyState) -> None:
        self.live_safety_state = state

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

    async def insert_order_event_if_absent(
        self,
        event: OrderLifecycleEvent,
    ) -> OrderLifecycleEvent:
        key = (event.order_request_id, event.lifecycle_state)
        self.order_events.setdefault(key, event)
        return self.order_events[key]


class FakeSignalClient:
    async def close(self) -> None:
        return None

    async def fetch_signal(self, *, symbol: str, interval_begin):
        del symbol
        signal = "BUY" if interval_begin == _candle(0).interval_begin else "HOLD"
        return _signal(interval_begin, signal=signal)


def _arm_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STREAMALPHA_ENABLE_LIVE", "true")
    monkeypatch.setenv("STREAMALPHA_LIVE_CONFIRM", LIVE_CONFIRMATION_PHRASE)
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def test_live_mode_config_validation_accepts_guarded_live_mode(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    assert config.execution.mode == "live"
    assert config.execution.live.enabled is True
    assert config.execution.live.expected_environment == "paper"


def test_paper_probe_policy_activates_only_in_paper_environment(tmp_path: Path) -> None:
    config = _live_config(tmp_path, paper_probe_enabled=True)
    order_request = _order_request(config, approved_notional=20.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)

    paper_result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert broker_client.submit_calls[0][3]["probe_policy_active"] is True
    assert paper_result.created_position is None
    assert paper_result.lifecycle_events[0].probe_policy_active is True

    live_broker_client = FakeBrokerClient()
    live_adapter = LiveExecutionAdapter(broker_client=live_broker_client)
    live_result = asyncio.run(
        live_adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="live"),
        )
    )

    assert live_broker_client.submit_calls[0][3]["probe_policy_active"] is False
    assert live_result.created_position is not None
    assert live_result.lifecycle_events[0].probe_policy_active is False


def test_paper_probe_symbol_whitelist_enforcement_uses_configured_probe_symbol(
    tmp_path: Path,
) -> None:
    config = _live_config(
        tmp_path,
        paper_probe_enabled=True,
        paper_probe_symbol_whitelist=("DOGE/USD",),
    )
    order_request = _order_request(config, approved_notional=20.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert broker_client.submit_calls[0][3]["probe_symbol"] == "DOGE/USD"
    assert result.lifecycle_events[0].probe_symbol == "DOGE/USD"


def test_paper_probe_integer_qty_enforcement_blocks_invalid_probe_config(
    tmp_path: Path,
) -> None:
    config = _live_config(
        tmp_path,
        paper_probe_enabled=True,
        paper_probe_fixed_qty=0,
    )
    order_request = _order_request(config, approved_notional=20.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_PAPER_PROBE_INTEGER_QTY_REQUIRED


def test_paper_probe_min_order_value_enforcement_blocks_small_approved_notional(
    tmp_path: Path,
) -> None:
    config = _live_config(
        tmp_path,
        paper_probe_enabled=True,
        paper_probe_min_order_value_usd=15.0,
    )
    order_request = _order_request(config, approved_notional=10.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_PAPER_PROBE_MIN_ORDER_VALUE_REQUIRED
    assert result.live_safety_state is not None
    assert result.live_safety_state.consecutive_live_failures == 0


def test_paper_probe_max_one_order_per_run_is_enforced(tmp_path: Path) -> None:
    config = _live_config(
        tmp_path,
        paper_probe_enabled=True,
        paper_probe_max_orders_per_run=1,
    )
    first_request = _order_request(config, approved_notional=20.0)
    second_request = replace(first_request, order_request_id=2, signal_row_id="BTC/USD|second")
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    adapter.begin_run()

    first_result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=first_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=first_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )
    second_result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(2),
            state=_pending_state(candle=_candle(2), order_request=second_request),
            open_position=None,
            signal=_hold_signal(_candle(2).interval_begin),
            portfolio=_portfolio(),
            order_request=second_request,
            live_safety_state=first_result.live_safety_state,
        )
    )

    assert len(broker_client.submit_calls) == 1
    assert second_result.lifecycle_events[0].reason_code == LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED


def test_canonical_live_path_is_unchanged_when_probe_is_disabled(tmp_path: Path) -> None:
    config = _live_config(tmp_path, paper_probe_enabled=False)
    order_request = _order_request(config, approved_notional=20.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert broker_client.submit_calls[0][3]["probe_policy_active"] is False
    assert result.created_position is not None
    assert len(result.ledger_entries) == 1


def test_paper_probe_orders_are_explicitly_tagged_in_audit(tmp_path: Path) -> None:
    config = _live_config(
        tmp_path,
        paper_probe_enabled=True,
        paper_probe_symbol_whitelist=("DOGE/USD",),
        paper_probe_fixed_qty=500,
    )
    order_request = _order_request(config, approved_notional=20.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(environment_name="paper"),
        )
    )

    assert result.lifecycle_events[0].probe_policy_active is True
    assert result.lifecycle_events[0].probe_symbol == "DOGE/USD"
    assert result.lifecycle_events[0].probe_qty == 500


def test_startup_validation_missing_apca_api_key_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("STREAMALPHA_ENABLE_LIVE", "true")
    monkeypatch.setenv("STREAMALPHA_LIVE_CONFIRM", LIVE_CONFIRMATION_PHRASE)

    checklist, state, _ = asyncio.run(validate_live_startup(config=_live_config(tmp_path)))

    assert checklist.passed is False
    assert state.startup_checks_passed is False
    assert any(check.name == "alpaca_api_key_present" and not check.passed for check in checklist.checks)


def test_startup_validation_missing_apca_api_secret_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("STREAMALPHA_ENABLE_LIVE", "true")
    monkeypatch.setenv("STREAMALPHA_LIVE_CONFIRM", LIVE_CONFIRMATION_PHRASE)

    checklist, state, _ = asyncio.run(validate_live_startup(config=_live_config(tmp_path)))

    assert checklist.passed is False
    assert state.startup_checks_passed is False
    assert any(
        check.name == "alpaca_api_secret_present" and not check.passed
        for check in checklist.checks
    )


def test_startup_validation_missing_alpaca_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    monkeypatch.setenv("STREAMALPHA_ENABLE_LIVE", "true")
    monkeypatch.setenv("STREAMALPHA_LIVE_CONFIRM", LIVE_CONFIRMATION_PHRASE)

    checklist, state, _ = asyncio.run(validate_live_startup(config=_live_config(tmp_path)))

    assert checklist.passed is False
    assert state.startup_checks_passed is False
    assert any(check.name == "alpaca_base_url_present" and not check.passed for check in checklist.checks)


def test_startup_validation_account_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _arm_live(monkeypatch)
    broker_client = FakeBrokerClient(
        account=BrokerAccount(
            broker_name="alpaca",
            account_id="WRONG-ACCOUNT",
            environment_name="paper",
            status="ACTIVE",
        )
    )

    checklist, _state, _ = asyncio.run(
        validate_live_startup(config=_live_config(tmp_path), broker_client=broker_client)
    )

    assert checklist.passed is False
    assert any(check.name == "account_id_match" and not check.passed for check in checklist.checks)


def test_startup_validation_environment_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _arm_live(monkeypatch)
    broker_client = FakeBrokerClient(
        account=BrokerAccount(
            broker_name="alpaca",
            account_id="PA12345",
            environment_name="live",
            status="ACTIVE",
        )
    )

    checklist, _state, _ = asyncio.run(
        validate_live_startup(config=_live_config(tmp_path), broker_client=broker_client)
    )

    assert checklist.passed is False
    assert any(
        check.name == "account_environment_match" and not check.passed
        for check in checklist.checks
    )


def test_live_not_armed_creates_no_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _live_config(tmp_path)
    order_request = _order_request(config)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)

    monkeypatch.delenv("STREAMALPHA_ENABLE_LIVE", raising=False)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(startup_checks_passed=False),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_STARTUP_CHECKS_NOT_PASSED
    assert result.state.pending_signal is None


def test_manual_disable_blocks_live_submit(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    Path(config.execution.live.manual_disable_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.execution.live.manual_disable_path).write_text("disabled", encoding="utf-8")
    order_request = _order_request(config)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_MANUAL_DISABLE_ACTIVE
    assert result.live_safety_state is not None
    assert result.live_safety_state.manual_disable_active is True


def test_non_whitelisted_symbol_blocks_live_submit(tmp_path: Path) -> None:
    config = _live_config(tmp_path, symbol_whitelist=("ETH/USD",))
    order_request = _order_request(config)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_SYMBOL_NOT_WHITELISTED


def test_order_above_max_order_notional_blocks_live_submit(tmp_path: Path) -> None:
    config = _live_config(tmp_path, max_order_notional=5.0)
    order_request = _order_request(config, approved_notional=10.0)
    broker_client = FakeBrokerClient()
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(),
        )
    )

    assert broker_client.submit_calls == []
    assert result.lifecycle_events[0].reason_code == LIVE_MAX_ORDER_NOTIONAL_EXCEEDED


def test_repeated_live_submit_failures_activate_hard_stop(tmp_path: Path) -> None:
    config = _live_config(tmp_path, failure_hard_stop_threshold=2)
    broker_client = FakeBrokerClient(
        submit_error=AlpacaResponseError("submit failed")
    )
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    first_order_request = _order_request(config, approved_notional=10.0)
    first_result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=first_order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=first_order_request,
            live_safety_state=_live_safety_state(),
        )
    )

    second_order_request = replace(first_order_request, order_request_id=2)
    second_result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(2),
            state=_pending_state(candle=_candle(2), order_request=second_order_request),
            open_position=None,
            signal=_hold_signal(_candle(2).interval_begin),
            portfolio=_portfolio(),
            order_request=second_order_request,
            live_safety_state=first_result.live_safety_state,
        )
    )

    assert first_result.live_safety_state is not None
    assert first_result.live_safety_state.consecutive_live_failures == 1
    assert second_result.live_safety_state is not None
    assert second_result.live_safety_state.failure_hard_stop_active is True
    assert second_result.live_safety_state.consecutive_live_failures == 2


def test_broker_order_constraint_rejects_without_incrementing_failure_count(
    tmp_path: Path,
) -> None:
    config = _live_config(tmp_path)
    order_request = _order_request(config, approved_notional=10.0)
    broker_client = FakeBrokerClient(
        submit_error=AlpacaOrderConstraintError(
            "ALPACA_CRYPTO_INTEGER_QTY_REQUIRED",
            "Alpaca PAPER crypto orders require integer quantity",
        )
    )
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(consecutive_live_failures=1),
        )
    )

    assert result.live_safety_state is not None
    assert result.live_safety_state.consecutive_live_failures == 1
    assert result.live_safety_state.failure_hard_stop_active is False
    assert result.lifecycle_events[0].reason_code == "ALPACA_CRYPTO_INTEGER_QTY_REQUIRED"


def test_successful_live_fill_resets_consecutive_failure_count(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    order_request = _order_request(config, approved_notional=10.0)
    broker_client = FakeBrokerClient(
        submit_results=[
            BrokerSubmitResult(
                broker_name="alpaca",
                external_order_id="ord-123",
                external_status="filled",
                account_id="PA12345",
                environment_name="paper",
                details='{"type":"market"}',
            )
        ]
    )
    adapter = LiveExecutionAdapter(broker_client=broker_client)
    result = asyncio.run(
        adapter.execute_candle(
            config=config,
            candle=_candle(1),
            state=_pending_state(candle=_candle(1), order_request=order_request),
            open_position=None,
            signal=_hold_signal(_candle(1).interval_begin),
            portfolio=_portfolio(),
            order_request=order_request,
            live_safety_state=_live_safety_state(consecutive_live_failures=2),
        )
    )

    assert result.live_safety_state is not None
    assert result.live_safety_state.consecutive_live_failures == 0
    assert [event.lifecycle_state for event in result.lifecycle_events] == [
        "ACCEPTED",
        "FILLED",
    ]
    assert result.created_position is not None
    assert len(result.ledger_entries) == 1


def test_live_order_intent_and_lifecycle_audit_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _arm_live(monkeypatch)
    candles = [_candle(0), _candle(1)]
    repository = FakeRepository(candles)
    broker_client = FakeBrokerClient()
    base_config = _live_config(tmp_path)
    config = replace(
        base_config,
        risk=replace(base_config.risk, initial_cash=80.0),
    )
    runner = PaperTradingRunner(
        config=config,
        repository=repository,
        signal_client=FakeSignalClient(),
        broker_client=broker_client,
    )

    asyncio.run(runner.startup())
    asyncio.run(runner.run_once())
    asyncio.run(runner.shutdown())

    assert repository.live_safety_state is not None
    assert repository.live_safety_state.startup_checks_passed is True
    assert len(repository.order_requests) == 1
    assert {event.lifecycle_state for event in repository.order_events.values()} == {
        "CREATED",
        "ACCEPTED",
        "FILLED",
    }
    filled_event = next(
        event
        for event in repository.order_events.values()
        if event.lifecycle_state == "FILLED"
    )
    assert filled_event.broker_name == "alpaca"
    assert filled_event.account_id == "PA12345"
    assert len(repository.positions) == 1
    assert len(repository.ledger) == 1


def test_startup_checklist_artifact_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _arm_live(monkeypatch)
    checklist, _state, _ = asyncio.run(
        validate_live_startup(
            config=_live_config(tmp_path),
            broker_client=FakeBrokerClient(),
        )
    )

    write_startup_checklist_artifact(
        checklist=checklist,
        artifact_path=_live_config(tmp_path).execution.live.startup_checklist_path,
    )

    payload = json.loads(
        Path(_live_config(tmp_path).execution.live.startup_checklist_path).read_text(
            encoding="utf-8"
        )
    )
    assert payload["passed"] is True
    assert "test-secret" not in json.dumps(payload)
    assert payload["validated_account_id"] == "PA12345"


def test_live_status_artifact_writing(tmp_path: Path) -> None:
    config = _live_config(tmp_path)
    state = _live_safety_state()

    write_live_status_artifact(state=state, config=config)

    payload = json.loads(
        Path(config.execution.live.live_status_path).read_text(encoding="utf-8")
    )
    assert payload["startup_checks_passed"] is True
    assert payload["account_id"] == "PA12345"
    assert payload["max_order_notional"] == config.execution.live.max_order_notional
