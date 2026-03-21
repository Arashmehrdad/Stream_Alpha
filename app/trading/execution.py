"""Execution abstraction for M11 paper and shadow order handling."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Protocol

from app.common.time import to_rfc3339
from app.trading.config import PaperTradingConfig
from app.trading.engine import process_candle
from app.trading.schemas import (
    ExecutionMode,
    ExecutionResult,
    FeatureCandle,
    OrderLifecycleEvent,
    OrderRequest,
    PaperEngineState,
    PaperPosition,
    PendingSignalState,
    PortfolioContext,
    RiskDecision,
    SignalDecision,
)


PAPER_ORDER_ACCEPTED = "PAPER_ORDER_ACCEPTED"
PAPER_ORDER_FILLED = "PAPER_ORDER_FILLED"
PAPER_ORDER_REJECTED = "PAPER_ORDER_REJECTED"
SHADOW_ORDER_ACCEPTED = "SHADOW_ORDER_ACCEPTED"
SHADOW_ORDER_FILLED = "SHADOW_ORDER_FILLED"
SHADOW_ORDER_REJECTED = "SHADOW_ORDER_REJECTED"


def build_idempotency_key(  # pylint: disable=too-many-arguments
    *,
    service_name: str,
    execution_mode: ExecutionMode,
    symbol: str,
    action: str,
    signal_row_id: str,
    target_fill_interval_begin,
    approved_notional: float,
    version: int,
) -> str:
    """Build the deterministic M11 idempotency key for one intended order."""
    approved_notional_rounded = _format_notional(approved_notional)
    return "|".join(
        [
            f"v{version}",
            service_name,
            execution_mode,
            symbol,
            action,
            signal_row_id,
            to_rfc3339(target_fill_interval_begin),
            approved_notional_rounded,
        ]
    )


def build_order_request(
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    signal: SignalDecision,
    decision: RiskDecision,
) -> OrderRequest | None:
    """Build one deterministic order request for an actionable risk-approved signal."""
    if signal.signal == "BUY" and decision.outcome in {"APPROVED", "MODIFIED"}:
        action = "BUY"
    elif signal.signal == "SELL" and decision.approved_notional > 0.0:
        action = "SELL"
    else:
        return None

    target_fill_interval_begin = candle.interval_begin + timedelta(
        minutes=config.interval_minutes
    )
    return OrderRequest(
        service_name=config.service_name,
        execution_mode=config.execution.mode,
        symbol=signal.symbol,
        action=action,
        signal_interval_begin=candle.interval_begin,
        signal_as_of_time=signal.as_of_time,
        signal_row_id=signal.row_id,
        target_fill_interval_begin=target_fill_interval_begin,
        requested_notional=decision.requested_notional,
        approved_notional=decision.approved_notional,
        idempotency_key=build_idempotency_key(
            service_name=config.service_name,
            execution_mode=config.execution.mode,
            symbol=signal.symbol,
            action=action,
            signal_row_id=signal.row_id,
            target_fill_interval_begin=target_fill_interval_begin,
            approved_notional=decision.approved_notional,
            version=config.execution.idempotency_key_version,
        ),
        model_name=signal.model_name,
        confidence=signal.confidence,
        regime_label=signal.regime_label,
        regime_run_id=signal.regime_run_id,
        risk_outcome=decision.outcome,
        risk_reason_codes=decision.reason_codes,
    )


def build_created_event(
    *,
    order_request: OrderRequest,
    event_time,
) -> OrderLifecycleEvent:
    """Build the CREATED lifecycle event for a new order request."""
    return OrderLifecycleEvent(
        order_request_id=_require_order_request_id(order_request),
        service_name=order_request.service_name,
        execution_mode=order_request.execution_mode,
        symbol=order_request.symbol,
        action=order_request.action,
        lifecycle_state="CREATED",
        event_time=event_time,
        reason_code="ORDER_REQUEST_CREATED",
    )


def build_pending_order_request(
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    pending_signal: PendingSignalState,
) -> OrderRequest:
    """Build the due order request for a legacy or restart-recovered pending signal."""
    approved_notional = pending_signal.approved_notional or 0.0
    return OrderRequest(
        service_name=config.service_name,
        execution_mode=config.execution.mode,
        symbol=candle.symbol,
        action=pending_signal.signal,
        signal_interval_begin=pending_signal.signal_interval_begin,
        signal_as_of_time=pending_signal.signal_as_of_time,
        signal_row_id=pending_signal.row_id,
        target_fill_interval_begin=candle.interval_begin,
        requested_notional=approved_notional,
        approved_notional=approved_notional,
        idempotency_key=build_idempotency_key(
            service_name=config.service_name,
            execution_mode=config.execution.mode,
            symbol=candle.symbol,
            action=pending_signal.signal,
            signal_row_id=pending_signal.row_id,
            target_fill_interval_begin=candle.interval_begin,
            approved_notional=approved_notional,
            version=config.execution.idempotency_key_version,
        ),
        model_name=pending_signal.model_name,
        confidence=pending_signal.confidence,
        regime_label=pending_signal.regime_label,
        regime_run_id=pending_signal.regime_run_id,
        risk_outcome=pending_signal.risk_outcome,
        risk_reason_codes=pending_signal.risk_reason_codes,
        order_request_id=pending_signal.order_request_id,
    )


class ExecutionAdapter(Protocol):  # pylint: disable=too-few-public-methods
    """Small mode-specific execution adapter contract."""

    mode: ExecutionMode

    def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
    ) -> ExecutionResult:
        """Execute any due order request plus existing mechanical exits."""


def build_execution_adapter(mode: ExecutionMode) -> ExecutionAdapter:
    """Return the configured M11 execution adapter."""
    if mode == "paper":
        return PaperExecutionAdapter()
    if mode == "shadow":
        return ShadowExecutionAdapter()
    raise ValueError(f"Unsupported execution mode: {mode}")


class PaperExecutionAdapter:  # pylint: disable=too-few-public-methods
    """Execution adapter that keeps the accepted simulated-fill paper path."""

    mode: ExecutionMode = "paper"

    def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
    ) -> ExecutionResult:
        """Execute the due paper order, if any, with the accepted engine math."""
        engine_result = process_candle(
            config=config,
            candle=candle,
            state=state,
            open_position=open_position,
            signal=signal,
            portfolio=portfolio,
            next_pending_signal=None,
        )
        lifecycle_events = _terminal_lifecycle_events(
            order_request=order_request,
            fill_time=candle.interval_begin,
            created_position=engine_result.created_position,
            closed_position=engine_result.closed_position,
            accepted_reason_code=PAPER_ORDER_ACCEPTED,
            filled_reason_code=PAPER_ORDER_FILLED,
            rejected_reason_code=PAPER_ORDER_REJECTED,
        )
        return ExecutionResult(
            state=replace(engine_result.state, execution_mode=config.execution.mode),
            open_position=engine_result.open_position,
            lifecycle_events=lifecycle_events,
            created_position=engine_result.created_position,
            closed_position=engine_result.closed_position,
            ledger_entries=engine_result.ledger_entries,
            cash_delta=engine_result.cash_delta,
        )


class ShadowExecutionAdapter:  # pylint: disable=too-few-public-methods
    """Execution adapter for the M11 shadow mode."""

    mode: ExecutionMode = "shadow"

    def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
    ) -> ExecutionResult:
        """Execute the due shadow order, if any, without any external placement."""
        engine_result = process_candle(
            config=config,
            candle=candle,
            state=state,
            open_position=open_position,
            signal=signal,
            portfolio=portfolio,
            next_pending_signal=None,
        )
        lifecycle_events = _terminal_lifecycle_events(
            order_request=order_request,
            fill_time=candle.interval_begin,
            created_position=engine_result.created_position,
            closed_position=engine_result.closed_position,
            accepted_reason_code=SHADOW_ORDER_ACCEPTED,
            filled_reason_code=SHADOW_ORDER_FILLED,
            rejected_reason_code=SHADOW_ORDER_REJECTED,
        )
        return ExecutionResult(
            state=replace(engine_result.state, execution_mode=config.execution.mode),
            open_position=engine_result.open_position,
            lifecycle_events=lifecycle_events,
            created_position=engine_result.created_position,
            closed_position=engine_result.closed_position,
            ledger_entries=engine_result.ledger_entries,
            cash_delta=engine_result.cash_delta,
        )


def _terminal_lifecycle_events(  # pylint: disable=too-many-arguments
    *,
    order_request: OrderRequest | None,
    fill_time,
    created_position: PaperPosition | None,
    closed_position: PaperPosition | None,
    accepted_reason_code: str,
    filled_reason_code: str,
    rejected_reason_code: str,
) -> tuple[OrderLifecycleEvent, ...]:
    """Build terminal lifecycle events for a due order request on one candle."""
    if order_request is None:
        return ()

    accepted_event = OrderLifecycleEvent(
        order_request_id=_require_order_request_id(order_request),
        service_name=order_request.service_name,
        execution_mode=order_request.execution_mode,
        symbol=order_request.symbol,
        action=order_request.action,
        lifecycle_state="ACCEPTED",
        event_time=fill_time,
        reason_code=accepted_reason_code,
    )

    was_filled = (
        created_position is not None
        if order_request.action == "BUY"
        else closed_position is not None
    )
    terminal_state = "FILLED" if was_filled else "REJECTED"
    terminal_reason = filled_reason_code if was_filled else rejected_reason_code
    terminal_event = OrderLifecycleEvent(
        order_request_id=_require_order_request_id(order_request),
        service_name=order_request.service_name,
        execution_mode=order_request.execution_mode,
        symbol=order_request.symbol,
        action=order_request.action,
        lifecycle_state=terminal_state,
        event_time=fill_time,
        reason_code=terminal_reason,
    )
    return (accepted_event, terminal_event)


def _require_order_request_id(order_request: OrderRequest) -> int:
    if order_request.order_request_id is None:
        raise ValueError("OrderRequest must have order_request_id before lifecycle events")
    return order_request.order_request_id


def _format_notional(value: float) -> str:
    return f"{round(value, 8):.8f}"
