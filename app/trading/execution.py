"""Execution abstraction for paper, shadow, and guarded live order handling."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Protocol

from app.common.time import to_rfc3339
from app.trading.alpaca import AlpacaClientError, AlpacaOrderConstraintError
from app.trading.config import PaperTradingConfig
from app.trading.engine import execute_pending_signal_only, process_candle
from app.trading.live import (
    LIVE_BROKER_SUBMIT_FAILED,
    LIVE_FAILURE_HARD_STOP_ACTIVE,
    LIVE_MANUAL_DISABLE_ACTIVE,
    LIVE_MAX_ORDER_NOTIONAL_EXCEEDED,
    LIVE_ORDER_ACCEPTED,
    LIVE_ORDER_FILLED,
    LIVE_ORDER_REJECTED,
    LIVE_STARTUP_CHECKS_NOT_PASSED,
    LIVE_SYMBOL_NOT_WHITELISTED,
    record_live_failure,
    record_live_success,
    refresh_manual_disable_state,
)
from app.trading.schemas import (
    BrokerAccount,
    BrokerSubmitResult,
    ExecutionMode,
    ExecutionResult,
    FeatureCandle,
    LiveSafetyState,
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


class BrokerClient(Protocol):  # pylint: disable=too-few-public-methods
    """Minimal broker client contract for the guarded live adapter."""

    broker_name: str

    async def validate_account(self) -> BrokerAccount:
        """Validate broker credentials and return normalized account details."""

    async def submit_order(
        self,
        *,
        order_request: OrderRequest,
        open_position: PaperPosition | None,
        candle: FeatureCandle,
    ) -> BrokerSubmitResult:
        """Submit one minimal broker order."""

    async def close(self) -> None:
        """Close any owned network resources."""


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

    async def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
        live_safety_state: LiveSafetyState | None = None,
    ) -> ExecutionResult:
        """Execute any due order request for one candle."""

    async def close(self) -> None:
        """Close any adapter-owned resources."""


def build_execution_adapter(
    mode: ExecutionMode,
    *,
    broker_client: BrokerClient | None = None,
) -> ExecutionAdapter:
    """Return the configured execution adapter."""
    if mode == "paper":
        return PaperExecutionAdapter()
    if mode == "shadow":
        return ShadowExecutionAdapter()
    if mode == "live":
        return LiveExecutionAdapter(broker_client=broker_client)
    raise ValueError(f"Unsupported execution mode: {mode}")


class PaperExecutionAdapter:  # pylint: disable=too-few-public-methods
    """Execution adapter that keeps the accepted simulated-fill paper path."""

    mode: ExecutionMode = "paper"

    async def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
        live_safety_state: LiveSafetyState | None = None,
    ) -> ExecutionResult:
        """Execute the due paper order, if any, with the accepted engine math."""
        del live_safety_state
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

    async def close(self) -> None:
        """Paper mode owns no network resources."""


class ShadowExecutionAdapter:  # pylint: disable=too-few-public-methods
    """Execution adapter for the M11 shadow mode."""

    mode: ExecutionMode = "shadow"

    async def execute_candle(  # pylint: disable=too-many-arguments
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
        live_safety_state: LiveSafetyState | None = None,
    ) -> ExecutionResult:
        """Execute the due shadow order, if any, without any external placement."""
        del live_safety_state
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

    async def close(self) -> None:
        """Shadow mode owns no network resources."""


class LiveExecutionAdapter:  # pylint: disable=too-few-public-methods
    """Guarded live adapter that submits through the configured broker client."""

    mode: ExecutionMode = "live"

    def __init__(self, *, broker_client: BrokerClient | None = None) -> None:
        self.broker_client = broker_client

    def set_broker_client(self, broker_client: BrokerClient) -> None:
        """Set the validated broker client resolved during live startup."""
        self.broker_client = broker_client

    async def execute_candle(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        config: PaperTradingConfig,
        candle: FeatureCandle,
        state: PaperEngineState,
        open_position: PaperPosition | None,
        signal: SignalDecision,
        portfolio: PortfolioContext,
        order_request: OrderRequest | None,
        live_safety_state: LiveSafetyState | None = None,
    ) -> ExecutionResult:
        """Execute one due live order after the M10 risk path has approved it."""
        del signal
        if live_safety_state is None:
            raise ValueError("Live execution requires a live safety state")

        refreshed_live_state = refresh_manual_disable_state(
            state=live_safety_state,
            manual_disable_path=config.execution.live.manual_disable_path,
        )
        if order_request is None:
            return ExecutionResult(
                state=_advance_state(state=state, candle=candle, clear_pending=False),
                open_position=open_position,
                live_safety_state=refreshed_live_state,
            )

        precheck_failure = _live_precheck_failure(
            config=config,
            order_request=order_request,
            live_safety_state=refreshed_live_state,
        )
        if precheck_failure is not None:
            return _build_live_rejection_result(
                candle=candle,
                state=state,
                open_position=open_position,
                order_request=order_request,
                live_safety_state=refreshed_live_state,
                reason_code=precheck_failure,
            )

        if self.broker_client is None:
            return _build_live_rejection_result(
                candle=candle,
                state=state,
                open_position=open_position,
                order_request=order_request,
                live_safety_state=record_live_failure(
                    state=refreshed_live_state,
                    threshold=config.execution.live.failure_hard_stop_threshold,
                    reason_code=LIVE_BROKER_SUBMIT_FAILED,
                ),
                reason_code=LIVE_BROKER_SUBMIT_FAILED,
                details="Live broker client was not initialized",
            )

        try:
            submit_result = await self.broker_client.submit_order(
                order_request=order_request,
                open_position=open_position,
                candle=candle,
            )
        except AlpacaOrderConstraintError as error:
            return _build_live_rejection_result(
                candle=candle,
                state=state,
                open_position=open_position,
                order_request=order_request,
                live_safety_state=refreshed_live_state,
                reason_code=error.reason_code,
                details=str(error),
            )
        except AlpacaClientError as error:
            return _build_live_rejection_result(
                candle=candle,
                state=state,
                open_position=open_position,
                order_request=order_request,
                live_safety_state=record_live_failure(
                    state=refreshed_live_state,
                    threshold=config.execution.live.failure_hard_stop_threshold,
                    reason_code=LIVE_BROKER_SUBMIT_FAILED,
                ),
                reason_code=LIVE_BROKER_SUBMIT_FAILED,
                details=str(error),
            )

        execution = execute_pending_signal_only(
            config=config,
            candle=candle,
            state=state,
            open_position=open_position,
            portfolio=portfolio,
        )
        terminal_state = _live_terminal_state(
            order_request=order_request,
            created_position=execution.created_position,
            closed_position=execution.closed_position,
        )
        next_live_state = (
            record_live_success(refreshed_live_state)
            if terminal_state == "FILLED"
            else record_live_failure(
                state=refreshed_live_state,
                threshold=config.execution.live.failure_hard_stop_threshold,
                reason_code=LIVE_ORDER_REJECTED,
            )
        )
        terminal_reason = (
            LIVE_ORDER_FILLED if terminal_state == "FILLED" else LIVE_ORDER_REJECTED
        )
        lifecycle_events = (
            _build_live_broker_event(
                order_request=order_request,
                event_time=candle.interval_begin,
                lifecycle_state="ACCEPTED",
                reason_code=LIVE_ORDER_ACCEPTED,
                submit_result=submit_result,
            ),
            _build_live_broker_event(
                order_request=order_request,
                event_time=candle.interval_begin,
                lifecycle_state=terminal_state,
                reason_code=terminal_reason,
                submit_result=submit_result,
            ),
        )
        return ExecutionResult(
            state=replace(execution.state, execution_mode=config.execution.mode),
            open_position=execution.open_position,
            lifecycle_events=lifecycle_events,
            created_position=execution.created_position,
            closed_position=execution.closed_position,
            ledger_entries=execution.ledger_entries,
            cash_delta=execution.cash_delta,
            live_safety_state=next_live_state,
        )

    async def close(self) -> None:
        """Close the broker client if one is attached."""
        if self.broker_client is not None:
            await self.broker_client.close()


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


def _live_precheck_failure(
    *,
    config: PaperTradingConfig,
    order_request: OrderRequest,
    live_safety_state: LiveSafetyState,
) -> str | None:
    if not live_safety_state.startup_checks_passed:
        return LIVE_STARTUP_CHECKS_NOT_PASSED
    if live_safety_state.manual_disable_active:
        return LIVE_MANUAL_DISABLE_ACTIVE
    if live_safety_state.failure_hard_stop_active:
        return LIVE_FAILURE_HARD_STOP_ACTIVE
    if order_request.symbol not in config.execution.live.symbol_whitelist:
        return LIVE_SYMBOL_NOT_WHITELISTED
    if order_request.approved_notional > config.execution.live.max_order_notional:
        return LIVE_MAX_ORDER_NOTIONAL_EXCEEDED
    return None


def _build_live_rejection_result(  # pylint: disable=too-many-arguments
    *,
    candle: FeatureCandle,
    state: PaperEngineState,
    open_position: PaperPosition | None,
    order_request: OrderRequest,
    live_safety_state: LiveSafetyState,
    reason_code: str,
    details: str | None = None,
) -> ExecutionResult:
    lifecycle_event = OrderLifecycleEvent(
        order_request_id=_require_order_request_id(order_request),
        service_name=order_request.service_name,
        execution_mode=order_request.execution_mode,
        symbol=order_request.symbol,
        action=order_request.action,
        lifecycle_state="REJECTED",
        event_time=candle.interval_begin,
        reason_code=reason_code,
        details=details,
        broker_name=live_safety_state.broker_name,
        account_id=live_safety_state.account_id,
        environment_name=live_safety_state.environment_name,
    )
    return ExecutionResult(
        state=_advance_state(state=state, candle=candle, clear_pending=True),
        open_position=open_position,
        lifecycle_events=(lifecycle_event,),
        live_safety_state=live_safety_state,
    )


def _build_live_broker_event(
    *,
    order_request: OrderRequest,
    event_time,
    lifecycle_state,
    reason_code: str,
    submit_result: BrokerSubmitResult,
) -> OrderLifecycleEvent:
    return OrderLifecycleEvent(
        order_request_id=_require_order_request_id(order_request),
        service_name=order_request.service_name,
        execution_mode=order_request.execution_mode,
        symbol=order_request.symbol,
        action=order_request.action,
        lifecycle_state=lifecycle_state,
        event_time=event_time,
        reason_code=reason_code,
        details=submit_result.details,
        external_order_id=submit_result.external_order_id,
        external_status=submit_result.external_status,
        account_id=submit_result.account_id,
        environment_name=submit_result.environment_name,
        broker_name=submit_result.broker_name,
    )


def _live_terminal_state(
    *,
    order_request: OrderRequest,
    created_position: PaperPosition | None,
    closed_position: PaperPosition | None,
) -> str:
    was_filled = (
        created_position is not None
        if order_request.action == "BUY"
        else closed_position is not None
    )
    return "FILLED" if was_filled else "REJECTED"


def _advance_state(
    *,
    state: PaperEngineState,
    candle: FeatureCandle,
    clear_pending: bool,
) -> PaperEngineState:
    return replace(
        state,
        execution_mode=state.execution_mode,
        last_processed_interval_begin=candle.interval_begin,
        pending_signal=None if clear_pending else state.pending_signal,
    )


def _require_order_request_id(order_request: OrderRequest) -> int:
    if order_request.order_request_id is None:
        raise ValueError("OrderRequest must have order_request_id before lifecycle events")
    return order_request.order_request_id


def _format_notional(value: float) -> str:
    return f"{round(value, 8):.8f}"
