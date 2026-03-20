"""Pure candle-by-candle paper-trading engine for Stream Alpha M5."""

from __future__ import annotations

from dataclasses import replace

from app.common.time import parse_rfc3339, utc_now
from app.trading.config import PaperTradingConfig
from app.trading.risk import (
    calculate_entry_fill_price,
    calculate_exit_fill_price,
    calculate_fee,
    can_open_position,
    capped_entry_cash,
    evaluate_barrier_exit,
    next_cooldown_boundary,
)
from app.trading.schemas import (
    EngineResult,
    FeatureCandle,
    PaperEngineState,
    PaperPosition,
    PendingSignalState,
    PortfolioContext,
    SignalDecision,
    TradeLedgerEntry,
)

_USE_SIGNAL_PENDING = object()


def process_candle(  # pylint: disable=too-many-arguments
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    state: PaperEngineState,
    open_position: PaperPosition | None,
    signal: SignalDecision,
    portfolio: PortfolioContext,
    next_pending_signal: PendingSignalState | None | object = _USE_SIGNAL_PENDING,
) -> EngineResult:
    """Process one newly finalized candle exactly once."""
    working_position = open_position
    created_position: PaperPosition | None = None
    closed_position: PaperPosition | None = None
    ledger_entries: list[TradeLedgerEntry] = []
    cash_delta = 0.0
    next_state = PaperEngineState(
        service_name=state.service_name,
        symbol=state.symbol,
        last_processed_interval_begin=candle.interval_begin,
        cooldown_until_interval_begin=state.cooldown_until_interval_begin,
        pending_signal=state.pending_signal,
    )

    if state.pending_signal is not None:
        execution = _execute_pending_signal(
            config=config,
            candle=candle,
            state=next_state,
            open_position=working_position,
            portfolio=portfolio,
        )
        working_position = execution.open_position
        created_position = execution.created_position
        closed_position = execution.closed_position
        ledger_entries.extend(execution.ledger_entries)
        cash_delta += execution.cash_delta
        next_state = execution.state

    barrier_reason = evaluate_barrier_exit(working_position, candle)
    if barrier_reason is not None and working_position is not None:
        barrier_exit = _close_position(
            config=config,
            position=working_position,
            candle=candle,
            fill_price=calculate_exit_fill_price(
                working_position.stop_loss_price
                if barrier_reason == "STOP_LOSS"
                else working_position.take_profit_price,
                config.risk.slippage_bps,
            ),
            reason=barrier_reason,
            signal_state=None,
            regime_label=signal.regime_label,
        )
        working_position = None
        closed_position = barrier_exit.position
        ledger_entries.append(barrier_exit.ledger_entry)
        cash_delta += barrier_exit.cash_delta
        next_state = replace(
            next_state,
            cooldown_until_interval_begin=next_cooldown_boundary(
                candle,
                config.risk.cooldown_candles,
            ),
        )

    next_state = replace(
        next_state,
        pending_signal=(
            _build_pending_signal(signal)
            if next_pending_signal is _USE_SIGNAL_PENDING
            else next_pending_signal
        ),
    )

    return EngineResult(
        state=next_state,
        open_position=working_position,
        created_position=created_position,
        closed_position=closed_position,
        ledger_entries=tuple(ledger_entries),
        cash_delta=cash_delta,
    )


def _execute_pending_signal(
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    state: PaperEngineState,
    open_position: PaperPosition | None,
    portfolio: PortfolioContext,
) -> EngineResult:
    pending = state.pending_signal
    cleared_state = replace(state, pending_signal=None)
    if pending is None:
        return EngineResult(state=cleared_state, open_position=open_position)

    if pending.signal == "BUY":
        if not can_open_position(
            config=config,
            state=state,
            open_position=open_position,
            candle=candle,
            portfolio=portfolio,
        ):
            return EngineResult(state=cleared_state, open_position=open_position)
        entry = _open_position(
            config=config,
            candle=candle,
            pending=pending,
            available_cash=portfolio.available_cash,
        )
        return EngineResult(
            state=cleared_state,
            open_position=entry.position,
            created_position=entry.position,
            ledger_entries=(entry.ledger_entry,),
            cash_delta=entry.cash_delta,
        )

    if pending.signal == "SELL" and open_position is not None:
        exit_result = _close_position(
            config=config,
            position=open_position,
            candle=candle,
            fill_price=calculate_exit_fill_price(candle.open_price, config.risk.slippage_bps),
            reason="SELL_SIGNAL",
            signal_state=pending,
        )
        cooldown_boundary = next_cooldown_boundary(candle, config.risk.cooldown_candles)
        return EngineResult(
            state=replace(
                cleared_state,
                cooldown_until_interval_begin=cooldown_boundary,
            ),
            open_position=None,
            closed_position=exit_result.position,
            ledger_entries=(exit_result.ledger_entry,),
            cash_delta=exit_result.cash_delta,
        )

    return EngineResult(state=cleared_state, open_position=open_position)


class _OpenPositionResult:  # pylint: disable=too-few-public-methods
    def __init__(self, position: PaperPosition, ledger_entry: TradeLedgerEntry, cash_delta: float):
        self.position = position
        self.ledger_entry = ledger_entry
        self.cash_delta = cash_delta


def _open_position(
    *,
    config: PaperTradingConfig,
    candle: FeatureCandle,
    pending: PendingSignalState,
    available_cash: float,
) -> _OpenPositionResult:
    entry_price = calculate_entry_fill_price(candle.open_price, config.risk.slippage_bps)
    fee_rate = config.risk.fee_bps / 10_000.0
    if pending.approved_notional is None:
        cash_budget = capped_entry_cash(config=config, available_cash=available_cash)
        entry_notional = cash_budget / (1.0 + fee_rate)
    else:
        entry_notional = pending.approved_notional
    quantity = 0.0 if entry_price <= 0 else entry_notional / entry_price
    entry_fee = calculate_fee(entry_notional, config.risk.fee_bps)
    opened_at = utc_now()
    position = PaperPosition(
        service_name=config.service_name,
        symbol=candle.symbol,
        status="OPEN",
        entry_signal_interval_begin=pending.signal_interval_begin,
        entry_signal_as_of_time=pending.signal_as_of_time,
        entry_signal_row_id=pending.row_id,
        entry_reason=pending.reason,
        entry_model_name=pending.model_name,
        entry_prob_up=pending.prob_up,
        entry_confidence=pending.confidence,
        entry_fill_interval_begin=candle.interval_begin,
        entry_fill_time=candle.interval_begin,
        entry_price=entry_price,
        quantity=quantity,
        entry_notional=entry_notional,
        entry_fee=entry_fee,
        stop_loss_price=entry_price * (1.0 - config.risk.stop_loss_pct),
        take_profit_price=entry_price * (1.0 + config.risk.take_profit_pct),
        entry_regime_label=pending.regime_label,
        entry_approved_notional=pending.approved_notional,
        entry_risk_outcome=pending.risk_outcome,
        entry_risk_reason_codes=pending.risk_reason_codes,
        opened_at=opened_at,
        updated_at=opened_at,
    )
    ledger_entry = TradeLedgerEntry(
        service_name=config.service_name,
        symbol=candle.symbol,
        action="BUY",
        reason="SIGNAL_BUY",
        fill_interval_begin=candle.interval_begin,
        fill_time=candle.interval_begin,
        fill_price=entry_price,
        quantity=quantity,
        notional=entry_notional,
        fee=entry_fee,
        slippage_bps=config.risk.slippage_bps,
        cash_flow=-(entry_notional + entry_fee),
        signal_interval_begin=pending.signal_interval_begin,
        signal_as_of_time=pending.signal_as_of_time,
        signal_row_id=pending.row_id,
        model_name=pending.model_name,
        prob_up=pending.prob_up,
        prob_down=pending.prob_down,
        confidence=pending.confidence,
        regime_label=pending.regime_label,
        approved_notional=pending.approved_notional,
        risk_outcome=pending.risk_outcome,
        risk_reason_codes=pending.risk_reason_codes,
    )
    return _OpenPositionResult(position, ledger_entry, ledger_entry.cash_flow)


class _ClosePositionResult:  # pylint: disable=too-few-public-methods
    def __init__(self, position: PaperPosition, ledger_entry: TradeLedgerEntry, cash_delta: float):
        self.position = position
        self.ledger_entry = ledger_entry
        self.cash_delta = cash_delta


def _close_position(  # pylint: disable=too-many-arguments
    *,
    config: PaperTradingConfig,
    position: PaperPosition,
    candle: FeatureCandle,
    fill_price: float,
    reason: str,
    signal_state: PendingSignalState | None,
    regime_label: str | None = None,
) -> _ClosePositionResult:
    exit_notional = fill_price * position.quantity
    exit_fee = calculate_fee(exit_notional, config.risk.fee_bps)
    realized_pnl = (exit_notional - exit_fee) - (position.entry_notional + position.entry_fee)
    invested_cash = position.entry_notional + position.entry_fee
    realized_return = 0.0 if invested_cash == 0 else realized_pnl / invested_cash
    closed_at = utc_now()
    closed_position = replace(
        position,
        status="CLOSED",
        exit_reason=reason,
        exit_signal_interval_begin=(
            None if signal_state is None else signal_state.signal_interval_begin
        ),
        exit_signal_as_of_time=None if signal_state is None else signal_state.signal_as_of_time,
        exit_signal_row_id=None if signal_state is None else signal_state.row_id,
        exit_model_name=None if signal_state is None else signal_state.model_name,
        exit_prob_up=None if signal_state is None else signal_state.prob_up,
        exit_confidence=None if signal_state is None else signal_state.confidence,
        exit_fill_interval_begin=candle.interval_begin,
        exit_fill_time=candle.interval_begin if reason == "SELL_SIGNAL" else candle.as_of_time,
        exit_price=fill_price,
        exit_notional=exit_notional,
        exit_fee=exit_fee,
        realized_pnl=realized_pnl,
        realized_return=realized_return,
        exit_regime_label=(
            regime_label if signal_state is None else signal_state.regime_label
        ),
        closed_at=closed_at,
        updated_at=closed_at,
    )
    ledger_entry = TradeLedgerEntry(
        service_name=config.service_name,
        symbol=position.symbol,
        action="SELL",
        reason=reason,
        fill_interval_begin=candle.interval_begin,
        fill_time=candle.interval_begin if reason == "SELL_SIGNAL" else candle.as_of_time,
        fill_price=fill_price,
        quantity=position.quantity,
        notional=exit_notional,
        fee=exit_fee,
        slippage_bps=config.risk.slippage_bps,
        cash_flow=exit_notional - exit_fee,
        signal_interval_begin=None if signal_state is None else signal_state.signal_interval_begin,
        signal_as_of_time=None if signal_state is None else signal_state.signal_as_of_time,
        signal_row_id=None if signal_state is None else signal_state.row_id,
        model_name=None if signal_state is None else signal_state.model_name,
        prob_up=None if signal_state is None else signal_state.prob_up,
        prob_down=None if signal_state is None else signal_state.prob_down,
        confidence=None if signal_state is None else signal_state.confidence,
        regime_label=regime_label if signal_state is None else signal_state.regime_label,
        approved_notional=(
            None if signal_state is None else signal_state.approved_notional
        ),
        risk_outcome=(
            None if signal_state is None else signal_state.risk_outcome
        ),
        risk_reason_codes=(
            () if signal_state is None else signal_state.risk_reason_codes
        ),
        realized_pnl=realized_pnl,
    )
    return _ClosePositionResult(closed_position, ledger_entry, ledger_entry.cash_flow)


def _build_pending_signal(signal: SignalDecision) -> PendingSignalState | None:
    if signal.signal == "HOLD":
        return None
    return PendingSignalState(
        signal=signal.signal,
        signal_interval_begin=_parse_row_id_interval_begin(signal.row_id),
        signal_as_of_time=signal.as_of_time,
        row_id=signal.row_id,
        reason=signal.reason,
        prob_up=signal.prob_up,
        prob_down=signal.prob_down,
        confidence=signal.confidence,
        predicted_class=signal.predicted_class,
        model_name=signal.model_name,
        regime_label=signal.regime_label,
        regime_run_id=signal.regime_run_id,
        approved_notional=signal.approved_notional,
        risk_outcome=signal.risk_outcome,
        risk_reason_codes=signal.risk_reason_codes,
    )


def _parse_row_id_interval_begin(row_id: str):
    _, _, timestamp = row_id.partition("|")
    return parse_rfc3339(timestamp)
