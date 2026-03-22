"""Decision-opportunity normalization for M18."""

# pylint: disable=too-many-locals

from __future__ import annotations

from collections import defaultdict

from app.trading.live import (
    LIVE_SIGNAL_STALE,
    LIVE_SYSTEM_HEALTH_STALE,
    LIVE_SYSTEM_HEALTH_UNAVAILABLE,
)
from app.trading.schemas import (
    DecisionTraceRecord,
    OrderLifecycleEvent,
    PaperPosition,
    TradeLedgerEntry,
)

from app.evaluation.schemas import (
    DecisionOpportunity,
    FillSummary,
    OrderLifecycleSummary,
    PositionOutcomeSummary,
)


_RELIABILITY_HOLD_PREFIX = "RELIABILITY_HOLD_"
_SAFETY_BLOCK_PREFIX = "LIVE_"
_NON_SAFETY_LIVE_REASON_CODES = {
    LIVE_SIGNAL_STALE,
    LIVE_SYSTEM_HEALTH_STALE,
    LIVE_SYSTEM_HEALTH_UNAVAILABLE,
}


def build_decision_opportunities(
    *,
    decision_traces: list[DecisionTraceRecord],
    order_events: list[OrderLifecycleEvent],
    ledger_entries: list[TradeLedgerEntry],
    positions: list[PaperPosition],
) -> list[DecisionOpportunity]:
    """Build canonical M18 decision-opportunity rows from persisted truths."""
    events_by_trace: dict[int, list[OrderLifecycleEvent]] = defaultdict(list)
    for event in order_events:
        if event.decision_trace_id is not None:
            events_by_trace[event.decision_trace_id].append(event)
    for events in events_by_trace.values():
        events.sort(key=lambda item: (item.event_time, item.event_id or 0))

    ledger_by_trace: dict[int, list[TradeLedgerEntry]] = defaultdict(list)
    for entry in ledger_entries:
        if entry.decision_trace_id is not None:
            ledger_by_trace[entry.decision_trace_id].append(entry)
    for entries in ledger_by_trace.values():
        entries.sort(key=lambda item: (item.fill_time, item.created_at or item.fill_time))

    positions_by_trace: dict[int, list[PaperPosition]] = defaultdict(list)
    for position in positions:
        if position.entry_decision_trace_id is not None:
            positions_by_trace[position.entry_decision_trace_id].append(position)
        if (
            position.exit_decision_trace_id is not None
            and position.exit_decision_trace_id != position.entry_decision_trace_id
        ):
            positions_by_trace[position.exit_decision_trace_id].append(position)

    opportunities: list[DecisionOpportunity] = []
    for trace in sorted(
        decision_traces,
        key=lambda item: (item.execution_mode, item.signal_as_of_time, item.decision_trace_id or 0),
    ):
        trace_id = _require_trace_id(trace)
        threshold_snapshot = trace.payload.threshold_snapshot
        signal_payload = trace.payload.signal
        risk_payload = trace.payload.risk
        order_summary = _summarize_order_lifecycle(
            trace=trace,
            events=events_by_trace.get(trace_id, []),
        )
        fill_summary = _summarize_fill(
            execution_mode=trace.execution_mode,
            ledger_entries=ledger_by_trace.get(trace_id, []),
        )
        position_summary = _summarize_position(
            positions=positions_by_trace.get(trace_id, []),
        )
        signal_reason_code = signal_payload.reason_code
        reliability_blocked = bool(
            signal_payload.decision_source == "reliability"
            or (
                signal_reason_code is not None
                and signal_reason_code.startswith(_RELIABILITY_HOLD_PREFIX)
            )
            or (
                order_summary.terminal_reason_code
                in _NON_SAFETY_LIVE_REASON_CODES
            )
        )
        safety_blocked = bool(
            order_summary.terminal_reason_code is not None
            and order_summary.terminal_reason_code.startswith(_SAFETY_BLOCK_PREFIX)
            and order_summary.terminal_reason_code not in _NON_SAFETY_LIVE_REASON_CODES
            and order_summary.terminal_state == "REJECTED"
        )
        opportunities.append(
            DecisionOpportunity(
                service_name=trace.service_name,
                execution_mode=trace.execution_mode,
                symbol=trace.symbol,
                signal_row_id=trace.signal_row_id,
                decision_trace_id=trace_id,
                signal_interval_begin=trace.signal_interval_begin,
                signal_as_of_time=trace.signal_as_of_time,
                model_name=trace.model_name,
                model_version=trace.model_version,
                regime_label=threshold_snapshot.regime_label if threshold_snapshot else None,
                regime_run_id=threshold_snapshot.regime_run_id if threshold_snapshot else None,
                signal_action=signal_payload.signal,
                decision_source=signal_payload.decision_source,
                signal_reason_code=signal_reason_code,
                freshness_status=signal_payload.freshness_status,
                health_overall_status=signal_payload.health_overall_status,
                buy_prob_up=(
                    None if threshold_snapshot is None else threshold_snapshot.buy_prob_up
                ),
                sell_prob_up=(
                    None if threshold_snapshot is None else threshold_snapshot.sell_prob_up
                ),
                allow_new_long_entries=(
                    None
                    if threshold_snapshot is None
                    else threshold_snapshot.allow_new_long_entries
                ),
                model_only_action=_model_only_action(trace),
                regime_aware_action=_regime_aware_action(trace),
                risk_gated_action=_risk_gated_action(trace),
                executed_action=_executed_action(
                    signal_action=signal_payload.signal,
                    order_summary=order_summary,
                    fill_summary=fill_summary,
                ),
                risk_outcome=None if risk_payload is None else risk_payload.outcome,
                risk_primary_reason_code=(
                    None if risk_payload is None else risk_payload.primary_reason_code
                ),
                requested_notional=(
                    None if risk_payload is None else float(risk_payload.requested_notional)
                ),
                approved_notional=(
                    None if risk_payload is None else float(risk_payload.approved_notional)
                ),
                risk_reason_codes=(
                    ()
                    if risk_payload is None
                    else tuple(str(value) for value in risk_payload.reason_codes)
                ),
                safety_blocked=safety_blocked,
                reliability_blocked=reliability_blocked,
                order=order_summary,
                fill=fill_summary,
                position=position_summary,
            )
        )
    return opportunities


def _summarize_order_lifecycle(
    *,
    trace: DecisionTraceRecord,
    events: list[OrderLifecycleEvent],
) -> OrderLifecycleSummary:
    truth_status = "OBSERVED" if trace.execution_mode == "live" else "SIMULATED"
    if not events:
        return OrderLifecycleSummary(truth_status="MISSING")
    first_response = next(
        (
            event.event_time
            for event in events
            if event.lifecycle_state
            in {
                "SUBMITTED",
                "ACCEPTED",
                "PARTIALLY_FILLED",
                "FILLED",
                "REJECTED",
                "CANCELED",
                "FAILED",
            }
        ),
        None,
    )
    terminal_event = next(
        (
            event
            for event in reversed(events)
            if event.lifecycle_state in {"FILLED", "REJECTED", "CANCELED", "FAILED"}
        ),
        None,
    )
    latest_event = events[-1]
    return OrderLifecycleSummary(
        truth_status=truth_status,
        order_request_id=events[0].order_request_id,
        created_at=next(
            (event.event_time for event in events if event.lifecycle_state == "CREATED"),
            None,
        ),
        first_response_at=first_response,
        terminal_at=None if terminal_event is None else terminal_event.event_time,
        terminal_state=None if terminal_event is None else terminal_event.lifecycle_state,
        terminal_reason_code=(
            None if terminal_event is None else terminal_event.reason_code
        ),
        lifecycle_states=tuple(event.lifecycle_state for event in events),
        broker_name=latest_event.broker_name,
        account_id=latest_event.account_id,
        environment_name=latest_event.environment_name,
    )


def _summarize_fill(
    *,
    execution_mode: str,
    ledger_entries: list[TradeLedgerEntry],
) -> FillSummary:
    if execution_mode == "shadow":
        return FillSummary(truth_status="NOT_APPLICABLE")
    if not ledger_entries:
        return FillSummary(
            truth_status="SIMULATED" if execution_mode == "paper" else "MISSING"
        )
    entry = ledger_entries[0]
    return FillSummary(
        truth_status="SIMULATED" if execution_mode == "paper" else "OBSERVED",
        action=entry.action,
        fill_time=entry.fill_time,
        fill_price=entry.fill_price,
        notional=entry.notional,
        fee=entry.fee,
        slippage_bps=entry.slippage_bps,
    )


def _summarize_position(*, positions: list[PaperPosition]) -> PositionOutcomeSummary:
    if not positions:
        return PositionOutcomeSummary()
    position = positions[0]
    return PositionOutcomeSummary(
        position_id=position.position_id,
        position_status=position.status,
        opened_at=position.opened_at,
        closed_at=position.closed_at,
        realized_pnl=position.realized_pnl,
        realized_return=position.realized_return,
        entry_decision_trace_id=position.entry_decision_trace_id,
        exit_decision_trace_id=position.exit_decision_trace_id,
    )


def _model_only_action(trace: DecisionTraceRecord) -> str:
    predicted_class = trace.payload.prediction.predicted_class.upper()
    if predicted_class == "UP":
        return "BUY"
    if predicted_class == "DOWN":
        return "SELL"
    return "HOLD"


def _regime_aware_action(trace: DecisionTraceRecord) -> str:
    threshold_snapshot = trace.payload.threshold_snapshot
    if threshold_snapshot is None:
        return "HOLD"
    prob_up = trace.payload.prediction.prob_up
    if prob_up >= threshold_snapshot.buy_prob_up:
        if threshold_snapshot.allow_new_long_entries:
            return "BUY"
        return "HOLD"
    if prob_up <= threshold_snapshot.sell_prob_up:
        return "SELL"
    return "HOLD"


def _risk_gated_action(trace: DecisionTraceRecord) -> str:
    risk = trace.payload.risk
    signal_action = trace.payload.signal.signal
    if signal_action == "HOLD":
        return "HOLD"
    if risk is None:
        return signal_action
    if risk.outcome == "BLOCKED" or risk.approved_notional <= 0.0:
        return "HOLD"
    return signal_action


def _executed_action(
    *,
    signal_action: str,
    order_summary: OrderLifecycleSummary,
    fill_summary: FillSummary,
) -> str:
    if (
        fill_summary.truth_status in {"OBSERVED", "SIMULATED"}
        and fill_summary.action is not None
    ):
        return fill_summary.action
    if order_summary.terminal_state == "FILLED" and signal_action in {"BUY", "SELL"}:
        return signal_action
    return "HOLD"


def _require_trace_id(trace: DecisionTraceRecord) -> int:
    if trace.decision_trace_id is None:
        raise ValueError("DecisionTraceRecord must have decision_trace_id for M18")
    return trace.decision_trace_id
