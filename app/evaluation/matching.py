"""Divergence matching and classification for M18."""

# pylint: disable=too-many-arguments,too-many-branches,too-many-locals

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339
from app.evaluation.schemas import ComparisonWindow, DecisionOpportunity, DivergenceEvent


_FAMILY_MODES = {
    "paper_vs_shadow": ("paper", "shadow"),
    "shadow_vs_tiny_live": ("shadow", "live"),
    "paper_to_tiny_live": ("paper", "live"),
}
_EPSILON = 1e-9


def build_comparison_windows(
    *,
    opportunities: list[DecisionOpportunity],
    comparison_families: tuple[str, ...],
) -> tuple[list[ComparisonWindow], list[DivergenceEvent]]:
    """Build comparison windows plus canonical divergence events."""
    windows: list[ComparisonWindow] = []
    divergence_events: list[DivergenceEvent] = []
    by_mode: dict[str, list[DecisionOpportunity]] = {}
    for mode in ("paper", "shadow", "live"):
        by_mode[mode] = sorted(
            [item for item in opportunities if item.execution_mode == mode],
            key=lambda item: (item.signal_as_of_time, item.symbol, item.signal_row_id),
        )

    for family in comparison_families:
        left_mode, right_mode = _FAMILY_MODES[family]
        window, events = _compare_family(
            comparison_family=family,
            left_mode=left_mode,
            right_mode=right_mode,
            left_rows=by_mode[left_mode],
            right_rows=by_mode[right_mode],
        )
        windows.append(window)
        divergence_events.extend(events)
    divergence_events.sort(
        key=lambda item: (
            item.comparison_family,
            item.event_time,
            item.divergence_stage,
            item.reason_code,
            item.symbol or "",
            item.signal_row_id or "",
        )
    )
    return windows, divergence_events


def summarize_divergence_counts(
    divergence_events: list[DivergenceEvent],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return counts by family and by reason code."""
    by_family = Counter(event.comparison_family for event in divergence_events)
    by_reason = Counter(event.reason_code for event in divergence_events)
    return dict(sorted(by_family.items())), dict(sorted(by_reason.items()))


def _compare_family(
    *,
    comparison_family: str,
    left_mode: str,
    right_mode: str,
    left_rows: list[DecisionOpportunity],
    right_rows: list[DecisionOpportunity],
) -> tuple[ComparisonWindow, list[DivergenceEvent]]:
    left_start = None if not left_rows else left_rows[0].signal_as_of_time
    left_end = None if not left_rows else left_rows[-1].signal_as_of_time
    right_start = None if not right_rows else right_rows[0].signal_as_of_time
    right_end = None if not right_rows else right_rows[-1].signal_as_of_time
    overlap_start = _max_optional(left_start, right_start)
    overlap_end = _min_optional(left_end, right_end)
    left_index = {
        (row.symbol, row.signal_row_id): row
        for row in left_rows
        if _within_overlap(row.signal_as_of_time, overlap_start, overlap_end)
    }
    right_index = {
        (row.symbol, row.signal_row_id): row
        for row in right_rows
        if _within_overlap(row.signal_as_of_time, overlap_start, overlap_end)
    }
    matched_keys = sorted(set(left_index) & set(right_index))
    events: list[DivergenceEvent] = []
    for row in left_rows:
        if not _within_overlap(row.signal_as_of_time, overlap_start, overlap_end):
            events.append(
                _coverage_gap_event(
                    comparison_family=comparison_family,
                    left_mode=left_mode,
                    right_mode=right_mode,
                    row=row,
                    side="left",
                )
            )
    for row in right_rows:
        if not _within_overlap(row.signal_as_of_time, overlap_start, overlap_end):
            events.append(
                _coverage_gap_event(
                    comparison_family=comparison_family,
                    left_mode=left_mode,
                    right_mode=right_mode,
                    row=row,
                    side="right",
                )
            )
    for key in sorted(set(left_index) - set(right_index)):
        events.append(
            _missing_counterpart_event(
                comparison_family=comparison_family,
                left_mode=left_mode,
                right_mode=right_mode,
                row=left_index[key],
                missing_mode=right_mode,
            )
        )
    for key in sorted(set(right_index) - set(left_index)):
        events.append(
            _missing_counterpart_event(
                comparison_family=comparison_family,
                left_mode=left_mode,
                right_mode=right_mode,
                row=right_index[key],
                missing_mode=left_mode,
            )
        )
    for key in matched_keys:
        events.extend(
            _classify_pair(
                comparison_family=comparison_family,
                left_mode=left_mode,
                right_mode=right_mode,
                left=left_index[key],
                right=right_index[key],
            )
        )
    return (
        ComparisonWindow(
            comparison_family=comparison_family,
            left_mode=left_mode,
            right_mode=right_mode,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            left_count=len(left_rows),
            right_count=len(right_rows),
            matched_count=len(matched_keys),
        ),
        events,
    )


def _classify_pair(
    *,
    comparison_family: str,
    left_mode: str,
    right_mode: str,
    left: DecisionOpportunity,
    right: DecisionOpportunity,
) -> list[DivergenceEvent]:
    events: list[DivergenceEvent] = []
    if (
        left.model_version != right.model_version
        or left.regime_run_id != right.regime_run_id
        or _float_mismatch(left.buy_prob_up, right.buy_prob_up)
        or _float_mismatch(left.sell_prob_up, right.sell_prob_up)
        or left.allow_new_long_entries != right.allow_new_long_entries
    ):
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="decision",
                reason_code="CONFIG_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched decisions used different model or threshold context.",
                detail=(
                    f"left model={left.model_version} regime_run={left.regime_run_id}; "
                    f"right model={right.model_version} regime_run={right.regime_run_id}"
                ),
            )
        )
    if left.reliability_blocked != right.reliability_blocked:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="reliability",
                reason_code="STALE_INPUT_BLOCK",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Reliability or stale-input blocking differed across modes.",
                detail=(
                    f"left reliability_blocked={left.reliability_blocked} "
                    f"reason={left.signal_reason_code or left.order.terminal_reason_code}; "
                    f"right reliability_blocked={right.reliability_blocked} "
                    f"reason={right.signal_reason_code or right.order.terminal_reason_code}"
                ),
            )
        )
    elif left.signal_action != right.signal_action:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="decision",
                reason_code="SIGNAL_ACTION_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched signal actions diverged across modes.",
                detail=f"left={left.signal_action} right={right.signal_action}",
            )
        )
    if left.risk_outcome != right.risk_outcome:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="risk",
                reason_code="RISK_OUTCOME_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched risk outcomes diverged across modes.",
                detail=f"left={left.risk_outcome} right={right.risk_outcome}",
            )
        )
    elif _float_mismatch(left.approved_notional, right.approved_notional):
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="risk",
                reason_code="APPROVED_NOTIONAL_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched approved notionals diverged across modes.",
                detail=f"left={left.approved_notional} right={right.approved_notional}",
            )
        )
    if left.safety_blocked != right.safety_blocked:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="safety_gate",
                reason_code="SAFETY_BLOCK_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Guarded execution safety blocking differed across modes.",
                detail=(
                    f"left safety_blocked={left.safety_blocked} "
                    f"reason={left.order.terminal_reason_code}; "
                    f"right safety_blocked={right.safety_blocked} "
                    f"reason={right.order.terminal_reason_code}"
                ),
            )
        )
    left_has_intent = left.order.order_request_id is not None
    right_has_intent = right.order.order_request_id is not None
    if left_has_intent != right_has_intent:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="order_intent",
                reason_code="ORDER_REQUEST_MISSING",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="One side created an order intent while the counterpart did not.",
                detail=(
                    f"left order_request_id={left.order.order_request_id}; "
                    f"right order_request_id={right.order.order_request_id}"
                ),
            )
        )
    elif left.order.terminal_state != right.order.terminal_state:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="order_lifecycle",
                reason_code="ORDER_TERMINAL_STATE_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched order lifecycle terminal states diverged.",
                detail=f"left={left.order.terminal_state} right={right.order.terminal_state}",
            )
        )
    elif left.order.terminal_reason_code != right.order.terminal_reason_code:
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="order_lifecycle",
                reason_code="ORDER_REJECTION_MISMATCH",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched order terminal reason codes diverged.",
                detail=(
                    f"left={left.order.terminal_reason_code} "
                    f"right={right.order.terminal_reason_code}"
                ),
            )
        )
    if (
        left.fill.truth_status in {"OBSERVED", "SIMULATED"}
        and right.fill.truth_status in {"OBSERVED", "SIMULATED"}
    ):
        if _float_mismatch(left.fill.fill_price, right.fill.fill_price):
            events.append(
                _paired_event(
                    comparison_family=comparison_family,
                    divergence_stage="fill_quality",
                    reason_code="FILL_PRICE_DRIFT",
                    left_mode=left_mode,
                    right_mode=right_mode,
                    left=left,
                    right=right,
                    summary_text="Matched fill prices diverged across modes.",
                    detail=f"left={left.fill.fill_price} right={right.fill.fill_price}",
                )
            )
        if _float_mismatch(left.fill.slippage_bps, right.fill.slippage_bps):
            events.append(
                _paired_event(
                    comparison_family=comparison_family,
                    divergence_stage="fill_quality",
                    reason_code="SLIPPAGE_DRIFT",
                    left_mode=left_mode,
                    right_mode=right_mode,
                    left=left,
                    right=right,
                    summary_text="Matched slippage measurements diverged across modes.",
                    detail=f"left={left.fill.slippage_bps} right={right.fill.slippage_bps}",
                )
            )
    left_latency = _latency_ms(left)
    right_latency = _latency_ms(right)
    if _float_mismatch(left_latency, right_latency):
        events.append(
            _paired_event(
                comparison_family=comparison_family,
                divergence_stage="order_lifecycle",
                reason_code="LATENCY_DRIFT",
                left_mode=left_mode,
                right_mode=right_mode,
                left=left,
                right=right,
                summary_text="Matched order lifecycle latency diverged across modes.",
                detail=f"left={left_latency} right={right_latency}",
            )
        )
    return events


def _coverage_gap_event(
    *,
    comparison_family: str,
    left_mode: str,
    right_mode: str,
    row: DecisionOpportunity,
    side: str,
) -> DivergenceEvent:
    return DivergenceEvent(
        comparison_family=comparison_family,
        divergence_stage="coverage",
        reason_code="COVERAGE_GAP",
        left_mode=left_mode,
        right_mode=right_mode,
        event_time=row.signal_as_of_time,
        summary_text="Decision opportunity fell outside the overlap window.",
        detail=f"{side} side outside overlap at {to_rfc3339(row.signal_as_of_time)}",
        symbol=row.symbol,
        signal_row_id=row.signal_row_id,
        left_decision_trace_id=row.decision_trace_id if side == "left" else None,
        right_decision_trace_id=row.decision_trace_id if side == "right" else None,
        payload={"execution_mode": row.execution_mode},
    )


def _missing_counterpart_event(
    *,
    comparison_family: str,
    left_mode: str,
    right_mode: str,
    row: DecisionOpportunity,
    missing_mode: str,
) -> DivergenceEvent:
    return DivergenceEvent(
        comparison_family=comparison_family,
        divergence_stage="coverage",
        reason_code="MISSING_COUNTERPART",
        left_mode=left_mode,
        right_mode=right_mode,
        event_time=row.signal_as_of_time,
        summary_text="No matched decision trace existed in the counterpart mode.",
        detail=f"missing_mode={missing_mode}",
        symbol=row.symbol,
        signal_row_id=row.signal_row_id,
        left_decision_trace_id=(
            row.decision_trace_id if row.execution_mode == left_mode else None
        ),
        right_decision_trace_id=(
            row.decision_trace_id if row.execution_mode == right_mode else None
        ),
        payload={"row": make_json_safe(asdict(row))},
    )


def _paired_event(
    *,
    comparison_family: str,
    divergence_stage: str,
    reason_code: str,
    left_mode: str,
    right_mode: str,
    left: DecisionOpportunity,
    right: DecisionOpportunity,
    summary_text: str,
    detail: str | None,
) -> DivergenceEvent:
    # The event payload is intentionally explicit because it becomes a canonical
    # divergence artifact row and should remain easy to inspect.
    return DivergenceEvent(
        comparison_family=comparison_family,
        divergence_stage=divergence_stage,
        reason_code=reason_code,
        left_mode=left_mode,
        right_mode=right_mode,
        event_time=max(left.signal_as_of_time, right.signal_as_of_time),
        summary_text=summary_text,
        detail=detail,
        symbol=left.symbol,
        signal_row_id=left.signal_row_id,
        left_decision_trace_id=left.decision_trace_id,
        right_decision_trace_id=right.decision_trace_id,
        payload={
            "left": {
                "signal_action": left.signal_action,
                "risk_outcome": left.risk_outcome,
                "approved_notional": left.approved_notional,
                "terminal_state": left.order.terminal_state,
                "terminal_reason_code": left.order.terminal_reason_code,
            },
            "right": {
                "signal_action": right.signal_action,
                "risk_outcome": right.risk_outcome,
                "approved_notional": right.approved_notional,
                "terminal_state": right.order.terminal_state,
                "terminal_reason_code": right.order.terminal_reason_code,
            },
        },
    )


def _latency_ms(opportunity: DecisionOpportunity) -> float | None:
    if opportunity.order.created_at is None or opportunity.order.first_response_at is None:
        return None
    return (
        opportunity.order.first_response_at - opportunity.order.created_at
    ).total_seconds() * 1000.0


def _within_overlap(
    value: datetime,
    overlap_start: datetime | None,
    overlap_end: datetime | None,
) -> bool:
    if overlap_start is None or overlap_end is None:
        return False
    return overlap_start <= value <= overlap_end


def _max_optional(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_optional(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _float_mismatch(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return abs(left - right) > _EPSILON
