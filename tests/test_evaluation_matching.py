"""Focused M18 divergence classification tests."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from datetime import datetime, timezone

from app.evaluation.matching import build_comparison_windows
from app.evaluation.schemas import (
    DecisionOpportunity,
    FillSummary,
    OrderLifecycleSummary,
    PositionOutcomeSummary,
)


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 22, hour, minute, tzinfo=timezone.utc)


def _opportunity(
    *,
    mode: str,
    trace_id: int,
    row_id: str,
    signal_action: str,
    signal_time: datetime,
    risk_outcome: str = "APPROVED",
    safety_blocked: bool = False,
    reliability_blocked: bool = False,
) -> DecisionOpportunity:
    return DecisionOpportunity(
        service_name="paper-trader",
        execution_mode=mode,
        symbol="BTC/USD",
        signal_row_id=row_id,
        decision_trace_id=trace_id,
        signal_interval_begin=signal_time,
        signal_as_of_time=signal_time,
        model_name="logistic_regression",
        model_version="m7-20260320T134537Z",
        regime_label="TREND_UP",
        regime_run_id="m8-20260321T120000Z",
        signal_action=signal_action,
        decision_source="model",
        signal_reason_code=None,
        freshness_status="FRESH",
        health_overall_status="HEALTHY",
        buy_prob_up=0.55,
        sell_prob_up=0.45,
        allow_new_long_entries=True,
        model_only_action="BUY",
        regime_aware_action=signal_action,
        risk_gated_action=signal_action if risk_outcome != "BLOCKED" else "HOLD",
        executed_action=signal_action if signal_action != "HOLD" else "HOLD",
        risk_outcome=risk_outcome,
        risk_primary_reason_code="BUY_APPROVED" if risk_outcome != "BLOCKED" else "KILL_SWITCH",
        requested_notional=1000.0,
        approved_notional=0.0 if risk_outcome == "BLOCKED" else 1000.0,
        safety_blocked=safety_blocked,
        reliability_blocked=reliability_blocked,
        order=OrderLifecycleSummary(
            truth_status="SIMULATED",
            order_request_id=trace_id,
            created_at=signal_time,
            first_response_at=signal_time,
            terminal_at=signal_time,
            terminal_state="FILLED" if signal_action != "HOLD" else None,
            terminal_reason_code="LIVE_MANUAL_DISABLE_ACTIVE" if safety_blocked else None,
            lifecycle_states=() if signal_action == "HOLD" else ("CREATED", "FILLED"),
        ),
        fill=FillSummary(
            truth_status="SIMULATED",
            action=signal_action if signal_action != "HOLD" else None,
            fill_time=signal_time if signal_action != "HOLD" else None,
            fill_price=100.0 if signal_action != "HOLD" else None,
            slippage_bps=5.0 if signal_action != "HOLD" else None,
        ),
        position=PositionOutcomeSummary(position_id=trace_id),
    )


def test_overlap_gap_and_missing_counterpart_are_explicit() -> None:
    windows, events = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=1,
                row_id="BTC/USD|1",
                signal_action="BUY",
                signal_time=_ts(12, 0),
            ),
            _opportunity(
                mode="shadow",
                trace_id=2,
                row_id="BTC/USD|2",
                signal_action="BUY",
                signal_time=_ts(12, 10),
            ),
        ],
        comparison_families=("paper_vs_shadow",),
    )

    assert windows[0].matched_count == 0
    assert {event.reason_code for event in events} == {"COVERAGE_GAP"}


def test_signal_and_safety_divergence_reason_codes_are_stable() -> None:
    _, events = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=10,
                row_id="BTC/USD|same",
                signal_action="BUY",
                signal_time=_ts(12, 5),
            ),
            _opportunity(
                mode="live",
                trace_id=11,
                row_id="BTC/USD|same",
                signal_action="HOLD",
                signal_time=_ts(12, 5),
                safety_blocked=True,
            ),
        ],
        comparison_families=("paper_to_tiny_live",),
    )

    reason_codes = {event.reason_code for event in events}
    assert "SIGNAL_ACTION_MISMATCH" in reason_codes
    assert "SAFETY_BLOCK_MISMATCH" in reason_codes
