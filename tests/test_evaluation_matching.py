"""Focused M18 divergence classification tests."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from datetime import datetime, timezone

from app.evaluation.config import EvaluationConfig
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
    fill_price: float | None = 100.0,
    slippage_bps: float | None = 5.0,
    latency_ms: float | None = 0.0,
) -> DecisionOpportunity:
    first_response_at = (
        None
        if latency_ms is None
        else signal_time.replace(microsecond=int(latency_ms * 1000))
    )
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
            first_response_at=first_response_at,
            terminal_at=signal_time,
            terminal_state="FILLED" if signal_action != "HOLD" else None,
            terminal_reason_code="LIVE_MANUAL_DISABLE_ACTIVE" if safety_blocked else None,
            lifecycle_states=() if signal_action == "HOLD" else ("CREATED", "FILLED"),
        ),
        fill=FillSummary(
            truth_status="SIMULATED",
            action=signal_action if signal_action != "HOLD" else None,
            fill_time=signal_time if signal_action != "HOLD" else None,
            fill_price=fill_price if signal_action != "HOLD" else None,
            slippage_bps=slippage_bps if signal_action != "HOLD" else None,
        ),
        position=PositionOutcomeSummary(position_id=trace_id),
    )


def _evaluation_config(
    *,
    latency_ms: float = 25.0,
    fill_price_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> EvaluationConfig:
    return EvaluationConfig(
        schema_version="m18_evaluation_config_v1",
        latency_drift_ms_threshold=latency_ms,
        fill_price_drift_bps_threshold=fill_price_bps,
        slippage_drift_bps_threshold=slippage_bps,
        cost_aware_precision_horizon_notes="closed realized pnl",
        minimum_comparable_count_notes="null when unavailable",
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
        evaluation_config=_evaluation_config(),
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
        evaluation_config=_evaluation_config(),
    )

    reason_codes = {event.reason_code for event in events}
    assert "SIGNAL_ACTION_MISMATCH" in reason_codes
    assert "SAFETY_BLOCK_MISMATCH" in reason_codes


def test_latency_drift_only_fires_above_threshold() -> None:
    _, under_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=21,
                row_id="BTC/USD|latency",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                latency_ms=5.0,
            ),
            _opportunity(
                mode="shadow",
                trace_id=22,
                row_id="BTC/USD|latency",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                latency_ms=20.0,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(latency_ms=25.0),
    )
    _, over_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=23,
                row_id="BTC/USD|latency-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                latency_ms=5.0,
            ),
            _opportunity(
                mode="shadow",
                trace_id=24,
                row_id="BTC/USD|latency-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                latency_ms=40.0,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(latency_ms=25.0),
    )

    assert "LATENCY_DRIFT" not in {event.reason_code for event in under_threshold}
    assert "LATENCY_DRIFT" in {event.reason_code for event in over_threshold}


def test_fill_price_drift_only_fires_above_threshold() -> None:
    _, under_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=31,
                row_id="BTC/USD|fill",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                fill_price=100.00,
            ),
            _opportunity(
                mode="shadow",
                trace_id=32,
                row_id="BTC/USD|fill",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                fill_price=100.08,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(fill_price_bps=10.0),
    )
    _, over_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=33,
                row_id="BTC/USD|fill-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                fill_price=100.00,
            ),
            _opportunity(
                mode="shadow",
                trace_id=34,
                row_id="BTC/USD|fill-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                fill_price=100.20,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(fill_price_bps=10.0),
    )

    assert "FILL_PRICE_DRIFT" not in {event.reason_code for event in under_threshold}
    assert "FILL_PRICE_DRIFT" in {event.reason_code for event in over_threshold}


def test_slippage_drift_only_fires_above_threshold() -> None:
    _, under_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=41,
                row_id="BTC/USD|slippage",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                slippage_bps=5.0,
            ),
            _opportunity(
                mode="shadow",
                trace_id=42,
                row_id="BTC/USD|slippage",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                slippage_bps=9.0,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(slippage_bps=5.0),
    )
    _, over_threshold = build_comparison_windows(
        opportunities=[
            _opportunity(
                mode="paper",
                trace_id=43,
                row_id="BTC/USD|slippage-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                slippage_bps=5.0,
            ),
            _opportunity(
                mode="shadow",
                trace_id=44,
                row_id="BTC/USD|slippage-over",
                signal_action="BUY",
                signal_time=_ts(12, 5),
                slippage_bps=12.0,
            ),
        ],
        comparison_families=("paper_vs_shadow",),
        evaluation_config=_evaluation_config(slippage_bps=5.0),
    )

    assert "SLIPPAGE_DRIFT" not in {event.reason_code for event in under_threshold}
    assert "SLIPPAGE_DRIFT" in {event.reason_code for event in over_threshold}
