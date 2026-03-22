"""Focused M18 normalization tests."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from datetime import datetime, timezone

from app.explainability.schemas import (
    DecisionTracePayload,
    DecisionTracePrediction,
    DecisionTraceRisk,
    DecisionTraceSignal,
    DecisionTracePortfolioContext,
    DecisionTraceServiceRiskState,
    PredictionExplanation,
    ThresholdSnapshot,
)
from app.evaluation.normalize import build_decision_opportunities
from app.trading.schemas import (
    DecisionTraceRecord,
    OrderLifecycleEvent,
    PaperPosition,
    TradeLedgerEntry,
)


def _ts(hour: int, minute: int) -> datetime:
    return datetime(2026, 3, 22, hour, minute, tzinfo=timezone.utc)


def _trace(
    *,
    trace_id: int,
    execution_mode: str,
    signal: str,
    decision_source: str = "model",
) -> DecisionTraceRecord:
    return DecisionTraceRecord(
        service_name="paper-trader",
        execution_mode=execution_mode,
        symbol="BTC/USD",
        signal=signal,
        signal_interval_begin=_ts(12, 0),
        signal_as_of_time=_ts(12, 5),
        signal_row_id="BTC/USD|2026-03-22T12:00:00Z",
        model_name="logistic_regression",
        model_version="m7-20260320T134537Z",
        payload=DecisionTracePayload(
            schema_version="m14_decision_trace_v1",
            service_name="paper-trader",
            execution_mode=execution_mode,
            symbol="BTC/USD",
            signal_row_id="BTC/USD|2026-03-22T12:00:00Z",
            signal_interval_begin="2026-03-22T12:00:00Z",
            signal_as_of_time="2026-03-22T12:05:00Z",
            model_name="logistic_regression",
            model_version="m7-20260320T134537Z",
            prediction=DecisionTracePrediction(
                model_name="logistic_regression",
                model_version="m7-20260320T134537Z",
                prob_up=0.70,
                prob_down=0.30,
                confidence=0.70,
                predicted_class="UP",
                prediction_explanation=PredictionExplanation(
                    method="ONE_AT_A_TIME_REFERENCE_ABLATION",
                    available=True,
                    reason_code="EXPLAINABILITY_AVAILABLE",
                    summary_text="top features available",
                ),
            ),
            signal=DecisionTraceSignal(
                signal=signal,
                reason="threshold decision",
                signal_status=(
                    "MODEL_SIGNAL" if signal != "HOLD" else "MODEL_HOLD"
                ),
                decision_source=decision_source,
                reason_code=(
                    None
                    if decision_source == "model"
                    else "RELIABILITY_HOLD_STALE_FEATURE_ROW"
                ),
            ),
            threshold_snapshot=ThresholdSnapshot(
                buy_prob_up=0.55,
                sell_prob_up=0.45,
                allow_new_long_entries=True,
                regime_label="TREND_UP",
                regime_run_id="m8-20260321T120000Z",
                regime_artifact_path="artifacts/regime/m8/20260321T120000Z/thresholds.json",
                high_vol_threshold=0.05,
                trend_abs_threshold=0.02,
            ),
            risk=DecisionTraceRisk(
                outcome="APPROVED",
                primary_reason_code="BUY_APPROVED",
                reason_codes=["BUY_APPROVED"],
                reason_texts=["approved"],
                requested_notional=1000.0,
                approved_notional=1000.0,
                portfolio_context=DecisionTracePortfolioContext(
                    available_cash=10000.0,
                    open_position_count=0,
                    current_equity=10000.0,
                    total_open_exposure_notional=0.0,
                    current_symbol_exposure_notional=0.0,
                ),
                service_risk_state=DecisionTraceServiceRiskState(
                    trading_day="2026-03-22",
                    realized_pnl_today=0.0,
                    equity_high_watermark=10000.0,
                    current_equity=10000.0,
                    loss_streak_count=0,
                    kill_switch_enabled=False,
                ),
            ),
        ),
        risk_outcome="APPROVED",
        decision_trace_id=trace_id,
        created_at=_ts(12, 5),
        updated_at=_ts(12, 5),
    )


def test_shadow_fill_truth_is_not_applicable() -> None:
    opportunities = build_decision_opportunities(
        decision_traces=[_trace(trace_id=101, execution_mode="shadow", signal="BUY")],
        order_events=[
            OrderLifecycleEvent(
                order_request_id=1,
                service_name="paper-trader",
                execution_mode="shadow",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="CREATED",
                event_time=_ts(12, 5),
                decision_trace_id=101,
            ),
            OrderLifecycleEvent(
                order_request_id=1,
                service_name="paper-trader",
                execution_mode="shadow",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="FILLED",
                event_time=_ts(12, 10),
                decision_trace_id=101,
            ),
        ],
        ledger_entries=[],
        positions=[],
    )

    assert len(opportunities) == 1
    row = opportunities[0]
    assert row.model_only_action == "BUY"
    assert row.regime_aware_action == "BUY"
    assert row.risk_gated_action == "BUY"
    assert row.executed_action == "BUY"
    assert row.fill.truth_status == "NOT_APPLICABLE"


def test_reliability_hold_is_marked_as_reliability_block() -> None:
    opportunities = build_decision_opportunities(
        decision_traces=[
            _trace(
                trace_id=102,
                execution_mode="paper",
                signal="HOLD",
                decision_source="reliability",
            )
        ],
        order_events=[],
        ledger_entries=[],
        positions=[],
    )

    assert opportunities[0].reliability_blocked is True


def test_paper_fill_and_position_are_attached() -> None:
    opportunities = build_decision_opportunities(
        decision_traces=[_trace(trace_id=103, execution_mode="paper", signal="BUY")],
        order_events=[
            OrderLifecycleEvent(
                order_request_id=2,
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="CREATED",
                event_time=_ts(12, 5),
                decision_trace_id=103,
            ),
            OrderLifecycleEvent(
                order_request_id=2,
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                lifecycle_state="FILLED",
                event_time=_ts(12, 10),
                decision_trace_id=103,
            ),
        ],
        ledger_entries=[
            TradeLedgerEntry(
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                action="BUY",
                reason="entry",
                fill_interval_begin=_ts(12, 10),
                fill_time=_ts(12, 10),
                fill_price=100.0,
                quantity=10.0,
                notional=1000.0,
                fee=1.0,
                slippage_bps=5.0,
                cash_flow=-1001.0,
                decision_trace_id=103,
            ),
        ],
        positions=[
            PaperPosition(
                service_name="paper-trader",
                execution_mode="paper",
                symbol="BTC/USD",
                status="CLOSED",
                entry_signal_interval_begin=_ts(12, 0),
                entry_signal_as_of_time=_ts(12, 5),
                entry_signal_row_id="BTC/USD|2026-03-22T12:00:00Z",
                entry_reason="buy",
                entry_model_name="logistic_regression",
                entry_prob_up=0.70,
                entry_confidence=0.70,
                entry_fill_interval_begin=_ts(12, 10),
                entry_fill_time=_ts(12, 10),
                entry_price=100.0,
                quantity=10.0,
                entry_notional=1000.0,
                entry_fee=1.0,
                stop_loss_price=95.0,
                take_profit_price=110.0,
                entry_order_request_id=2,
                entry_decision_trace_id=103,
                position_id=7,
                realized_pnl=50.0,
                realized_return=0.05,
            ),
        ],
    )

    row = opportunities[0]
    assert row.fill.truth_status == "SIMULATED"
    assert row.fill.fill_price == 100.0
    assert row.position.position_id == 7
    assert row.position.realized_pnl == 50.0
