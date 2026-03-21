"""Canonical M14 decision-trace builders for the accepted trading path."""

from __future__ import annotations

from dataclasses import replace

from app.common.time import parse_rfc3339, to_rfc3339
from app.explainability.schemas import (
    DecisionTraceBlockedTrade,
    DecisionTracePayload,
    DecisionTracePortfolioContext,
    DecisionTracePrediction,
    DecisionTraceRisk,
    DecisionTraceServiceRiskState,
    DecisionTraceSignal,
    OrderedRiskAdjustment,
)
from app.trading.schemas import (
    DecisionTraceRecord,
    PortfolioContext,
    RiskDecision,
    ServiceRiskState,
    SignalDecision,
)


DECISION_TRACE_SCHEMA_VERSION = "m14_decision_trace_v1"


def build_initial_decision_trace(
    *,
    service_name: str,
    execution_mode: str,
    signal: SignalDecision,
) -> DecisionTraceRecord:
    """Build the canonical initial decision trace from the authoritative M4 signal."""
    signal_interval_begin = _parse_row_id_interval_begin(signal.row_id)
    model_version = _require_model_version(signal)
    payload = DecisionTracePayload(
        schema_version=DECISION_TRACE_SCHEMA_VERSION,
        service_name=service_name,
        execution_mode=execution_mode,
        symbol=signal.symbol,
        signal_row_id=signal.row_id,
        signal_interval_begin=to_rfc3339(signal_interval_begin),
        signal_as_of_time=to_rfc3339(signal.as_of_time),
        model_name=signal.model_name,
        model_version=model_version,
        prediction=DecisionTracePrediction(
            model_name=signal.model_name,
            model_version=model_version,
            prob_up=signal.prob_up,
            prob_down=signal.prob_down,
            confidence=signal.confidence,
            predicted_class=signal.predicted_class,
            top_features=[feature.model_copy(deep=True) for feature in signal.top_features],
            prediction_explanation=(
                None
                if signal.prediction_explanation is None
                else signal.prediction_explanation.model_copy(deep=True)
            ),
        ),
        signal=DecisionTraceSignal(
            signal=signal.signal,
            reason=signal.reason,
            signal_status=signal.signal_status,
            decision_source=signal.decision_source,
            reason_code=signal.reason_code,
            freshness_status=signal.freshness_status,
            health_overall_status=signal.health_overall_status,
            signal_explanation=(
                None
                if signal.signal_explanation is None
                else signal.signal_explanation.model_copy(deep=True)
            ),
        ),
        threshold_snapshot=(
            None
            if signal.threshold_snapshot is None
            else signal.threshold_snapshot.model_copy(deep=True)
        ),
        regime_reason=(
            None if signal.regime_reason is None else signal.regime_reason.model_copy(deep=True)
        ),
    )
    return DecisionTraceRecord(
        service_name=service_name,
        execution_mode=execution_mode,
        symbol=signal.symbol,
        signal=signal.signal,
        signal_interval_begin=signal_interval_begin,
        signal_as_of_time=signal.as_of_time,
        signal_row_id=signal.row_id,
        model_name=signal.model_name,
        model_version=model_version,
        payload=payload,
    )


def enrich_decision_trace_with_risk(
    *,
    trace: DecisionTraceRecord,
    decision: RiskDecision,
    portfolio: PortfolioContext,
    service_risk_state: ServiceRiskState,
) -> DecisionTraceRecord:
    """Attach the canonical M10 rationale to an existing decision trace."""
    risk_section = build_risk_section(
        decision=decision,
        portfolio=portfolio,
        service_risk_state=service_risk_state,
    )
    blocked_trade = None
    if decision.outcome == "BLOCKED":
        blocked_trade = DecisionTraceBlockedTrade(
            blocked_stage=decision.blocked_stage or "risk",
            reason_code=decision.primary_reason_code,
            reason_texts=list(decision.reason_texts),
        )
    return replace(
        trace,
        risk_outcome=decision.outcome,
        payload=trace.payload.model_copy(
            update={
                "risk": risk_section,
                "blocked_trade": blocked_trade,
            },
            deep=True,
        ),
    )


def build_risk_section(
    *,
    decision: RiskDecision,
    portfolio: PortfolioContext,
    service_risk_state: ServiceRiskState,
) -> DecisionTraceRisk:
    """Build the canonical JSONB risk section for one evaluated M10 decision."""
    return DecisionTraceRisk(
        outcome=decision.outcome,
        primary_reason_code=decision.primary_reason_code,
        reason_codes=list(decision.reason_codes),
        reason_texts=list(decision.reason_texts),
        requested_notional=decision.requested_notional,
        approved_notional=decision.approved_notional,
        portfolio_context=DecisionTracePortfolioContext(
            available_cash=portfolio.available_cash,
            open_position_count=portfolio.open_position_count,
            current_equity=portfolio.current_equity,
            total_open_exposure_notional=portfolio.total_open_exposure_notional,
            current_symbol_exposure_notional=portfolio.current_symbol_exposure_notional,
        ),
        service_risk_state=DecisionTraceServiceRiskState(
            trading_day=service_risk_state.trading_day.isoformat(),
            realized_pnl_today=service_risk_state.realized_pnl_today,
            equity_high_watermark=service_risk_state.equity_high_watermark,
            current_equity=service_risk_state.current_equity,
            loss_streak_count=service_risk_state.loss_streak_count,
            loss_streak_cooldown_until_interval_begin=(
                None
                if service_risk_state.loss_streak_cooldown_until_interval_begin is None
                else to_rfc3339(service_risk_state.loss_streak_cooldown_until_interval_begin)
            ),
            kill_switch_enabled=service_risk_state.kill_switch_enabled,
        ),
        ordered_adjustments=[
            OrderedRiskAdjustment(
                step_index=step.step_index,
                reason_code=step.reason_code,
                reason_text=step.reason_text,
                before_notional=step.before_notional,
                after_notional=step.after_notional,
            )
            for step in decision.ordered_adjustments
        ],
    )


def _parse_row_id_interval_begin(row_id: str):
    _, _, timestamp = row_id.partition("|")
    return parse_rfc3339(timestamp)


def _require_model_version(signal: SignalDecision) -> str:
    if signal.model_version is None:
        raise ValueError("SignalDecision.model_version is required for M14 decision traces")
    return signal.model_version
