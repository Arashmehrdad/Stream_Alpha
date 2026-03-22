"""Canonical M14 decision-trace builders for the accepted trading path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from app.common.serialization import make_json_safe
from app.common.time import parse_rfc3339, to_rfc3339
from app.adaptation.schemas import AdaptationContextPayload, EffectiveThresholds
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
    OrderLifecycleEvent,
    OrderRequest,
    PortfolioContext,
    RiskDecision,
    ServiceRiskState,
    SignalDecision,
)


DECISION_TRACE_SCHEMA_VERSION = "m14_decision_trace_v1"
RATIONALE_REPORT_SCHEMA_VERSION = "m14_rationale_report_v1"


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
        adaptation=_build_adaptation_payload(signal),
        ensemble=(
            None if signal.ensemble is None else signal.ensemble.model_copy(deep=True)
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


def _build_adaptation_payload(signal: SignalDecision) -> AdaptationContextPayload | None:
    if (
        signal.adaptation_profile_id is None
        and not signal.adaptation_reason_codes
        and signal.calibrated_confidence is None
        and signal.adaptive_size_multiplier is None
    ):
        return None
    effective_thresholds = None
    if signal.effective_buy_prob_up is not None and signal.effective_sell_prob_up is not None:
        effective_thresholds = EffectiveThresholds(
            buy_prob_up=signal.effective_buy_prob_up,
            sell_prob_up=signal.effective_sell_prob_up,
        )
    return AdaptationContextPayload(
        adaptation_profile_id=signal.adaptation_profile_id,
        threshold_policy_id=signal.adaptation_profile_id,
        sizing_policy_id=signal.adaptation_profile_id,
        calibration_profile_id=signal.adaptation_profile_id,
        drift_status=signal.drift_status,
        recent_performance_summary=signal.recent_performance_summary,
        adaptation_reason_codes=list(signal.adaptation_reason_codes),
        frozen_by_health_gate=signal.frozen_by_health_gate,
        calibrated_confidence=signal.calibrated_confidence,
        adaptive_size_multiplier=signal.adaptive_size_multiplier,
        effective_thresholds=effective_thresholds,
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


def write_rationale_reports(
    *,
    trace: DecisionTraceRecord,
    artifact_root: Path,
    order_request: OrderRequest | None = None,
    lifecycle_events: Sequence[OrderLifecycleEvent] = (),
) -> DecisionTraceRecord:
    """Write deterministic JSON and Markdown rationale reports for one trace."""
    if trace.decision_trace_id is None:
        raise ValueError("DecisionTraceRecord must have decision_trace_id before report write")
    json_path, markdown_path = resolve_rationale_report_paths(
        trace=trace,
        artifact_root=artifact_root,
    )
    report_payload = build_rationale_report_payload(
        trace=trace,
        json_report_path=json_path.as_posix(),
        markdown_report_path=markdown_path.as_posix(),
        order_request=order_request,
        lifecycle_events=lifecycle_events,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(make_json_safe(report_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_rationale_report_markdown(report_payload),
        encoding="utf-8",
    )
    return replace(
        trace,
        json_report_path=json_path.as_posix(),
        markdown_report_path=markdown_path.as_posix(),
    )


def resolve_rationale_report_paths(
    *,
    trace: DecisionTraceRecord,
    artifact_root: Path,
) -> tuple[Path, Path]:
    """Resolve deterministic rationale report paths for one persisted trace."""
    if trace.decision_trace_id is None:
        raise ValueError("DecisionTraceRecord must have decision_trace_id before path resolution")
    trace_dir = artifact_root / trace.service_name / trace.execution_mode
    stem = str(trace.decision_trace_id)
    return trace_dir / f"{stem}.json", trace_dir / f"{stem}.md"


def build_rationale_report_payload(
    *,
    trace: DecisionTraceRecord,
    json_report_path: str,
    markdown_report_path: str,
    order_request: OrderRequest | None = None,
    lifecycle_events: Sequence[OrderLifecycleEvent] = (),
) -> dict[str, Any]:
    """Build a deterministic rationale report from the canonical trace plus linkage rows."""
    payload = trace.payload.model_dump(mode="json")
    return {
        "schema_version": RATIONALE_REPORT_SCHEMA_VERSION,
        "decision_trace_id": trace.decision_trace_id,
        "service_name": trace.service_name,
        "execution_mode": trace.execution_mode,
        "symbol": trace.symbol,
        "signal": trace.signal,
        "signal_row_id": trace.signal_row_id,
        "signal_interval_begin": to_rfc3339(trace.signal_interval_begin),
        "signal_as_of_time": to_rfc3339(trace.signal_as_of_time),
        "model_name": trace.model_name,
        "model_version": trace.model_version,
        "prediction": payload["prediction"],
        "signal_section": payload["signal"],
        "threshold_snapshot": payload.get("threshold_snapshot"),
        "regime_reason": payload.get("regime_reason"),
        "risk": payload.get("risk"),
        "blocked_trade": payload.get("blocked_trade"),
        "execution_intent": (
            None if order_request is None else _execution_intent_payload(order_request)
        ),
        "lifecycle_events": [
            _lifecycle_event_payload(event)
            for event in _sorted_lifecycle_events(lifecycle_events)
        ],
        "json_report_path": json_report_path,
        "markdown_report_path": markdown_report_path,
    }


def render_rationale_report_markdown(report_payload: dict[str, Any]) -> str:
    """Render a deterministic Markdown rationale report for one trace."""
    lines = [
        f"# Decision Trace {report_payload['decision_trace_id']}",
        "",
        "## Metadata",
        f"- service_name: `{report_payload['service_name']}`",
        f"- execution_mode: `{report_payload['execution_mode']}`",
        f"- symbol: `{report_payload['symbol']}`",
        f"- signal: `{report_payload['signal']}`",
        f"- signal_row_id: `{report_payload['signal_row_id']}`",
        f"- signal_interval_begin: `{report_payload['signal_interval_begin']}`",
        f"- signal_as_of_time: `{report_payload['signal_as_of_time']}`",
        f"- model_name: `{report_payload['model_name']}`",
        f"- model_version: `{report_payload['model_version']}`",
        "",
        "## Prediction",
    ]
    prediction = report_payload["prediction"]
    lines.extend(
        [
            f"- predicted_class: `{prediction['predicted_class']}`",
            f"- prob_up: `{prediction['prob_up']}`",
            f"- prob_down: `{prediction['prob_down']}`",
            f"- confidence: `{prediction['confidence']}`",
            "",
            "### Prediction Explanation",
        ]
    )
    lines.extend(_bullet_lines(prediction.get("prediction_explanation")))
    lines.extend(
        [
            "",
            "### Top Features",
        ]
    )
    top_features = prediction.get("top_features", [])
    if top_features:
        for feature in top_features:
            lines.append(
                "- "
                f"{feature['feature_name']}: value={feature['feature_value']}, "
                f"reference={feature['reference_value']}, "
                f"signed_contribution={feature['signed_contribution']}, "
                f"direction={feature['direction']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Signal",
        ]
    )
    lines.extend(_bullet_lines(report_payload["signal_section"]))
    lines.extend(
        [
            "",
            "## Threshold Snapshot",
        ]
    )
    lines.extend(_bullet_lines(report_payload.get("threshold_snapshot")))
    lines.extend(
        [
            "",
            "## Regime Reason",
        ]
    )
    lines.extend(_bullet_lines(report_payload.get("regime_reason")))
    lines.extend(
        [
            "",
            "## Risk",
        ]
    )
    lines.extend(_bullet_lines(report_payload.get("risk")))
    blocked_trade = report_payload.get("blocked_trade")
    if blocked_trade is not None:
        lines.extend(
            [
                "",
                "## Blocked Trade",
            ]
        )
        lines.extend(_bullet_lines(blocked_trade))
    execution_intent = report_payload.get("execution_intent")
    if execution_intent is not None:
        lines.extend(
            [
                "",
                "## Execution Intent",
            ]
        )
        lines.extend(_bullet_lines(execution_intent))
    lifecycle_events = report_payload.get("lifecycle_events", [])
    if lifecycle_events:
        lines.extend(
            [
                "",
                "## Lifecycle Events",
            ]
        )
        for event in lifecycle_events:
            lines.append(
                "- "
                f"{event['lifecycle_state']} at {event['event_time']}"
                f" reason_code={event['reason_code']}"
                f" external_status={event['external_status']}"
                f" broker_name={event['broker_name']}"
            )
    lines.extend(
        [
            "",
            "## Report Paths",
            f"- json_report_path: `{report_payload['json_report_path']}`",
            f"- markdown_report_path: `{report_payload['markdown_report_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_row_id_interval_begin(row_id: str):
    _, _, timestamp = row_id.partition("|")
    return parse_rfc3339(timestamp)


def _require_model_version(signal: SignalDecision) -> str:
    if signal.model_version is None:
        raise ValueError("SignalDecision.model_version is required for M14 decision traces")
    return signal.model_version


def _execution_intent_payload(order_request: OrderRequest) -> dict[str, Any]:
    return {
        "order_request_id": order_request.order_request_id,
        "decision_trace_id": order_request.decision_trace_id,
        "symbol": order_request.symbol,
        "action": order_request.action,
        "signal_row_id": order_request.signal_row_id,
        "target_fill_interval_begin": to_rfc3339(order_request.target_fill_interval_begin),
        "requested_notional": order_request.requested_notional,
        "approved_notional": order_request.approved_notional,
        "idempotency_key": order_request.idempotency_key,
        "model_name": order_request.model_name,
        "model_version": order_request.model_version,
        "confidence": order_request.confidence,
        "regime_label": order_request.regime_label,
        "regime_run_id": order_request.regime_run_id,
        "risk_outcome": order_request.risk_outcome,
        "risk_reason_codes": list(order_request.risk_reason_codes),
    }


def _lifecycle_event_payload(event: OrderLifecycleEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "decision_trace_id": event.decision_trace_id,
        "order_request_id": event.order_request_id,
        "lifecycle_state": event.lifecycle_state,
        "event_time": to_rfc3339(event.event_time),
        "reason_code": event.reason_code,
        "details": event.details,
        "external_order_id": event.external_order_id,
        "external_status": event.external_status,
        "account_id": event.account_id,
        "environment_name": event.environment_name,
        "broker_name": event.broker_name,
        "probe_policy_active": event.probe_policy_active,
        "probe_symbol": event.probe_symbol,
        "probe_qty": event.probe_qty,
    }


def _sorted_lifecycle_events(
    lifecycle_events: Sequence[OrderLifecycleEvent],
) -> list[OrderLifecycleEvent]:
    order = {
        "CREATED": 0,
        "SUBMITTED": 1,
        "ACCEPTED": 2,
        "PARTIALLY_FILLED": 3,
        "FILLED": 4,
        "REJECTED": 5,
        "CANCELED": 6,
        "FAILED": 7,
    }
    return sorted(
        lifecycle_events,
        key=lambda event: (
            event.event_time,
            order.get(event.lifecycle_state, 99),
            event.event_id or 0,
        ),
    )


def _bullet_lines(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return ["- none"]
    return [f"- {key}: `{value}`" for key, value in payload.items()]
