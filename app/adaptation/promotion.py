"""Evidence-based M19 promotion helpers for adaptive profiles and challengers."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from app.adaptation.config import PromotionThresholdConfig
from app.adaptation.schemas import AdaptivePerformanceWindow, AdaptivePromotionDecisionRecord
from app.common.time import utc_now


def decide_promotion(
    *,
    decision_id: str,
    target_type: str,
    target_id: str,
    incumbent_id: str | None,
    challenger: AdaptivePerformanceWindow | None,
    incumbent: AdaptivePerformanceWindow | None,
    thresholds: PromotionThresholdConfig,
    reliability_healthy: bool,
) -> AdaptivePromotionDecisionRecord:
    """Return one bounded promotion decision tied to realistic evaluation evidence."""
    if challenger is None:
        return AdaptivePromotionDecisionRecord(
            decision_id=decision_id,
            target_type=target_type,
            target_id=target_id,
            incumbent_id=incumbent_id,
            decision="HOLD",
            reason_codes=["NO_CHALLENGER_EVIDENCE"],
            summary_text="No challenger evidence was available for promotion.",
            decided_at=incumbent.window_end if incumbent is not None else utc_now(),
        )
    reason_codes: list[str] = []
    decision = "PROMOTE"
    if thresholds.require_reliability_healthy and not reliability_healthy:
        decision = "HOLD"
        reason_codes.append("RELIABILITY_DEGRADED")
    if challenger.trade_count < 1:
        decision = "HOLD"
        reason_codes.append("INSUFFICIENT_SAMPLE")
    if challenger.profit_factor < thresholds.min_profit_factor:
        decision = "REJECT"
        reason_codes.append("PROFIT_FACTOR_TOO_LOW")
    if challenger.win_rate < thresholds.min_win_rate:
        decision = "REJECT"
        reason_codes.append("WIN_RATE_TOO_LOW")
    if challenger.shadow_divergence_rate > thresholds.max_shadow_divergence_rate:
        decision = "REJECT"
        reason_codes.append("SHADOW_DIVERGENCE_TOO_HIGH")
    if challenger.blocked_trade_rate > thresholds.max_blocked_trade_rate:
        decision = "REJECT"
        reason_codes.append("BLOCKED_TRADE_RATE_TOO_HIGH")
    if incumbent is not None:
        pnl_delta = challenger.net_pnl_after_costs - incumbent.net_pnl_after_costs
        drawdown_delta = challenger.max_drawdown - incumbent.max_drawdown
        if pnl_delta < thresholds.min_net_pnl_delta_after_costs:
            decision = "REJECT"
            reason_codes.append("NET_PNL_DELTA_TOO_LOW")
        if drawdown_delta > thresholds.max_drawdown_degradation:
            decision = "REJECT"
            reason_codes.append("DRAWDOWN_DEGRADED")
        metrics_delta = {
            "net_pnl_after_costs": pnl_delta,
            "max_drawdown": drawdown_delta,
            "profit_factor": challenger.profit_factor - incumbent.profit_factor,
            "win_rate": challenger.win_rate - incumbent.win_rate,
        }
    else:
        metrics_delta = {
            "net_pnl_after_costs": challenger.net_pnl_after_costs,
            "max_drawdown": challenger.max_drawdown,
            "profit_factor": challenger.profit_factor,
            "win_rate": challenger.win_rate,
        }
    return AdaptivePromotionDecisionRecord(
        decision_id=decision_id,
        target_type=target_type,
        target_id=target_id,
        incumbent_id=incumbent_id,
        decision=decision,
        metrics_delta_json=metrics_delta,
        safety_checks_json={
            "reliability_healthy": reliability_healthy,
            "shadow_divergence_rate": challenger.shadow_divergence_rate,
            "blocked_trade_rate": challenger.blocked_trade_rate,
        },
        research_integrity_json={
            "trade_count": challenger.trade_count,
            "window_id": challenger.window_id,
            "window_type": challenger.window_type,
        },
        reason_codes=reason_codes or ["PROMOTION_CRITERIA_PASSED"],
        summary_text=(
            "Adaptive promotion passed bounded evidence checks."
            if decision == "PROMOTE"
            else "Adaptive promotion did not pass bounded evidence checks."
        ),
        decided_at=challenger.window_end,
    )
